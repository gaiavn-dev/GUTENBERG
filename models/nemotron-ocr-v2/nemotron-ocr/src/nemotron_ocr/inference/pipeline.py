# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nemotron_ocr.inference.encoders.recognizer_encoder import RecognitionTargetEncoder
from nemotron_ocr.inference.encoders.relational_encoder import RelationalTargetEncoder
from nemotron_ocr.inference.models.detector.fots_detector import FOTSDetector
from nemotron_ocr.inference.models.recognizer import TransformerRecognizer
from nemotron_ocr.inference.models.relational import GlobalRelationalModel
from nemotron_ocr.inference.post_processing.indirect_grid_sample import IndirectGridSample
from nemotron_ocr.inference.post_processing.data.text_region import TextBlock
from nemotron_ocr.inference.post_processing.quad_rectify import QuadRectify
from nemotron_ocr.inference.post_processing.research_ops import parse_relational_results, reorder_boxes
from nemotron_ocr.inference.pre_processing import interpolate_and_pad, pad_to_square
from huggingface_hub import hf_hub_download
from nemotron_ocr_cpp import quad_non_maximal_suppression, region_counts_to_indices, rrect_to_quads
from PIL import Image, ImageDraw, ImageFont
from torch import amp
from torchvision.io import read_image, decode_image
from torchvision.transforms.functional import convert_image_dtype

PAD_COLOR_CPU = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16)
PAD_COLOR = None  # Will be initialized on CUDA on first use
DEFAULT_INFER_LENGTH = 1024
DETECTOR_DOWNSAMPLE = 4
NMS_PROB_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
NMS_MAX_REGIONS = 0

MERGE_LEVELS = {"word", "sentence", "paragraph"}
DEFAULT_MERGE_LEVEL = "paragraph"

# HuggingFace repositories for downloading model weights
HF_REPO_ID = "nvidia/nemotron-ocr-v1"
HF_REPO_ID_V2 = "nvidia/nemotron-ocr-v2"
CHECKPOINT_FILES = ["detector.pth", "recognizer.pth", "relational.pth", "charset.txt", "model_config.json"]

# User-facing ``lang`` → (repo_id, path prefix inside repo)
LANG_HUB_PATH: Dict[str, Tuple[str, str]] = {
    "en": (HF_REPO_ID_V2, "v2_english"),
    "english": (HF_REPO_ID_V2, "v2_english"),
    "multi": (HF_REPO_ID_V2, "v2_multilingual"),
    "multilingual": (HF_REPO_ID_V2, "v2_multilingual"),
    "v1": (HF_REPO_ID, "checkpoints"),
    "legacy": (HF_REPO_ID, "checkpoints"),
}
DEFAULT_LANG = "multi"


class NemotronOCR:
    """
    A high-level pipeline for performing OCR on images.
    
    Model weights are automatically downloaded from Hugging Face Hub when no
    complete local checkpoint directory is provided. The default is Nemotron OCR
    **v2 multilingual** (``nvidia/nemotron-ocr-v2`` / ``v2_multilingual``).

    Automatically detects model parameters from model_config.json if available,
    otherwise falls back to defaults for backwards compatibility.

    Args:
        model_dir: Path to a local directory containing model checkpoints
            (``detector.pth``, ``recognizer.pth``, ``relational.pth``, ``charset.txt``).
            If provided and complete, this path is used and ``lang`` is ignored.
        infer_length: Resolution (in pixels) to which images are resized before
            being fed to the detector.  When None the value is read from
            ``model_config.json`` (key ``infer_length``), falling back to 1024.
        lang: Which checkpoint to fetch from Hugging Face when ``model_dir`` is
            missing or incomplete: ``"en"`` / ``"english"`` (v2 English from
            ``nvidia/nemotron-ocr-v2`` / ``v2_english``), ``"multi"`` / ``"multilingual"``
            (v2 multilingual from ``nvidia/nemotron-ocr-v2`` / ``v2_multilingual``, the
            default), or ``"v1"`` / ``"legacy"`` (original v1 Hub layout).
            When ``None``, **v2 multilingual** is downloaded.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        infer_length: Optional[int] = None,
        *,
        lang: Optional[str] = None,
        detector_only: bool = False,
        skip_relational: bool = False,
    ):
        self._detector_only = getattr(self, "_detector_only", detector_only)
        self._skip_relational = getattr(self, "_skip_relational", skip_relational)
        # If model_dir is provided and contains all required files, use it directly
        if model_dir is not None:
            local_path = Path(model_dir)
            if all((local_path / f).is_file() for f in CHECKPOINT_FILES):
                self._model_dir = local_path
            else:
                self._model_dir = self._download_checkpoints(lang=lang)
        else:
            self._model_dir = self._download_checkpoints(lang=lang)

        self._load_config(infer_length_override=infer_length)
        self._load_charset()
        self._load_models()
        self._initialize_processors()

    @staticmethod
    def _resolve_lang(lang: Optional[str]) -> str:
        """Return canonical lang key for Hub download (default: v2 multilingual)."""
        if lang is None:
            return DEFAULT_LANG
        key = str(lang).strip().lower()
        if key not in LANG_HUB_PATH:
            allowed = ", ".join(sorted(set(LANG_HUB_PATH)))
            raise ValueError(f"lang must be one of: {allowed}; got {lang!r}")
        return key

    @staticmethod
    def _download_checkpoints(lang: Optional[str] = None) -> Path:
        """Download model checkpoints from HuggingFace Hub (cached locally after first download)."""
        key = NemotronOCR._resolve_lang(lang)
        repo_id, subdir = LANG_HUB_PATH[key]
        downloaded_path = None
        for filename in CHECKPOINT_FILES:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{subdir}/{filename}",
            )
        return Path(downloaded_path).parent

    def _load_config(self, infer_length_override: Optional[int] = None):
        """Load model configuration from model_config.json if available."""
        config_path = self._model_dir / "model_config.json"
        self.config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
        
        # Get parameters from config or use defaults for backwards compatibility
        self.scope = self.config.get('scope', 512)
        self.max_width = self.config.get('max_width', 32)
        self.num_tokens = self.config.get('num_tokens', 858)

        # Inference resolution: constructor arg > config file > default
        if infer_length_override is not None:
            self.infer_length = infer_length_override
        else:
            self.infer_length = self.config.get('infer_length', DEFAULT_INFER_LENGTH)

    def _load_charset(self):
        with open(self._model_dir / "charset.txt", "r", encoding="utf-8") as file:
            self.charset = json.load(file)
        
        # Calculate num_tokens from charset if not in config
        if 'num_tokens' not in self.config:
            if isinstance(self.charset, list):
                self.num_tokens = len(self.charset) + 3  # +3 for special tokens
            else:
                self.num_tokens = 858  # Default fallback

    def _load_models(self):
        """Loads models into memory.  Respects ``_detector_only`` and
        ``_skip_relational`` flags to avoid loading unused components."""
        backbone = self.config.get('backbone', 'regnet_y_8gf')
        self.detector = FOTSDetector(
            coordinate_mode="RBOX", 
            backbone=backbone, 
            scope=self.scope,
            verbose=False
        )
        self.detector.load_state_dict(torch.load(self._model_dir / "detector.pth", weights_only=True), strict=True)
        self.detector = self.detector.cuda()
        self.detector.eval()
        self.detector.inference_mode = True

        if self._detector_only:
            torch.backends.cudnn.benchmark = True
            return

        # Auto-detect dimensions from weights if not in config
        recognizer_state = torch.load(self._model_dir / "recognizer.pth", map_location='cpu', weights_only=True)
        if 'num_tokens' not in self.config and 'classifier.weight' in recognizer_state:
            self.num_tokens = recognizer_state['classifier.weight'].shape[0]
        if 'max_width' not in self.config and 'position_encoding' in recognizer_state:
            self.max_width = recognizer_state['position_encoding'].shape[-1]

        use_pre_norm = self.config.get('has_pre_norm', 'pre_norm.weight' in recognizer_state)
        use_final_norm = self.config.get('has_tx_norm', 'tx.norm.weight' in recognizer_state)
        
        variant = self.config.get('recognizer_variant', 'prenorm' if (use_pre_norm or use_final_norm) else 'standard')
        # if variant != 'standard':
        #     print(f"Loading recognizer variant: {variant} (pre_norm={use_pre_norm}, final_norm={use_final_norm})")

        recog_depth = self.config.get('depth', 128)
        recog_num_layers = self.config.get('num_layers', 3)
        recog_nhead = self.config.get('nhead', 8)
        recog_dim_feedforward = self.config.get('dim_feedforward', None)
        recog_norm_first = self.config.get('norm_first', False)
        
        self.recognizer = TransformerRecognizer(
            nic=self.detector.num_features[-1], 
            num_tokens=self.num_tokens, 
            max_width=self.max_width,
            use_pre_norm=use_pre_norm,
            use_final_norm=use_final_norm,
            norm_first=recog_norm_first,
            depth=recog_depth,
            num_layers=recog_num_layers,
            nhead=recog_nhead,
            dim_feedforward=recog_dim_feedforward,
        )
        self.recognizer.load_state_dict(recognizer_state, strict=True)
        self.recognizer = self.recognizer.cuda()
        self.recognizer.eval()
        self.recognizer.inference_mode = True

        if self._skip_relational:
            torch.backends.cudnn.benchmark = True
            return

        self.relational = GlobalRelationalModel(
            num_input_channels=self.detector.num_features,
            recog_feature_depth=self.recognizer.feature_depth,
            dropout=0.1,
            k=16,
            num_layers=4,
        )
        self.relational.load_state_dict(torch.load(self._model_dir / "relational.pth", weights_only=True), strict=True)
        self.relational = self.relational.cuda()
        self.relational.eval()
        self.relational.inference_mode = True
        
        torch.backends.cudnn.benchmark = True

    def _initialize_processors(self):
        """Initializes helper classes for pre/post-processing.
        Respects ``_detector_only`` and ``_skip_relational`` flags."""
        self.grid_sampler = IndirectGridSample()

        if self._detector_only:
            return

        self.recognizer_quad_rectifier = QuadRectify(8, self.max_width)
        self.recog_encoder = RecognitionTargetEncoder(
            charset=self.charset,
            input_size=[self.infer_length, self.infer_length],
            sequence_length=self.max_width,
            amp_opt=2,
            combine_duplicates=False,
            is_train=False,
        )

        if self._skip_relational:
            return

        self.relational_quad_rectifier = QuadRectify(2, 3, isotropic=False)
        self.relation_encoder = RelationalTargetEncoder(
            input_size=[self.infer_length, self.infer_length], amp_opt=2, is_train=False
        )

    def __call__(self, image, merge_level=DEFAULT_MERGE_LEVEL, visualize=False, include_invalid=False):
        """
        Performs OCR on a single image.

        Args:
            image (str | bytes | np.ndarray | Image.Image): The input image. Can be a:
                - file path (str)
                - base64 encoded string (bytes)
                - NumPy array (H, W, C)
                - In-memory byte stream (io.BytesIO)
            merge_level (str): The granularity of text merging ('word', 'sentence', 'paragraph').
            visualize (bool): If True, saves an annotated image.
            include_invalid (bool): If True, include text regions marked as invalid.

        Returns:
            list: A list of prediction dictionaries.
        """
        image_tensor = self._load_image_to_tensor(image)

        predictions = self._process_tensor(image_tensor, merge_level, include_invalid=include_invalid)

        original_path = image if isinstance(image, str) and Path(image).is_file() else None
        if visualize:
            if original_path is None:
                raise ValueError("Visualization is only supported when the input is a file path.")
            self._save_annotated_image(original_path, predictions)

        return predictions

    def _load_image_to_tensor(self, image):
        """
        Loads an image from various sources and converts it to a standardized tensor.
        """
        if isinstance(image, str):
            image_path = Path(image)
            if not image_path.is_file():
                raise FileNotFoundError(f"Input string is not a valid file path: {image}")
            img_tensor = read_image(str(image_path), mode="RGB")

        elif isinstance(image, bytes):
            try:
                img_bytes = base64.b64decode(image)
                img_tensor = decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode="RGB")
            except (ValueError, TypeError, base64.binascii.Error) as e:
                raise ValueError("Input is not a valid base64-encoded image.") from e

        elif isinstance(image, np.ndarray):
            # PyTorch expects CHW, NumPy use HWC, so we permute
            if image.ndim == 2:  # Handle grayscale by stacking
                image = np.stack([image] * 3, axis=-1)
            # Handle RGBA images by stripping the alpha channel
            if image.shape[2] == 4:
                image = image[..., :3]
            img_tensor = torch.from_numpy(image).permute(2, 0, 1)

        elif isinstance(image, io.BytesIO):
            image.seek(0)
            img_bytes = image.getvalue()
            img_tensor = decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode="RGB")

        else:
            raise TypeError(
                f"Unsupported input type: {type(image)}. "
                "Supported types are file path (str), base64 (str/bytes), NumPy array, and io.BytesIO"
            )

        return convert_image_dtype(img_tensor, dtype=torch.float16)

    def _process_tensor(self, image_tensor, merge_level, include_invalid=False):
        """
        Runs the core OCR inference pipeline on a standardized image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor
            merge_level: Text merging granularity
            include_invalid: If True, don't prune invalid text regions
        """
        if merge_level not in MERGE_LEVELS:
            raise ValueError(f"Invalid merge level: {merge_level}. Must be one of {MERGE_LEVELS}.")

        original_shape = image_tensor.shape[1:]
        padded_length = max(original_shape)

        # Ensure PAD_COLOR is on CUDA (lazy init for efficiency)
        global PAD_COLOR
        if PAD_COLOR is None or not PAD_COLOR.is_cuda:
            PAD_COLOR = PAD_COLOR_CPU.cuda()

        padded_image = interpolate_and_pad(
            pad_to_square(image_tensor, padded_length, how="bottom_right").unsqueeze(0),
            PAD_COLOR,
            self.infer_length,
        )

        with amp.autocast("cuda", enabled=True), torch.no_grad():
            # padded_image is already on CUDA from interpolate_and_pad
            det_conf, _, det_rboxes, det_feature_3 = self.detector(padded_image)

        with amp.autocast("cuda", enabled=True), torch.no_grad():
            e2e_det_conf = torch.sigmoid(det_conf)
            e2e_det_coords = rrect_to_quads(det_rboxes.float(), DETECTOR_DOWNSAMPLE)

            quads, confidence, region_counts = quad_non_maximal_suppression(
                e2e_det_coords,
                e2e_det_conf,
                prob_threshold=NMS_PROB_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD,
                kernel_height=2,
                kernel_width=3,
                max_regions=NMS_MAX_REGIONS,
                verbose=False,
            )[:3]

        # Move quads to CUDA once for reuse
        quads_cuda = quads.cuda() if quads.numel() > 0 and not quads.is_cuda else quads

        if quads.shape[0] == 0:
            rec_rectified_quads = torch.empty(0, 128, 8, self.max_width, dtype=torch.float32, device=padded_image.device)
            rel_rectified_quads = torch.empty(0, 128, 2, 3, dtype=torch.float32, device=padded_image.device)
        else:
            rec_rectified_quads = self.recognizer_quad_rectifier(
                quads_cuda.detach(), padded_image.shape[2], padded_image.shape[3]
            )
            rel_rectified_quads = self.relational_quad_rectifier(
                quads_cuda.detach(), padded_image.shape[2], padded_image.shape[3]
            )

            input_indices = region_counts_to_indices(region_counts, quads.shape[0])
            input_indices_cuda = input_indices.cuda() if not input_indices.is_cuda else input_indices

            # Grid sampling requires float32
            det_features_f32 = det_feature_3.float()
            rec_rectified_quads = self.grid_sampler(det_features_f32, rec_rectified_quads.float(), input_indices_cuda)
            rel_rectified_quads = self.grid_sampler(
                det_features_f32,
                rel_rectified_quads,
                input_indices_cuda,
            )

        if rec_rectified_quads.shape[0] == 0:
            rec_output = torch.empty(0, self.max_width, self.num_tokens, dtype=torch.float16, device=padded_image.device)
            rec_features = torch.empty(0, self.max_width, 256, dtype=torch.float16, device=padded_image.device)
        else:
            with amp.autocast("cuda", enabled=True), torch.no_grad():
                # rec_rectified_quads is already on CUDA from grid_sampler
                rec_output, rec_features = self.recognizer(rec_rectified_quads)

        predictions = []

        if region_counts.sum() > 0:
            # All inputs should already be on CUDA from earlier processing
            rel_output = self.relational(
                rel_rectified_quads,
                quads_cuda,
                region_counts.cpu(),
                rec_features,
            )
            words, lines, line_var = (
                rel_output["words"],
                rel_output["lines"],
                rel_output["line_log_var_unc"],
            )

            with amp.autocast("cuda", enabled=True), torch.no_grad():
                words = [F.softmax(r, dim=1, dtype=torch.float32)[:, 1:] for r in words]

                rec_logits = rec_output.float()
                log_norm = rec_logits.logsumexp(dim=2, keepdim=True)
                seq_ids = rec_logits.argmax(dim=2)
                seq_probs = (rec_logits.gather(2, seq_ids.unsqueeze(2)) - log_norm).squeeze(2).exp()

                output = {
                    "sequences": seq_ids,
                    "sequence_probs": seq_probs,
                    "region_counts": region_counts,
                    "quads": quads,
                    "raw_detector_confidence": e2e_det_conf,
                    "confidence": confidence,
                    "relations": words,
                    "line_relations": lines,
                    "line_rel_var": line_var,
                    "fg_colors": None,
                    "fonts": None,
                    "tt_log_var_uncertainty": None,
                    "e2e_recog_features": rec_features,
                }

            quads = output["quads"]

            lengths = [padded_length / self.infer_length] * region_counts.item()

            lengths_tensor = torch.tensor(lengths, dtype=torch.float32, device=quads.device).view(quads.shape[0], 1, 1)

            quads *= lengths_tensor

            # TODO: Incorporate the quad scale factor
            batch = self.recog_encoder.convert_targets_to_labels(output, image_size=None, is_gt=False)
            relation_batch = self.relation_encoder.convert_targets_to_labels(output, image_size=None, is_gt=False)

            for example, rel_example in zip(batch, relation_batch):
                example.relation_graph = rel_example.relation_graph
                if not include_invalid:
                    example.prune_invalid_relations()

            for example in batch:
                if example.relation_graph is None:
                    continue
                for paragraph in example.relation_graph:
                    block = []
                    for line in paragraph:
                        for relational_idx in line:
                            block.append(example[relational_idx])
                    if block:
                        example.blocks.append(TextBlock(block))

            for example in batch:
                for text_region in example:
                    text_region.region = text_region.region.vertices

            for example in batch:
                boxes, texts, scores = parse_relational_results(example, level=merge_level)
                boxes, texts, scores = reorder_boxes(boxes, texts, scores, mode="top_left", dbscan_eps=10)

                orig_h, orig_w = original_shape

                if len(boxes) == 0:
                    boxes = ["nan"]
                    texts = ["nan"]
                    scores = ["nan"]
                else:
                    # Convert to numpy array and reshape to (N, 4, 2) for easier processing
                    boxes_array = np.array(boxes).reshape(-1, 4, 2)

                    # Divide X coordinates by orig_w and Y coordinates by orig_h
                    boxes_array[:, :, 0] = boxes_array[:, :, 0] / orig_w  # X coordinates
                    boxes_array[:, :, 1] = boxes_array[:, :, 1] / orig_h  # Y coordinates
                    boxes = boxes_array.astype(np.float16).tolist()

                for box, text, conf in zip(boxes, texts, scores):
                    if box == "nan":
                        break
                    predictions.append(
                        {
                            "text": text,
                            "confidence": conf,
                            "left": min(p[0] for p in box),
                            "upper": max(p[1] for p in box),
                            "right": max(p[0] for p in box),
                            "lower": min(p[1] for p in box),
                        }
                    )

        return predictions

    def _save_annotated_image(self, image_path, predictions):
        """Saves a copy of the image with bounding boxes overlaid."""
        output_path = os.path.splitext(image_path)[0] + "-annotated" + os.path.splitext(image_path)[1]

        pil_image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(pil_image)

        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

        img_width, img_height = pil_image.size

        color = (255, 0, 0)

        for pred in predictions:
            if isinstance(pred.get("left"), str) and pred["left"] == "nan":
                continue

            left = int(pred["left"] * img_width)
            right = int(pred["right"] * img_width)
            upper = int(pred["upper"] * img_height)
            lower = int(pred["lower"] * img_height)

            confidence = pred["confidence"]
            text = pred["text"]

            draw.rectangle([left, lower, right, upper], outline=color, width=2)

            display_text = f"{text}"
            conf_text = f"({confidence:.2f})"

            text_y = max(0, upper - 25)

            text_bbox = draw.textbbox((left, text_y), display_text, font=font)
            conf_bbox = draw.textbbox((left, text_y + 18), conf_text, font=small_font)

            draw.rectangle(
                [
                    text_bbox[0] - 2,
                    text_bbox[1] - 2,
                    text_bbox[2] + 2,
                    text_bbox[3] + 2,
                ],
                fill=(255, 255, 255, 180),
                outline=color,
            )
            draw.rectangle(
                [
                    conf_bbox[0] - 2,
                    conf_bbox[1] - 2,
                    conf_bbox[2] + 2,
                    conf_bbox[3] + 2,
                ],
                fill=(255, 255, 255, 180),
                outline=color,
            )

            draw.text((left, text_y), display_text, fill=color, font=font)
            draw.text((left, text_y + 18), conf_text, fill=color, font=small_font)

        pil_image.save(output_path)

        print(f"Annotated image saved to: {output_path}")
        print(
            f"Total predictions overlaid: {len([p for p in predictions if not (isinstance(p.get('left'), str) and p['left'] == 'nan')])}"
        )
