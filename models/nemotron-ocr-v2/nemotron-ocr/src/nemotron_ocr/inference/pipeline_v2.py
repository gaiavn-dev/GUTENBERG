# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched OCR inference pipeline.

Extends :class:`NemotronOCR` with:
  - Multi-image detector batching
  - Chunked recognizer with early argmax (low VRAM)
  - Pre-NMS centerness + peak filter for consistent speed
  - Detector-only and skip-relational inference modes
  - Optional per-phase timing via ``verbose_post``
"""

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

from nemotron_ocr.inference.pipeline import (
    NemotronOCR,
    DETECTOR_DOWNSAMPLE,
    MERGE_LEVELS,
    DEFAULT_MERGE_LEVEL,
    NMS_PROB_THRESHOLD,
    NMS_IOU_THRESHOLD,
    NMS_MAX_REGIONS,
)
from nemotron_ocr.inference.pre_processing import interpolate_and_pad, pad_to_square
from nemotron_ocr.inference.post_processing.data.text_region import TextBlock
from nemotron_ocr.inference.post_processing.research_ops import (
    parse_relational_results,
    reorder_boxes,
)
from nemotron_ocr_cpp import (
    quad_non_maximal_suppression,
    region_counts_to_indices,
    rrect_to_quads,
)

logger = logging.getLogger(__name__)

# Fallback defaults (used when the parent class doesn't set these).
_DEFAULT_PAD_COLOR = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float16)
_DEFAULT_INFER_LENGTH = 1024
_DEFAULT_MAX_WIDTH = 32
_DEFAULT_NUM_TOKENS = 858


class NemotronOCRV2(NemotronOCR):
    """Batched OCR inference pipeline.

    Inherits model loading from :class:`NemotronOCR` and adds batched
    detection, chunked recognition, and optional relational grouping.

    Args:
        detector_max_batch_size: Max images per detector forward pass.
        recognizer_chunk_size: Regions per recognizer forward call.
            Padded to this size for consistent CUDA kernel shapes.
        relational_chunk_size: Pad-to multiple for per-image region
            counts inside the relational model.
        use_prefilter: Apply centerness + local-peak filter before NMS
            to prevent O(n^2) slowdowns on dense confidence maps.
        prefilter_peak_kernel: Kernel size for the local-max filter.
        detector_only: Load only the detector; ``__call__`` returns
            bounding boxes without text.
        skip_relational: Skip the relational model; returns per-word
            text without reading-order grouping.
        pad_color: RGB padding colour as ``[R, G, B]`` floats in [0, 1].
        pad_how: ``"bottom_right"`` or ``"center"`` padding placement.
        infer_length: Detector input resolution in pixels (default 1024).
        verbose_post: When True, CUDA-syncs each phase and emits
            per-batch timing via ``logger.info``.
        **kwargs: Forwarded to :class:`NemotronOCR` (``model_dir``, etc.).
    """

    def __init__(
        self,
        *,
        detector_max_batch_size: int = 8,
        recognizer_chunk_size: int = 128,
        relational_chunk_size: int = 128,
        use_prefilter: bool = True,
        prefilter_peak_kernel: int = 3,
        detector_only: bool = False,
        skip_relational: bool = False,
        pad_color=None,
        pad_how=None,
        infer_length=None,
        verbose_post: bool = False,
        **kwargs,
    ):
        # Set mode flags before super().__init__ so _load_models can see them
        self._detector_only = detector_only
        self._skip_relational = skip_relational

        super().__init__(**kwargs)

        self.detector_max_batch_size = detector_max_batch_size
        self.recognizer_chunk_size = recognizer_chunk_size
        self.relational_chunk_size = relational_chunk_size
        self._use_prefilter = use_prefilter
        self._prefilter_peak_kernel = prefilter_peak_kernel
        self._verbose_post = verbose_post

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # ── pad_color ────────────────────────────────────────────────
        if pad_color is not None:
            self._pad_color_cpu = torch.tensor(pad_color, dtype=torch.float16)
            self._pad_color = None  # reset lazy CUDA cache
        if not hasattr(self, "_pad_color_cpu"):
            self._pad_color_cpu = _DEFAULT_PAD_COLOR.clone()
        if not hasattr(self, "_pad_color"):
            self._pad_color = None

        # ── pad_how ──────────────────────────────────────────────────
        if pad_how is not None:
            self._pad_how = pad_how
        if not hasattr(self, "_pad_how"):
            self._pad_how = "bottom_right"

        # ── infer_length ─────────────────────────────────────────────
        if infer_length is not None:
            self.infer_length = infer_length
        if not hasattr(self, "infer_length"):
            self.infer_length = _DEFAULT_INFER_LENGTH

        # ── recognizer dims (may already be set by a local parent) ───
        if not hasattr(self, "max_width"):
            self.max_width = _DEFAULT_MAX_WIDTH
        if not hasattr(self, "num_tokens"):
            self.num_tokens = _DEFAULT_NUM_TOKENS

        if hasattr(self, "relational"):
            self.relational.chunk_size = relational_chunk_size

        if verbose_post and hasattr(self, "relation_encoder") and hasattr(self.relation_encoder, "_verbose"):
            self.relation_encoder._verbose = True

        if not self._detector_only and hasattr(self, "recognizer"):
            self._pad_classifier_for_alignment(64)

    # ------------------------------------------------------------------
    # Tensor-core alignment
    # ------------------------------------------------------------------

    def _pad_classifier_for_alignment(self, alignment: int = 64):
        """Pad the recognizer's classifier output dim to a multiple of *alignment*.

        This makes the final Linear matmul hit efficient tensor-core tile sizes.
        Extra output positions get ``bias = -inf`` so they never affect argmax
        or softmax over the real vocabulary.
        """
        cls = self.recognizer.classifier
        real = cls.out_features
        padded = ((real + alignment - 1) // alignment) * alignment
        if padded == real:
            return

        new_cls = nn.Linear(cls.in_features, padded, bias=cls.bias is not None)
        with torch.no_grad():
            new_cls.weight[:real] = cls.weight
            new_cls.weight[real:] = 0
            if cls.bias is not None:
                new_cls.bias[:real] = cls.bias
                new_cls.bias[real:] = float("-inf")

        new_cls = new_cls.to(device=cls.weight.device, dtype=cls.weight.dtype)
        self.recognizer.classifier = new_cls
        logger.info(
            "Padded recognizer classifier %d -> %d (alignment=%d)",
            real, padded, alignment,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        images,
        merge_level: str = DEFAULT_MERGE_LEVEL,
        include_invalid: bool = False,
    ):
        """Run OCR on one or more images.

        Args:
            images: A single image or a list of images.  Each image can be
                any type accepted by :meth:`_load_image_to_tensor` (file path,
                numpy array, bytes, BytesIO).
            merge_level: Text merging granularity
                (``'word'``, ``'sentence'``, ``'paragraph'``).
            include_invalid: If True, include low-confidence text regions.

        Returns:
            ``list[list[dict]]`` — one prediction list per input image.
            If a single (non-list) image was passed, returns a flat
            ``list[dict]`` for backwards compatibility.
        """
        single = not isinstance(images, list)
        if single:
            images = [images]

        if merge_level not in MERGE_LEVELS:
            raise ValueError(
                f"Invalid merge level: {merge_level}. Must be one of {MERGE_LEVELS}."
            )

        results = self._process_batch(images, merge_level, include_invalid)
        return results[0] if single else results

    # ------------------------------------------------------------------
    # Phase 1 — Preprocess
    # ------------------------------------------------------------------

    def _preprocess_batch(self, images, timings=None):
        """Load, pad, resize all images and return a stacked GPU tensor.

        Transfers uint8 images to GPU first (smaller PCIe payload), then
        converts dtype and pads/resizes entirely on device.

        Returns:
            resized_batch: (N, C, H_infer, W_pad) on CUDA
            original_shapes: list of (H, W) per image
            padded_lengths: list of max(H, W) per image
        """
        if self._pad_color is None or not self._pad_color.is_cuda:
            self._pad_color = self._pad_color_cpu.cuda()

        t_load = 0.0
        t_gpu = 0.0
        resized_list = []
        original_shapes = []
        padded_lengths = []

        for img in images:
            t0 = time.perf_counter()
            tensor = self._load_image_to_tensor_uint8(img)
            t_load += time.perf_counter() - t0

            h, w = tensor.shape[1], tensor.shape[2]
            original_shapes.append((h, w))
            padded_length = max(h, w)
            padded_lengths.append(padded_length)

            t0 = time.perf_counter()
            tensor_gpu = tensor.to("cuda", non_blocking=True)
            tensor_gpu = tensor_gpu.to(torch.float16).div_(255.0)

            padded = pad_to_square(tensor_gpu, padded_length, how=self._pad_how)
            del tensor_gpu

            resized = interpolate_and_pad(
                padded.unsqueeze(0), self._pad_color, self.infer_length,
            ).squeeze(0)
            del padded
            resized_list.append(resized)
            t_gpu += time.perf_counter() - t0

        resized_batch = torch.stack(resized_list, dim=0)
        del resized_list

        if timings is not None:
            timings["img_load"] = t_load * 1000
            timings["gpu_preproc"] = t_gpu * 1000

        return resized_batch, original_shapes, padded_lengths

    @staticmethod
    def _load_image_to_tensor_uint8(image):
        """Load image and return a uint8 CHW tensor on CPU.

        Accepts file paths, numpy arrays, torch tensors, base64 bytes,
        or BytesIO objects.
        """
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 2:
                t = t.unsqueeze(0).expand(3, -1, -1)
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                if t.shape[0] == 4:
                    t = t[:3]
                if t.dtype != torch.uint8:
                    t = (t * 255).clamp(0, 255).to(torch.uint8) if t.is_floating_point() else t.to(torch.uint8)
                return t
            if t.ndim == 3 and t.shape[2] in (1, 3, 4):
                t = t.permute(2, 0, 1)
                if t.shape[0] == 4:
                    t = t[:3]
                if t.dtype != torch.uint8:
                    t = (t * 255).clamp(0, 255).to(torch.uint8) if t.is_floating_point() else t.to(torch.uint8)
                return t
            raise ValueError(f"Unsupported tensor shape: {image.shape}")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.shape[2] == 4:
                image = image[..., :3]
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            return torch.from_numpy(image).permute(2, 0, 1)

        from torchvision.io import read_image, decode_image
        if isinstance(image, (str, os.PathLike)):
            return read_image(str(image), mode="RGB")
        if isinstance(image, bytes):
            import base64
            img_bytes = base64.b64decode(image)
            return decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode="RGB")
        if hasattr(image, 'read'):
            image.seek(0)
            return decode_image(torch.frombuffer(image.getvalue(), dtype=torch.uint8), mode="RGB")

        raise TypeError(f"Unsupported input type: {type(image)}")

    # ------------------------------------------------------------------
    # Phase 2 — Detector (chunked)
    # ------------------------------------------------------------------

    def _run_detector_batched(self, resized_batch):
        """Run detector in chunks of ``detector_max_batch_size``.

        Returns:
            det_conf: (N, H/ds, W/ds)
            det_rboxes: (N, H/ds, W/ds, 5)
            det_features: (N, C, H/ds, W/ds)
        """
        n = resized_batch.shape[0]
        conf_parts, rbox_parts, feat_parts = [], [], []

        with amp.autocast("cuda", enabled=True), torch.inference_mode():
            for start in range(0, n, self.detector_max_batch_size):
                end = min(start + self.detector_max_batch_size, n)
                chunk = resized_batch[start:end]
                c, _, r, f = self.detector(chunk)
                conf_parts.append(c)
                rbox_parts.append(r)
                feat_parts.append(f)

        if len(conf_parts) == 1:
            return conf_parts[0], rbox_parts[0], feat_parts[0]
        return (
            torch.cat(conf_parts, dim=0),
            torch.cat(rbox_parts, dim=0),
            torch.cat(feat_parts, dim=0),
        )

    # ------------------------------------------------------------------
    # Phase 3 — NMS
    # ------------------------------------------------------------------

    def _prefilter_detections(self, det_conf, det_rboxes):
        """Reduce dense confidence maps before NMS using centerness + local peaks.

        Computes FCOS-style centerness from the detector's TRBL predictions
        to suppress edge pixels, then keeps only local maxima.  This prevents
        the O(n^2) NMS adjacency blowup on images with large text regions
        while preserving all real detection peaks.
        """
        with torch.inference_mode():
            d_top = det_rboxes[..., 0].float()
            d_right = det_rboxes[..., 1].float()
            d_bottom = det_rboxes[..., 2].float()
            d_left = det_rboxes[..., 3].float()

            lr_min = torch.minimum(d_left, d_right)
            lr_max = torch.maximum(d_left, d_right).clamp(min=1.0)
            tb_min = torch.minimum(d_top, d_bottom)
            tb_max = torch.maximum(d_top, d_bottom).clamp(min=1.0)

            centerness = torch.sqrt((lr_min / lr_max) * (tb_min / tb_max))

            conf_sigmoid = torch.sigmoid(det_conf.float())
            adjusted = conf_sigmoid * centerness

            k = self._prefilter_peak_kernel
            pad = k // 2
            adj_4d = adjusted.unsqueeze(1)
            pooled = F.max_pool2d(adj_4d, k, stride=1, padding=pad)
            pooled = pooled[:, 0, :det_conf.shape[1], :det_conf.shape[2]]

            peaks = (adjusted == pooled) & (adjusted > NMS_PROB_THRESHOLD)

            filtered = det_conf.clone()
            filtered[~peaks] = -100.0

        return filtered

    def _run_nms(self, det_conf, det_rboxes):
        """Sigmoid + rrect_to_quads + NMS.

        Returns:
            (quads, confidence, region_counts, e2e_det_conf) or None if
            zero detections.
        """
        with torch.inference_mode():
            e2e_det_conf = torch.sigmoid(det_conf)

        with amp.autocast("cuda", enabled=True), torch.inference_mode():
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

        if quads.shape[0] == 0:
            return None

        return quads, confidence, region_counts, e2e_det_conf

    # ------------------------------------------------------------------
    # Phase 4 — Quad rectify + grid sample
    # ------------------------------------------------------------------

    def _run_rectify_and_sample(self, quads, region_counts, det_features):
        """Compute sampling grids and extract region features.

        Returns:
            quads_cuda, rec_quads, rel_quads, region_counts_cpu, input_indices_cuda
        """
        quads_cuda = quads

        h = det_features.shape[2] * DETECTOR_DOWNSAMPLE
        w = det_features.shape[3] * DETECTOR_DOWNSAMPLE

        with torch.no_grad():
            rec_rectified = self.recognizer_quad_rectifier(quads_cuda, h, w)

            input_indices = region_counts_to_indices(region_counts, quads.shape[0])
            input_indices_cuda = input_indices.cuda(non_blocking=True)
            region_counts_cpu = region_counts.cpu()

            det_fp32 = det_features.float()
            rec_quads = self.grid_sampler(det_fp32, rec_rectified, input_indices_cuda)

            rel_quads = None
            if not self._skip_relational:
                rel_rectified = self.relational_quad_rectifier(quads_cuda, h, w)
                rel_quads = self.grid_sampler(det_fp32, rel_rectified, input_indices_cuda)

        return quads_cuda, rec_quads, rel_quads, region_counts_cpu

    # ------------------------------------------------------------------
    # Phase 5 — Recognizer (chunked)
    # ------------------------------------------------------------------

    def _run_recognizer_chunked(self, rec_quads):
        """Run recognizer in fixed-size region chunks.

        Every chunk is padded to exactly ``recognizer_chunk_size`` regions
        so CUDA always sees the same tensor shape.

        Computes argmax + softmax probability of the winning token inside
        each chunk loop and discards the full logits immediately, reducing
        peak VRAM from ``[N, T, num_tokens]`` to ``[chunk_size, T, num_tokens]``.

        Returns:
            rec_ids: (N_regions, T) int64 — argmax token indices
            rec_probs: (N_regions, T) float32 — softmax probability of winning token
            rec_features: (N_regions, T, D) — transformer features for relational model
        """
        n = rec_quads.shape[0]
        if n == 0:
            device = rec_quads.device
            return (
                torch.empty(0, self.max_width, dtype=torch.int64, device=device),
                torch.empty(0, self.max_width, dtype=torch.float32, device=device),
                torch.empty(0, self.max_width, self.recognizer.feature_depth, dtype=torch.float16, device=device),
            )

        cs = self.recognizer_chunk_size
        ids_parts, probs_parts, feat_parts = [], [], []
        with amp.autocast("cuda", enabled=True), torch.inference_mode():
            for start in range(0, n, cs):
                end = min(start + cs, n)
                chunk = rec_quads[start:end].half()
                real_n = chunk.shape[0]
                if real_n < cs:
                    pad = torch.zeros(cs - real_n, *chunk.shape[1:], dtype=chunk.dtype, device=chunk.device)
                    chunk = torch.cat([chunk, pad], dim=0)
                logits, feats = self.recognizer(chunk)

                ids = logits.argmax(dim=2)
                probs = torch.softmax(logits, dim=2).gather(
                    2, ids.unsqueeze(2)
                ).squeeze(2).float()

                ids_parts.append(ids[:real_n])
                probs_parts.append(probs[:real_n])
                feat_parts.append(feats[:real_n])

        def _cat_or_single(parts):
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        return _cat_or_single(ids_parts), _cat_or_single(probs_parts), _cat_or_single(feat_parts)

    # ------------------------------------------------------------------
    # Phase 6 — Relational model + build output dict
    # ------------------------------------------------------------------

    def _run_relational_and_build_output(
        self,
        rel_quads,
        quads_cuda,
        region_counts,
        region_counts_cpu,
        rec_features,
        rec_ids,
        rec_probs,
        e2e_det_conf,
        confidence,
        padded_lengths,
    ):
        """Run relational model and assemble the output dict.

        Accepts pre-computed argmax ids and per-token probabilities from the
        recognizer (avoids materializing the full logits tensor).
        """
        with amp.autocast("cuda", enabled=True), torch.inference_mode():
            rel_output = self.relational(
                rel_quads,
                quads_cuda,
                region_counts_cpu,
                rec_features,
            )
        words = rel_output["words"]
        lines = rel_output["lines"]
        line_var = rel_output["line_log_var_unc"]

        with amp.autocast("cuda", enabled=True), torch.inference_mode():
            words = [F.softmax(r, dim=1, dtype=torch.float32)[:, 1:] for r in words]

            # Vectorized quad scaling
            scale_factors = torch.as_tensor(
                padded_lengths, dtype=torch.float32, device=quads_cuda.device,
            ) / float(self.infer_length)
            counts_long = region_counts.to(dtype=torch.long, device=quads_cuda.device)
            scale_per_region = torch.repeat_interleave(scale_factors, counts_long, dim=0)
            quads_scaled = quads_cuda * scale_per_region.view(-1, 1, 1)

            seq_ids = rec_ids
            seq_probs = rec_probs

            _before_eos = (seq_ids == 1).cumsum(dim=1) == 0
            _real = (seq_ids != 0) & _before_eos
            _counts = _real.sum(dim=1).clamp(min=1).float()
            _log_sum = (torch.log(seq_probs.clamp(min=1e-8)) * _real.float()).sum(dim=1)
            text_confidence = torch.exp(_log_sum / _counts)

            output = {
                "sequences": seq_ids.cpu(),
                "sequence_probs": seq_probs.cpu(),
                "text_confidence": text_confidence.cpu(),
                "region_counts": region_counts.cpu(),
                "quads": quads_scaled.cpu(),
                "raw_detector_confidence": e2e_det_conf,
                "confidence": confidence.cpu(),
                "relations": words,
                "line_relations": lines,
                "line_rel_var": line_var,
                "fg_colors": None,
                "fonts": None,
                "tt_log_var_uncertainty": None,
                "e2e_recog_features": rec_features,
            }

        return output

    # ------------------------------------------------------------------
    # Phase 7 — Post-process
    # ------------------------------------------------------------------

    def _decode_with_fallback(self, output):
        """Decode recognized sequences using pre-computed argmax indices."""
        return self.recog_encoder.convert_targets_to_labels(
            output, image_size=None, is_gt=False,
        )

    def _postprocess_batch(self, output, original_shapes, merge_level, include_invalid, timings=None):
        """Decode sequences, build relation graphs, format per-image results."""
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            return self._postprocess_batch_inner(
                output, original_shapes, merge_level, include_invalid, timings,
            )
        finally:
            if gc_was_enabled:
                gc.enable()

    def _postprocess_batch_inner(self, output, original_shapes, merge_level, include_invalid, timings=None):
        _t = time.perf_counter

        t0 = _t()
        batch = self._decode_with_fallback(output)
        recog_decode_ms = (_t() - t0) * 1000

        t0 = _t()
        relation_batch = self.relation_encoder.convert_targets_to_labels(output, image_size=None, is_gt=False)
        rel_decode_ms = (_t() - t0) * 1000

        t0 = _t()
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
                    for idx in line:
                        block.append(example[idx])
                if block:
                    example.blocks.append(TextBlock(block))

        for example in batch:
            for text_region in example:
                v = text_region.region.vertices
                text_region.region = v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
        graph_build_ms = (_t() - t0) * 1000

        t0 = _t()
        all_predictions = []
        for img_idx, example in enumerate(batch):
            orig_h, orig_w = original_shapes[img_idx]

            if example.relation_graph is None:
                all_predictions.append([])
                continue

            boxes, texts, scores = parse_relational_results(example, level=merge_level)
            boxes, texts, scores = reorder_boxes(boxes, texts, scores, mode="top_left", dbscan_eps=10)

            if len(boxes) == 0:
                all_predictions.append([])
                continue

            boxes_array = np.array(boxes).reshape(-1, 4, 2)
            boxes_array[:, :, 0] = boxes_array[:, :, 0] / orig_w
            boxes_array[:, :, 1] = boxes_array[:, :, 1] / orig_h

            preds = []
            for box, text, conf in zip(boxes_array, texts, scores):
                preds.append({
                    "text": text,
                    "confidence": float(conf),
                    "left": float(box[:, 0].min()),
                    "upper": float(box[:, 1].max()),
                    "right": float(box[:, 0].max()),
                    "lower": float(box[:, 1].min()),
                })
            all_predictions.append(preds)
        format_ms = (_t() - t0) * 1000

        if timings is not None:
            timings["post_recog_decode"] = recog_decode_ms
            timings["post_rel_decode"] = rel_decode_ms
            timings["post_graph_build"] = graph_build_ms
            timings["post_format"] = format_ms

        return all_predictions

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def _process_batch(self, images, merge_level, include_invalid):
        """Full batched pipeline.

        When ``verbose_post`` is True, CUDA-syncs each phase and emits
        per-batch timing via ``logger.info``.  Otherwise runs without
        any synchronisation overhead.
        """
        num_images = len(images)
        profile = self._verbose_post
        T = {} if profile else None

        if profile:
            torch.cuda.synchronize()
        t_wall = time.perf_counter()

        # Phase 1: preprocess
        if profile:
            t0 = time.perf_counter()
        resized_batch, original_shapes, padded_lengths = self._preprocess_batch(
            images, timings=T,
        )
        if profile:
            torch.cuda.synchronize()
            T["preproc_total"] = (time.perf_counter() - t0) * 1000

        # Phase 2: detector
        if profile:
            t0 = time.perf_counter()
        det_conf, det_rboxes, det_features = self._run_detector_batched(resized_batch)
        del resized_batch
        if profile:
            torch.cuda.synchronize()
            T["detector"] = (time.perf_counter() - t0) * 1000

        # Phase 2.5: centerness + peak prefilter
        if self._use_prefilter:
            if profile:
                t0 = time.perf_counter()
            det_conf = self._prefilter_detections(det_conf, det_rboxes)
            if profile:
                torch.cuda.synchronize()
                T["prefilter"] = (time.perf_counter() - t0) * 1000

        # Phase 3: NMS
        if profile:
            t0 = time.perf_counter()
        nms_result = self._run_nms(det_conf, det_rboxes)
        del det_conf, det_rboxes
        if profile:
            torch.cuda.synchronize()
            T["nms"] = (time.perf_counter() - t0) * 1000

        if nms_result is None:
            if profile:
                T["total"] = (time.perf_counter() - t_wall) * 1000
                logger.info(
                    "batch=%d  regions=0  det=%.1f  nms=%.1f  total=%.1f ms (no detections)",
                    num_images, T["detector"], T["nms"], T["total"],
                )
            return [[] for _ in range(num_images)]

        quads, confidence, region_counts, e2e_det_conf = nms_result
        total_regions = quads.shape[0]

        # ── Detector-only mode: return scaled boxes immediately ──
        if self._detector_only:
            with torch.inference_mode():
                scale_factors = torch.as_tensor(
                    padded_lengths, dtype=torch.float32, device=quads.device,
                ) / float(self.infer_length)
                counts_long = region_counts.to(dtype=torch.long, device=quads.device)
                scale_per_region = torch.repeat_interleave(scale_factors, counts_long, dim=0)
                quads_scaled = quads * scale_per_region.view(-1, 1, 1)

            region_counts_cpu = region_counts.cpu()
            quads_np = quads_scaled.cpu().numpy()
            confs_np = confidence.cpu().numpy()

            all_predictions = []
            offset = 0
            for img_idx in range(num_images):
                n = region_counts_cpu[img_idx].item()
                orig_h, orig_w = original_shapes[img_idx]
                predictions = []
                for i in range(n):
                    q = quads_np[offset + i]
                    predictions.append({
                        "quad": q.tolist(),
                        "confidence": float(confs_np[offset + i]),
                        "left": float(q[:, 0].min() / orig_w),
                        "right": float(q[:, 0].max() / orig_w),
                        "upper": float(q[:, 1].max() / orig_h),
                        "lower": float(q[:, 1].min() / orig_h),
                    })
                offset += n
                all_predictions.append(predictions)

            if profile:
                T["total"] = (time.perf_counter() - t_wall) * 1000
                logger.info(
                    "batch=%d  regions=%d  det=%.1f  nms=%.1f  total=%.1f ms (detector_only)",
                    num_images, total_regions, T["detector"], T["nms"], T["total"],
                )
            return all_predictions

        # Phase 4: rectify + grid sample
        if profile:
            t0 = time.perf_counter()
        quads_cuda, rec_quads, rel_quads, region_counts_cpu = self._run_rectify_and_sample(
            quads, region_counts, det_features,
        )
        del det_features
        if profile:
            torch.cuda.synchronize()
            T["rectify"] = (time.perf_counter() - t0) * 1000

        # Phase 5: recognizer
        if profile:
            t0 = time.perf_counter()
        rec_ids, rec_probs, rec_features = self._run_recognizer_chunked(rec_quads)
        del rec_quads
        if profile:
            torch.cuda.synchronize()
            T["recognizer"] = (time.perf_counter() - t0) * 1000

        # Phase 6: relational model (skipped in no-relational mode)
        if self._skip_relational:
            with amp.autocast("cuda", enabled=True), torch.inference_mode():
                scale_factors = torch.as_tensor(
                    padded_lengths, dtype=torch.float32, device=quads_cuda.device,
                ) / float(self.infer_length)
                counts_long = region_counts.to(dtype=torch.long, device=quads_cuda.device)
                scale_per_region = torch.repeat_interleave(scale_factors, counts_long, dim=0)
                quads_scaled = quads_cuda * scale_per_region.view(-1, 1, 1)

                seq_ids = rec_ids
                seq_probs = rec_probs
                _before_eos = (seq_ids == 1).cumsum(dim=1) == 0
                _real = (seq_ids != 0) & _before_eos
                _counts = _real.sum(dim=1).clamp(min=1).float()
                _log_sum = (torch.log(seq_probs.clamp(min=1e-8)) * _real.float()).sum(dim=1)
                text_confidence = torch.exp(_log_sum / _counts)

            batch = self._decode_with_fallback({
                "sequences": seq_ids.cpu(),
                "sequence_probs": seq_probs.cpu(),
                "text_confidence": text_confidence.cpu(),
                "region_counts": region_counts.cpu(),
                "quads": quads_scaled.cpu(),
                "confidence": confidence.cpu(),
            })
            all_predictions = []
            for img_idx, example in enumerate(batch):
                orig_h, orig_w = original_shapes[img_idx]
                predictions = []
                for text_region in example:
                    v = text_region.region.vertices
                    v = v.cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v)
                    v_norm = v.copy()
                    v_norm[:, 0] /= orig_w
                    v_norm[:, 1] /= orig_h
                    predictions.append({
                        "text": text_region.text,
                        "confidence": text_region.confidence,
                        "left": float(v_norm[:, 0].min()),
                        "right": float(v_norm[:, 0].max()),
                        "upper": float(v_norm[:, 1].max()),
                        "lower": float(v_norm[:, 1].min()),
                    })
                all_predictions.append(predictions)

            if profile:
                T["total"] = (time.perf_counter() - t_wall) * 1000
                counts_str = "+".join(str(c) for c in region_counts_cpu.tolist())
                logger.info(
                    "batch=%d  regions=%d(%s)  det=%.1f  nms=%.1f  "
                    "rectify=%.1f  recog=%.1f  total=%.1f ms (skip_relational)",
                    num_images, total_regions, counts_str,
                    T["detector"], T["nms"],
                    T["rectify"], T["recognizer"], T["total"],
                )
            return all_predictions

        # Phase 6: relational model
        if profile:
            t0 = time.perf_counter()
        output = self._run_relational_and_build_output(
            rel_quads,
            quads_cuda,
            region_counts,
            region_counts_cpu,
            rec_features,
            rec_ids,
            rec_probs,
            e2e_det_conf,
            confidence,
            padded_lengths,
        )
        del rel_quads
        if profile:
            torch.cuda.synchronize()
            T["relational"] = (time.perf_counter() - t0) * 1000

        # Phase 7: postprocess
        if profile:
            t0 = time.perf_counter()
        results = self._postprocess_batch(
            output, original_shapes, merge_level, include_invalid, timings=T,
        )
        if profile:
            T["post_total"] = (time.perf_counter() - t0) * 1000
            torch.cuda.synchronize()
            T["total"] = (time.perf_counter() - t_wall) * 1000
            counts_str = "+".join(str(c) for c in region_counts_cpu.tolist())
            logger.info(
                "batch=%d  regions=%d(%s)  "
                "img_load=%.1f  gpu_pre=%.1f  det=%.1f  prefilt=%.1f  nms=%.1f  "
                "rectify=%.1f  recog=%.1f  rel=%.1f  "
                "post[decode=%.1f rel=%.1f graph=%.1f fmt=%.1f]=%.1f  "
                "total=%.1f ms",
                num_images, total_regions, counts_str,
                T.get("img_load", 0), T.get("gpu_preproc", 0),
                T.get("detector", 0), T.get("prefilter", 0), T.get("nms", 0),
                T.get("rectify", 0), T.get("recognizer", 0), T.get("relational", 0),
                T.get("post_recog_decode", 0), T.get("post_rel_decode", 0),
                T.get("post_graph_build", 0), T.get("post_format", 0),
                T.get("post_total", 0),
                T["total"],
            )
        return results
