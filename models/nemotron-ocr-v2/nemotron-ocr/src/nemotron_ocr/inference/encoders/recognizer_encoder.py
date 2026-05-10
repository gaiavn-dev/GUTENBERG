#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target encoder for recognition models (inference-only)."""

from collections import defaultdict
import logging
import math
import os
from typing import Tuple, Optional, Callable, Dict

import torch

import nemotron_ocr.inference.post_processing.data.text_region as tr
from nemotron_ocr.inference.post_processing.data.quadrangle import Quadrangle
from nemotron_ocr.inference.encoders.base import TargetEncoderBase

from nemotron_ocr_cpp import (
    beam_decode,
    decode_sequences,
    create_sbo_lm,
    create_token_mapping,
)
from nemotron_ocr.inference.models.utils import f_measure

logger = logging.getLogger(__name__)
logging.getLogger("shapely.geos").setLevel(logging.FATAL)

_NUM_SPECIAL = 3


class RecognitionTargetEncoder(TargetEncoderBase):
    def __init__(
        self,
        charset: str,
        input_size,
        sequence_length: int,
        amp_opt=0,
        combine_duplicates=False,
        is_train=True,
        lm_path=None,
        verbose=False,
    ):
        super().__init__(input_size, amp_opt, verbose)

        self.sequence_length = sequence_length
        self.combine_duplicates = combine_duplicates
        self.is_train = is_train

        logger.info("Combine duplicates: {}".format(combine_duplicates))

        self.charset = charset
        self.lm_path = lm_path
        self.cpp_token_mapping = None

        self._initialized = False

        self.beam_lm = None

        self.send_buffers = None

    def _initialize(self):
        if self._initialized:
            return
        self._initialized = True

        self.idx_to_char = {i + _NUM_SPECIAL: c for i, c in enumerate(self.charset)}

        self.char_to_idx = {c: i + _NUM_SPECIAL for i, c in enumerate(self.charset)}

        if self.lm_path is not None:
            if not os.path.exists(self.lm_path):
                raise ValueError(f"The language model path '{self.lm_path}' doesn't exist!")
            self.beam_lm = create_sbo_lm(self.lm_path, self.idx_to_char)

        self.cpp_token_mapping = create_token_mapping(self.idx_to_char)

    def __getstate__(self):
        ret = dict(self.__dict__)
        if self._initialized:
            del ret["cpp_token_mapping"]
            del ret["idx_to_char"]
            del ret["char_to_idx"]
            del ret["beam_lm"]
            del ret["send_buffers"]
        ret["_initialized"] = False

        return ret

    @property
    def charset_size(self):
        return _NUM_SPECIAL + len(self.charset)

    def is_recognition(self):
        return True

    def get_charset(self):
        return self.charset

    def cb_convert_targets_to_labels(
        self,
        target_dict: Dict[str, torch.Tensor],
        image_size,
        limit_idxs: Optional[torch.Tensor],
        is_gt,
        subsel_fn: Optional[Callable[[int, int, int], Dict[str, torch.Tensor]]],
        geometry_fn: Callable[[Dict, int, int, int], torch.Tensor],
        **kwargs,
    ):
        self._initialize()

        target_dict = self.subselect_targets(target_dict, limit_idxs, subsel_fn)

        sequences = target_dict["sequences"].cpu()
        region_counts = target_dict["region_counts"].cpu()
        confidence = target_dict.get("confidence", None)
        if confidence is not None:
            confidence = confidence.cpu()

        decoded_seq_probs = None
        combine_duplicates = not is_gt and self.combine_duplicates
        language_model = self.beam_lm if not is_gt else None
        if sequences.dim() == 3:
            if sequences.shape[0] > 0:
                decoded_seq_ids, decoded_seq_probs, combine_duplicates = self.convert_preds_to_idxs(
                    sequences, combine_duplicates, language_model
                )
            else:
                decoded_seq_ids = torch.empty(
                    0, sequences.shape[1], dtype=torch.int64, device=sequences.device
                )
        elif sequences.dim() == 2:
            decoded_seq_ids = sequences
            if "sequence_probs" in target_dict:
                decoded_seq_probs = target_dict["sequence_probs"].cpu()
        else:
            raise ValueError("Unsupported sequence tensor!")

        decoded_strings = decode_sequences(
            decoded_seq_ids, self.cpp_token_mapping, decoded_seq_probs
        )

        _text_confs = None
        if "text_confidence" in target_dict:
            _text_confs = target_dict["text_confidence"].tolist()
        elif decoded_seq_probs is not None and decoded_seq_ids.shape[0] > 0:
            _ids = decoded_seq_ids
            _probs = decoded_seq_probs.float().clamp(min=1e-8)
            _before_eos = (_ids == 1).cumsum(dim=1) == 0
            _real = (_ids != 0) & _before_eos
            _counts = _real.sum(dim=1).clamp(min=1).float()
            _log_sum = (torch.log(_probs) * _real.float()).sum(dim=1)
            _text_confs = torch.exp(_log_sum / _counts).tolist()

        examples = []
        offset = 0
        for ex_idx, region_count in enumerate(region_counts):
            region_count = region_count.item()

            regions = []
            for i in range(region_count):
                text, text_conf = decoded_strings[offset]
                if _text_confs is not None:
                    text_conf = _text_confs[offset]
                region_conf = confidence[offset].item() if confidence is not None else 1
                geo = geometry_fn(target_dict, ex_idx, i, offset)
                offset += 1

                overall_conf = f_measure(region_conf, text_conf)

                region = tr.TextRegion(
                    Quadrangle(geo), text, valid=len(text) > 0 and overall_conf > 0.5
                )
                region.quad_prob = region_conf
                region.text_prob = text_conf
                region.confidence = overall_conf
                regions.append(region)

            examples.append(tr.Example(regions))

        return tr.Batch(examples)

    def subselect_targets(
        self,
        target_dict: Dict[str, torch.Tensor],
        limit_idxs: torch.Tensor,
        limit_fn: Optional[Callable[[int, int, int], Dict[str, torch.Tensor]]] = None,
    ):
        if limit_idxs is None:
            return target_dict

        sequences = target_dict["sequences"].cpu()
        region_counts = target_dict["region_counts"].cpu()
        geo_idxs = target_dict["geo_idxs"].cpu()
        confidence = target_dict.get("confidence", None)
        if confidence is not None:
            confidence = confidence.cpu()

        new_seqs = []
        new_counts = []
        new_confidence = []
        new_geo_idxs = []
        other_limits = defaultdict(lambda: [])
        cs_region_counts = torch.cumsum(region_counts, 0)
        for limit_idx in limit_idxs:
            limit_idx = limit_idx.item()
            start_offset = cs_region_counts[limit_idx - 1].item() if limit_idx > 0 else 0
            end_offset = cs_region_counts[limit_idx].item()
            new_seqs.append(sequences[start_offset:end_offset])
            new_geo_idxs.append(geo_idxs[start_offset:end_offset])

            if limit_fn is not None:
                others = limit_fn(limit_idx, start_offset, end_offset)
                for k, v in others.items():
                    other_limits[k].append(v)

            if confidence is not None:
                new_confidence.append(confidence[start_offset:end_offset])
            new_counts.append(region_counts[limit_idx].item())

        sequences = torch.cat(new_seqs)
        geo_idxs = torch.cat(new_geo_idxs)
        if confidence is not None:
            confidence = torch.cat(new_confidence)
        region_counts = torch.tensor(new_counts, dtype=torch.int64)
        for k, v in other_limits.items():
            other_limits[k] = torch.cat(v, dim=0)

        ret = {k: v for k, v in target_dict.items()}
        ret.update(
            sequences=sequences,
            region_counts=region_counts,
            geo_idxs=geo_idxs,
            confidence=confidence,
        )
        ret.update(other_limits)

        return ret

    @staticmethod
    def convert_preds_to_idxs(
        seq: torch.Tensor, combine_duplicates=False, language_model=None
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Converts a prediction distribution to the set of preferred sequences.
        seq: BxTxC, where B=batch, T=timestep, C=char
        Returns: Tuple[indices, probs, combine_duplicates]
        """

        if combine_duplicates or language_model is not None:
            output, scores = beam_decode(
                seq,
                100,
                lang_model=language_model,
                lm_weight=1,
                combine_duplicates=combine_duplicates,
            )
        else:
            scores, output = torch.max(seq, dim=2)

        return output, scores, False

    def decode_sequence(
        self, seq: torch.Tensor, remove_duplicates=False, probs: torch.Tensor = None
    ) -> Tuple[str, float]:
        self._initialize()

        text = ""
        prev = None
        prob = 0
        for i, tok_idx in enumerate(seq):
            tok_idx = tok_idx.item()
            if tok_idx == prev and remove_duplicates:
                continue
            prev = tok_idx

            if tok_idx != 1 and probs is not None and probs.dim() == 1:
                tok_prob = math.log(probs[i].item())

                prob += tok_prob

            if tok_idx == 0:
                continue
            elif tok_idx == 1:
                break
            elif tok_idx == 2:
                text += "^"
            else:
                text += self.idx_to_char[tok_idx]

        prob = math.exp(prob)
        if probs is not None and probs.dim() == 0:
            prob = probs.item()
        return text, prob

    def convert_targets_to_labels(
        self, target_dict, image_size, limit_idxs=None, is_gt=True, **kwargs
    ):
        quads_cpu = target_dict["quads"].cpu()

        def subsel_quads(limit_idx: int, start_offset: int, end_offset: int):
            return {"quads": quads_cpu[start_offset:end_offset]}

        def get_quad(target_dict: Dict[str, torch.Tensor], ex_idx: int, r_idx: int, r_offset: int):
            return quads_cpu[r_offset]

        return self.cb_convert_targets_to_labels(
            target_dict, image_size, limit_idxs, is_gt, subsel_fn=subsel_quads, geometry_fn=get_quad
        )
