#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relational target encoder (inference-only)."""

import logging
import math

import torch

from nemotron_ocr.inference.post_processing.data.quadrangle import Quadrangle
import nemotron_ocr.inference.post_processing.data.text_region as tr
from nemotron_ocr.inference.encoders.base import TargetEncoderBase

from nemotron_ocr_cpp import dense_relations_to_graph as cpp_dense_relations_to_graph

logging.getLogger("shapely.geos").setLevel(logging.FATAL)


class RelationalTargetEncoder(TargetEncoderBase):
    def __init__(self, input_size, amp_opt=0, is_train=True):
        super().__init__(input_size, amp_opt, False)

        self.is_train = is_train

    def convert_targets_to_labels(
        self, target_dict, image_size, limit_idxs=None, is_gt=True, **kwargs
    ):
        all_word_relations = target_dict["relations"]
        all_line_relations = target_dict["line_relations"]
        all_line_unc = target_dict.get("line_rel_var", None)
        region_counts = target_dict["region_counts"].cpu()
        all_quads = target_dict["quads"]

        if all_word_relations[0].dim() == 1:
            all_word_relations = [sparse_to_dense(gt_rel) for gt_rel in all_word_relations]

        all_word_relations_cpu = [r.cpu() if r is not None else r for r in all_word_relations]

        region_counts = region_counts.cpu()
        all_quads_cpu = all_quads.cpu()
        cs_region_counts = torch.cumsum(region_counts, 0)

        examples = []
        for i, word_relations_cpu in enumerate(all_word_relations_cpu):
            start_offset = cs_region_counts[i - 1] if i > 0 else 0
            end_offset = cs_region_counts[i]
            quads = all_quads_cpu[start_offset:end_offset]
            line_relations = all_line_relations[i]
            line_unc = all_line_unc[i] if all_line_unc is not None else None

            regions = [tr.TextRegion(Quadrangle(q), "") for q in quads]
            graph = None
            if end_offset > start_offset:
                graph = self.dense_relations_to_graph(
                    word_relations_cpu, line_relations, line_unc, is_gt
                )
            else:
                graph = tr.RelationGraph()

            ex = tr.Example(regions, relation_graph=graph)
            examples.append(ex)

        if limit_idxs is not None:
            examples = [examples[idx] for idx in limit_idxs]

        return tr.Batch(examples)

    def dense_relations_to_graph(
        self,
        word_relations: torch.Tensor,
        line_logits: torch.Tensor,
        line_log_uncertainty: torch.Tensor = None,
        is_gt=False,
    ):
        lines = [p[0] for p in cpp_dense_relations_to_graph(word_relations)]

        dev = line_logits.device
        line_logits = line_logits.float()

        if is_gt:
            null_conn = (line_logits.sum(dim=1, keepdim=True) == 0).to(line_logits.dtype)
            line_logits = torch.cat((null_conn, line_logits), dim=1)

        if line_log_uncertainty is not None:
            line_log_uncertainty = line_log_uncertainty.float()
            inv_uncertainty = torch.exp(-line_log_uncertainty)
        else:
            inv_uncertainty = torch.ones_like(line_logits)

        n_words = word_relations.shape[0]

        line_lengths = torch.tensor([len(line) for line in lines], dtype=torch.int64, device=dev)
        word_indices = torch.tensor(
            [word for line in lines for word in line], dtype=torch.int64, device=dev
        )
        line_ids = torch.repeat_interleave(
            torch.arange(len(lines), dtype=torch.int64, device=dev), line_lengths
        )
        w_idx_to_line_map = torch.empty(n_words, dtype=torch.int64, device=dev)
        w_idx_to_line_map[word_indices] = line_ids
        line_idx = len(lines)

        valid_mask = line_logits != -math.inf

        null_w_idx_to_line_map = torch.cat(
            (torch.zeros(1, dtype=w_idx_to_line_map.dtype, device=dev), w_idx_to_line_map + 1), dim=0
        )

        sa_w2l_idxs = null_w_idx_to_line_map.reshape(1, -1).expand(n_words, -1)
        sa_l2l_idxs = w_idx_to_line_map.reshape(-1, 1).expand(-1, line_idx + 1)

        line_logits = torch.where(valid_mask, line_logits, torch.zeros_like(line_logits))
        inv_uncertainty = torch.where(
            valid_mask, inv_uncertainty, torch.zeros_like(inv_uncertainty)
        )

        word_to_line_unc = torch.zeros(
            n_words, line_idx + 1, dtype=line_logits.dtype, device=dev
        )
        word_to_line_unc.scatter_add_(dim=1, index=sa_w2l_idxs, src=inv_uncertainty)

        line_to_line_unc = torch.zeros(line_idx, line_idx + 1, dtype=line_logits.dtype, device=dev)
        line_to_line_unc.scatter_add_(dim=0, index=sa_l2l_idxs, src=word_to_line_unc)

        unc_sums = line_to_line_unc[w_idx_to_line_map][:, null_w_idx_to_line_map].clamp_min(1e-6)

        unc_weights = inv_uncertainty / unc_sums

        w_logits = torch.where(valid_mask, unc_weights * line_logits, torch.zeros_like(line_logits))

        word_to_line_logits = torch.zeros_like(word_to_line_unc)
        word_to_line_logits.scatter_add_(dim=1, index=sa_w2l_idxs, src=w_logits)

        line_to_line_logits = torch.zeros_like(line_to_line_unc)
        line_to_line_logits.scatter_add_(dim=0, index=sa_l2l_idxs, src=word_to_line_logits)

        self_mask = torch.full(
            (line_to_line_logits.shape[0],), -math.inf, dtype=line_to_line_logits.dtype, device=dev
        ).diag()
        self_mask = torch.cat(
            (torch.zeros(self_mask.shape[0], 1, dtype=self_mask.dtype, device=dev), self_mask), dim=1
        )

        line_to_line_logits += self_mask

        line_to_line_probs = torch.softmax(line_to_line_logits, dim=1, dtype=torch.float32)
        line_maxes = torch.max(line_to_line_probs, dim=1, keepdim=True).values
        line_to_line_probs = torch.where(
            line_to_line_probs == line_maxes,
            torch.ones_like(line_to_line_probs),
            torch.zeros_like(line_to_line_probs),
        )
        line_to_line_probs = line_to_line_probs[:, 1:]

        rel_lines = set(tuple(p[0]) for p in cpp_dense_relations_to_graph(line_to_line_probs.cpu()))

        paragraphs = []
        for rel_line in rel_lines:
            para = [lines[l_idx] for l_idx in rel_line]
            paragraphs.append(para)

        return tr.RelationGraph(paragraphs)


def sparse_to_dense(sparse: torch.Tensor, handle_invalid=True, encode_null=False):
    cols = sparse.shape[0]
    if encode_null:
        sparse = sparse + 1
        cols += 1

    rel_valid_mask = None
    ones = torch.ones(sparse.shape[0], dtype=torch.float32, device=sparse.device)
    if handle_invalid:
        rel_valid_mask = torch.where(sparse >= 0, ones, torch.zeros_like(ones))

    if not encode_null:
        sparse = sparse.clamp_min(0)

    dense = torch.zeros(sparse.shape[0], cols, dtype=torch.float32, device=sparse.device)
    dense.scatter_(dim=1, index=sparse.unsqueeze(1), src=ones.unsqueeze(1))

    if rel_valid_mask is not None:
        dense *= rel_valid_mask[:, None]
    return dense
