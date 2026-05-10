# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, List, Tuple, TYPE_CHECKING

# This prevents a cyclic import while allowing typing
if TYPE_CHECKING:
    from data.text_region import TextRegion, Example


import torch

from nemotron_ocr_cpp import get_rel_continuation_cos as _get_rel_continuation_cos

NEW_LINE_THRESHOLD = math.cos(80 / 180 * math.pi)


def _clone_paragraph(paragraph: List[List[int]]):
    return [list(s) for s in paragraph]


class RelationGraph:
    def __init__(self, paragraphs: Optional[List[List[List[int]]]] = None):
        self.paragraphs = paragraphs if paragraphs is not None else []

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return self.paragraphs[idx]

    def __iter__(self):
        return iter(self.paragraphs)

    def __str__(self):
        return str(self.paragraphs)

    def __repr__(self):
        return str(self)

    @property
    def word_count(self):
        ct = 0
        for paragraph in self:
            ct += sum(len(s) for s in paragraph)
        return ct

    @property
    def is_proper_unique(self):
        items = set()
        for paragraph in self:
            for sentence in paragraph:
                for word in sentence:
                    if word in items:
                        return False
                    items.add(word)
        return True

    def is_valid_for_example(self, example: "Example"):
        if self.word_count != len(example):
            return False
        return self.is_proper_unique

    def non_trivial(self):
        nt = []
        for paragraph in self:
            num_words = sum(len(s) for s in paragraph)
            if num_words > 1:
                nt.append(_clone_paragraph(paragraph))

        return RelationGraph(nt)

    def non_trivial_paragraph(self):
        nt = []
        for paragraph in self:
            if len(paragraph) > 1:
                nt.append(_clone_paragraph(paragraph))
        return RelationGraph(nt)

    def multi_line(self, example: "Example"):
        ml = []
        for paragraph in self:
            if is_relation_multiline(example, paragraph):
                ml.append(_clone_paragraph(paragraph))
        return RelationGraph(ml)

    def flatten(self):
        ret = []
        for paragraph in self:
            flat = sum(paragraph, [])
            if flat:
                ret.append([flat])
        return RelationGraph(ret)

    def isolate_sentences(self):
        ret = []
        for paragraph in self:
            for sentence in paragraph:
                ret.append([list(sentence)])
        return RelationGraph(ret)

    def split_lines(self, example: "Example"):
        fg = self.flatten()

        ret = []
        for paragraph in fg:
            sentence = paragraph[0]
            out_para = []
            out_sentence = [sentence[0]]
            for i in range(1, len(sentence)):
                region_a = example[out_sentence[-1]]
                region_b = example[sentence[i]]
                if is_new_line(region_a, region_b):
                    out_para.append(out_sentence)
                    out_sentence = []
                out_sentence.append(sentence[i])
            if out_sentence:
                out_para.append(out_sentence)
            ret.append(out_para)
        return RelationGraph(ret)

    def graph_to_sparse_tensor(self):
        num_words = self.word_count
        ret = torch.full((num_words,), -1, dtype=torch.int64)
        flat = self.flatten()
        for para in flat:
            sent = para[0]
            for i in range(1, len(sent)):
                p_idx = sent[i - 1]
                c_idx = sent[i]
                ret[p_idx] = c_idx
        return ret

    def get_connection_pairs(self, example: "Example" = None):
        pairs = []
        for paragraph in self.flatten():
            sentence = paragraph[0]
            for i in range(1, len(sentence)):
                pairs.append((sentence[i - 1], sentence[i]))

        if example is not None:
            pairs = self.filter_valid_pairs(example, pairs)

        return pairs

    def filter_valid_pairs(self, example: "Example", pairs: List[Tuple[int, int]]):
        out_pairs = []
        for a, b in pairs:
            a_valid = example[a].valid
            b_valid = example[b].valid

            if a_valid and b_valid:
                out_pairs.append((a, b))
        return out_pairs


def is_relation_multiline(example: "Example", paragraph: List[List[int]]):
    """
    Look to see if the absolute angle between two relations is greater than 80 degrees
    """
    flat_text = sum(paragraph, [])

    for i in range(1, len(flat_text)):
        region_a = example[flat_text[i - 1]]
        region_b = example[flat_text[i]]

        if is_new_line(region_a, region_b):
            return True

    return False


def get_rel_continuation_cos(region_a: "TextRegion", region_b: "TextRegion"):
    try:
        rect_a = region_a.region.min_rrect
        rect_b = region_b.region.min_rrect
    except RuntimeError:
        return 1.0

    return _get_rel_continuation_cos(rect_a, rect_b)


def is_new_line(region_a: "TextRegion", region_b: "TextRegion"):
    cos_t = get_rel_continuation_cos(region_a, region_b)

    return cos_t < NEW_LINE_THRESHOLD
