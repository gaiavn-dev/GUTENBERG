# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text region class."""

from typing import List, Iterator, Optional

import torch

from nemotron_ocr.inference.post_processing.data.data_container import DataContainer
from nemotron_ocr.inference.post_processing.data.quadrangle import Quadrangle
from nemotron_ocr.inference.post_processing.data.relation_graph import RelationGraph

from nemotron_ocr_cpp import text_region_grouping


HEUR_HORIZONTAL_TOLERANCE = 2.0
HEUR_VERTICAL_TOLERANCE = 0.5


class TextRegion(DataContainer):
    def __init__(
        self,
        region: Quadrangle,
        text: str,
        valid: Optional[bool] = True,
        language=None,
        confidence=1,
    ):
        self.region = region
        self.text = text
        self.valid = valid
        self.quad_prob = 1
        self.text_prob = 1
        self.confidence = confidence
        self.language = language

    def __iter__(self) -> Iterator[Quadrangle]:
        yield self.region

    def to_string(self, indent="") -> str:
        """Creates a string representation of the region for easy printing."""
        vertices = self.region
        if isinstance(vertices, Quadrangle):
            vertices = vertices.vertices

        ret = '{}Region (T="{}", Valid={}) {}'.format(indent, self.text, self.valid, vertices)
        return ret

    def __str__(self):
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def clone(self):
        ret = TextRegion(
            region=self.region.clone(),
            text=self.text,
            valid=self.valid,
        )
        ret.quad_prob = self.quad_prob
        ret.text_prob = self.text_prob
        ret.confidence = self.confidence
        return ret


class TextBlock(DataContainer):
    def __init__(self, regions: List[TextRegion]):
        self.regions = regions

    def __len__(self) -> int:
        return len(self.regions)

    def __iter__(self) -> Iterator[TextRegion]:
        return iter(self.regions)

    def __getitem__(self, idx) -> TextRegion:
        return self.regions[idx]

    def __str__(self):
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def to_string(self, indent="") -> str:
        ret = indent + " ".join(tr.text for tr in self.regions)
        return ret


class Example(DataContainer):
    def __init__(
        self,
        regions: List[TextRegion],
        label=None,
        is_synthetic=False,
        relation_graph: Optional[RelationGraph] = None,
    ):
        self.regions = regions
        self.valid = True
        self.label = label
        self.is_synthetic = is_synthetic
        self.relation_graph = relation_graph
        self._coalesced = None
        self.bounds: Quadrangle = None
        self.blocks: List[TextBlock] = []

    def __len__(self):
        return len(self.regions)

    def __iter__(self) -> Iterator[TextRegion]:
        return iter(self.regions)

    def __getitem__(self, idx) -> TextRegion:
        return self.regions[idx]

    def to_string(self, indent="") -> str:
        ret = "{}Example:\n".format(indent)
        for r in self.regions:
            ret += "{}\n".format(r.to_string(indent=indent + "\t"))
        return ret

    def __str__(self):
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def clone(self):
        ret = Example([r.clone() for r in self.regions], label=self.label)
        ret.valid = self.valid
        return ret

    def compute_text_relations(self):
        if self.relation_graph is not None:
            return self.relation_graph
        Batch([self]).compute_text_relations()
        return self.relation_graph

    def prune_invalid_relations(self):
        if self.relation_graph is None:
            return

        def is_valid(sentence):
            return any(self[w_idx].valid for w_idx in sentence)

        nt = []
        for paragraph in self.relation_graph:
            flat_para = sum(paragraph, [])
            if is_valid(flat_para):
                nt.append(paragraph)
        self.relation_graph = RelationGraph(nt)

    def relations_str(self, graph: RelationGraph = None):
        if graph is None:
            graph = self.relation_graph

        if graph is None:
            return [r.text for r in self]

        ret = []
        for paragraph in graph:
            st = "\n".join(" ".join(self[i].text for i in s) for s in paragraph)
            ret.append(st)

        return ret

    def coalesce_homogeneous(self):
        """
        Coalesce all of the quad buffers into a single tensor, and make the tensors
        homogeneous
        """
        if self._coalesced is not None:
            return self._coalesced

        num_vertices = 0
        tensors = []
        for tr in self:
            tensors.append(tr.region.vertices)
            num_vertices += tensors[-1].shape[0]

        coal = torch.empty(num_vertices, 3, dtype=torch.float32)
        coal[:, 2] = 1
        if tensors:
            torch.cat(tensors, dim=0, out=coal[:, :2])

        self._coalesced = coal
        offset = 0
        for tr in self:
            ex_verts = tr.region.vertices
            coal_slice = coal[offset : offset + ex_verts.shape[0], :2]
            tr.region.vertices = coal_slice
            offset += ex_verts.shape[0]

        return self._coalesced

    def apply_stm(self, stm, perspective=False, **kwargs):
        if self.bounds is not None:
            self.bounds.apply_stm(stm, perspective, **kwargs)

        if self._coalesced is None:
            return super().apply_stm(stm, perspective=perspective, **kwargs)

        tx = torch.matmul(self._coalesced, stm)
        self._coalesced.copy_(tx)

        if perspective:
            self._coalesced /= self._coalesced[:, 2:]

    def translate(self, delta_vector):
        if self.bounds is not None:
            self.bounds.translate(delta_vector)

        if self._coalesced is None:
            return super().translate(delta_vector)

        self._coalesced[:, :2] += delta_vector

    def scale(self, scale_vector, **kwargs):
        if self.bounds is not None:
            self.bounds.scale(scale_vector, **kwargs)

        if self._coalesced is None:
            return super().scale(scale_vector, **kwargs)

        self._coalesced[:, :2] *= scale_vector

    def rotate(self, rot_mat):
        if self.bounds is not None:
            self.bounds.rotate(rot_mat)

        if self._coalesced is None:
            return super().rotate(rot_mat)

        view = self._coalesced[:, :2]
        tx = torch.matmul(view, rot_mat.t())
        view.copy_(tx)

    def validate(self):
        coal = self.coalesce_homogeneous()
        return torch.all(torch.isfinite(coal[:, :2])).item()


class Batch(DataContainer):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, idx) -> Example:
        return self.examples[idx]

    def to_string(self, indent="") -> str:
        ret = "{}Batch:\n".format(indent)
        for ex in self.examples:
            ret += "{}\n".format(ex.to_string(indent=indent + "\t"))
        return ret

    def __str__(self):
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def clone(self):
        return Batch([ex.clone() for ex in self.examples])

    def compute_text_relations(self):
        graphs = self.get_text_relations()

        for i, ex in enumerate(self):
            ex.relation_graph = graphs[i]

    def get_text_relations(self):
        cts = torch.zeros(len(self), dtype=torch.int64)
        total_ct = 0
        for i, ex in enumerate(self):
            cts[i] = len(ex)
            total_ct += len(ex)

        quads = torch.zeros(total_ct, 4, 2, dtype=torch.float32)
        off = 0
        for ex in self:
            for tr in ex:
                quads[off] = tr.region.vertices
                off += 1

        all_relations = text_region_grouping(
            quads,
            cts,
            horizontal_tolerance=HEUR_HORIZONTAL_TOLERANCE,
            vertical_tolerance=HEUR_VERTICAL_TOLERANCE,
        )

        graphs = []

        for ex, phrases in zip(self, all_relations):
            paragraphs = []
            for i in range(len(phrases) - 1, -1, -1):
                r_flat = []
                for line in phrases[i]:
                    r_flat.extend(line)

                any_valid = False
                for tr_idx in r_flat:
                    if ex[tr_idx].valid:
                        any_valid = True
                        break

                if not any_valid:
                    del phrases[i]
                else:
                    curr_sentence = []
                    sentences = []
                    for word_idx in r_flat:
                        tr = ex[word_idx]
                        curr_sentence.append(word_idx)
                        if tr.text.endswith((".", "?", "!")):
                            sentences.append(curr_sentence)
                            curr_sentence = []
                    if curr_sentence:
                        sentences.append(curr_sentence)
                    paragraphs.append(sentences)

            graphs.append(RelationGraph(paragraphs))

        return graphs

    def get_valid_list(self):
        return [sub.validate() for sub in self]
