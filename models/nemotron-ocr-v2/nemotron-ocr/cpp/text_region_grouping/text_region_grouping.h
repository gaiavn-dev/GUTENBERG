// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>


typedef std::vector<int64_t> TextLine;
typedef std::vector<TextLine> Phrase;
typedef std::vector<Phrase> PhraseList;


std::vector<PhraseList> text_region_grouping(torch::Tensor sparseQuads, torch::Tensor sparseCounts,
                                             float horizontalTolerance = 2.0f,
                                             float verticalTolerance = 1.0f,
                                             bool verbose = false);

PhraseList dense_relations_to_graph(torch::Tensor relations);

typedef std::tuple<int64_t, float> relation_t;
typedef std::vector<relation_t> text_line_t;
typedef std::vector<text_line_t> relations_list_t;

relations_list_t dense_relations_to_graph_with_probs(torch::Tensor relationsTensor);
relations_list_t sparse_relations_to_graph(torch::Tensor relationsTensor, torch::Tensor neighborIdxs);
