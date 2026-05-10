// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include "language_model.h"

std::tuple<torch::Tensor, torch::Tensor>
    beam_decode(torch::Tensor probs, int64_t beamSize, int64_t blank,
                float minProb,
                const LanguageModel *langModel,
                float lmWeight,
                bool combineDuplicates);

std::unique_ptr<LanguageModel> create_sbo_lm(const std::string &dataFilePath, token_mapping_t tokenMapping, float_t backoffWeight);
