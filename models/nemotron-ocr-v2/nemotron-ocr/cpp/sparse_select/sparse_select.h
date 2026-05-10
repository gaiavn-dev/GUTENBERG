// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, std::vector<torch::Tensor>> sparse_select(torch::Tensor sparseCounts,
                                                                    const std::vector<torch::Tensor> sparseTensors,
                                                                    torch::Tensor selectIndices);
