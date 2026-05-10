// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

torch::Tensor ragged_quad_all_2_all_distance_v2(torch::Tensor embedQuads, torch::Tensor quadsPerExample,
                                                float xFactor, float yFactor,
                                                bool allowSelfDistance);
