// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include <torch/torch.h>

torch::Tensor quad_rectify_cpu_calc_quad_width(torch::Tensor quads,
                                               int64_t outputHeight,
                                               int64_t roundFactor,
                                               float maxWidth);

torch::Tensor quad_rectify_cpu_forward(torch::Tensor quads,
                                       int64_t imageHeight,
                                       int64_t imageWidth,
                                       int64_t outputHeight,
                                       int64_t outputWidth,
                                       bool isotropic);

torch::Tensor quad_rectify_cpu_backward(torch::Tensor quads,
                                        torch::Tensor gradOutput,
                                        int64_t imageHeight,
                                        int64_t imageWidth,
                                        bool isotropic);
