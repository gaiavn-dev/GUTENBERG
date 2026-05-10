// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

torch::Tensor rrect_to_quads(torch::Tensor rrects, float cellSize);
torch::Tensor rrect_to_quads_backward(torch::Tensor rrects, torch::Tensor gradOutput);

torch::Tensor calc_poly_min_rrect(torch::Tensor vertices);

float get_rel_continuation_cos(torch::Tensor rrectA, torch::Tensor rrectB);

torch::Tensor get_poly_bounds_quad(torch::Tensor poly);
