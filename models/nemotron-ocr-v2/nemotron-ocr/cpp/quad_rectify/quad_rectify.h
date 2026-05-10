// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include "quad_rectify_cpu.h"
#include "quad_rectify_gpu.h"

inline
torch::Tensor quad_rectify_calc_quad_width(torch::Tensor quads,
                                           int64_t outputHeight,
                                           int64_t roundFactor,
                                           float maxWidth)
{
    if (quads.dim() < 2 || quads.dim() > 3) {
        throw std::runtime_error("Invalid quads dimensions.");
    }

    if (quads.size(-1) != 2 || quads.size(-2) != 4) {
        throw std::runtime_error("The final 2 quad dimensions must be 4x2.");
    }

    if (quads.dim() == 2) {
        quads = quads.unsqueeze(0);
    }

    if (quads.is_cuda()) {
        return quad_rectify_gpu_calc_quad_width(quads, outputHeight, roundFactor, maxWidth);
    } else {
        return quad_rectify_cpu_calc_quad_width(quads, outputHeight, roundFactor, maxWidth);
    }
}

inline
torch::Tensor quad_rectify_forward(torch::Tensor quads,
                                   int64_t imageHeight,
                                   int64_t imageWidth,
                                   int64_t outputHeight,
                                   int64_t outputWidth,
                                   bool isotropic)
{
    if (quads.dim() < 2 || quads.dim() > 3) {
        throw std::runtime_error("Invalid quads dimensions.");
    }

    if (quads.size(-1) != 2 || quads.size(-2) != 4) {
        throw std::runtime_error("The final 2 quad dimensions must be 4x2.");
    }

    bool flatten = false;
    if (quads.dim() == 2) {
        quads = quads.unsqueeze(0);
        flatten = true;
    }

    torch::Tensor ret;
    if (quads.is_cuda()) {
        ret = quad_rectify_gpu_forward(quads, imageHeight, imageWidth, outputHeight, outputWidth, isotropic);
    }
    else {
        ret = quad_rectify_cpu_forward(quads, imageHeight, imageWidth, outputHeight, outputWidth, isotropic);
    }

    if (flatten) {
        ret = ret[0];
    }

    return ret;
}

inline
torch::Tensor quad_rectify_backward(torch::Tensor quads, torch::Tensor gradOutput,
                                    int64_t imageHeight, int64_t imageWidth,
                                    bool isotropic)
{
    if (quads.is_cuda() != gradOutput.is_cuda()) {
        throw std::runtime_error("Either both 'quads' and 'gradOutput' must be cuda, or neither.");
    }

    if (quads.dim() != 3 || quads.size(-2) != 4 || quads.size(-1) != 2) {
        throw std::runtime_error("Expected quads to be 3 dimensional. Nx4x2.");
    }

    if (gradOutput.dim() != 4 ||
        gradOutput.size(3) != 2) {
        throw std::runtime_error("Expected 'gradOutput' to be 4d: Nx<outputHeight>x<outputWidth>x2.");
    }

    if (quads.is_cuda()) {
        return quad_rectify_gpu_backward(quads, gradOutput, imageHeight, imageWidth, isotropic);
    }
    else {
        return quad_rectify_cpu_backward(quads, gradOutput, imageHeight, imageWidth, isotropic);
    }
}
