// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_api.h"

#include "geometry_api_common.h"

using namespace std;

torch::Tensor rrect_to_quads_gpu(torch::Tensor rrects, float cellSize);

template<typename T>
torch::Tensor rrect_to_quads_impl(torch::Tensor rrects, T cellSize)
{
    // BHW(5)
    auto rrectAccess = rrects.accessor<T, 4>();

    T cellOff = cellSize / 2;

    auto quads = torch::empty({ rrects.size(0), rrects.size(1), rrects.size(2), 4, 2 }, rrects.options());

    auto quadsAccess = quads.accessor<T, 5>();

    for (long b = 0; b < rrects.size(0); ++b) {
        for (long y = 0; y < rrects.size(1); ++y) {
            for (long x = 0; x < rrects.size(2); ++x) {
                auto rrect = rrectAccess[b][y][x];

                auto quad = quadsAccess[b][y][x];

                assign_rrect_to_quad(rrect, quad, cellSize, cellOff,
                                     static_cast<T>(x),
                                     static_cast<T>(y));
            }
        }
    }

    return quads;
}

torch::Tensor rrect_to_quads(torch::Tensor rrects, float cellSize)
{
    if (rrects.is_cuda()) {
        return rrect_to_quads_gpu(rrects, cellSize);
    }

    torch::Tensor quads;
    AT_DISPATCH_FLOATING_TYPES(
        rrects.scalar_type(),
        "rrect_to_quads_impl",
        ([&] {
            quads = rrect_to_quads_impl<scalar_t>(rrects, scalar_t(cellSize));
        })
    );

    return quads;
}


template<typename T>
torch::Tensor rrect_to_quads_backward_impl(torch::Tensor rrects, torch::Tensor gradOutput)
{
    // BHW(5)
    auto gradInput = torch::empty_like(rrects);

    auto rrectAccess = rrects.accessor<T, 4>();
    // BHW42
    auto gradOutputAccess = gradOutput.accessor<T, 5>();
    auto gradInputAccess = gradInput.accessor<T, 4>();

    for (long b = 0; b < rrects.size(0); ++b) {
        for (long y = 0; y < rrects.size(1); ++y) {
            for (long x = 0; x < rrects.size(2); ++x) {
                assign_grad_rrect_to_quad<T>(rrectAccess[b][y][x], gradOutputAccess[b][y][x], gradInputAccess[b][y][x]);
            }
        }
    }

    return gradInput;
}

torch::Tensor rrect_to_quads_backward_gpu(torch::Tensor rrects, torch::Tensor gradOutput);

torch::Tensor rrect_to_quads_backward(torch::Tensor rrects, torch::Tensor gradOutput)
{
    if (rrects.is_cuda()) {
        return rrect_to_quads_backward_gpu(rrects, gradOutput);
    }

    torch::Tensor gradInput;
    AT_DISPATCH_FLOATING_TYPES(
        rrects.scalar_type(),
        "rrect_to_quads_backward_impl",
        ([&] {
            gradInput = rrect_to_quads_backward_impl<scalar_t>(rrects, gradOutput);
        })
    );

    return gradInput;
}
