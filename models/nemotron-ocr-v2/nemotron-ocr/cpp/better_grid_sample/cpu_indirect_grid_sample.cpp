// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample.h"
#include "gpu_grid_sample_utils.cuh"

template<typename T>
void indirect_grid_sample_forward_bilinear(torch::TensorAccessor<T, 4> input,
                                           torch::TensorAccessor<T, 4> grid,
                                           torch::TensorAccessor<int64_t, 1> inputIndices,
                                           torch::TensorAccessor<T, 4> output)
{
    const int64_t N = inputIndices.size(0);
    const int64_t C = output.size(1);

    T fInputHeight = input.size(2);
    T fInputWidth = input.size(3);
    int64_t outputHeight = output.size(2);
    int64_t outputWidth = output.size(3);

    #pragma omp parallel for num_threads(8)
    for (int64_t i = 0; i < N; ++i) {
        int64_t inputIdx = inputIndices[i];

        for (int64_t c = 0; c < C; ++c) {
            for (int64_t outY = 0; outY < outputHeight; ++outY) {
                for (int64_t outX = 0; outX < outputWidth; ++outX) {
                    T u = grid[i][outY][outX][0];
                    T v = grid[i][outY][outX][1];

                    if (u < -1 || u > 1 || v < -1 || v > 1) {
                        output[i][c][outY][outX] = 0;
                        continue;
                    }

                    // Denormalize the coordinates
                    u = (u + 1) * ((fInputWidth - 1) / 2);
                    v = (v + 1) * ((fInputHeight - 1) / 2);

                    // Calculate coordinates
                    const T inX = u;
                    const T inXint = std::floor(inX);
                    const T inXfrac = inX - inXint;

                    const T inY = v;
                    const T inYint = std::floor(inY);
                    const T inYfrac = inY - inYint;

                    T ps[] = { 1 - inXfrac, inXfrac };
                    T rs[] = { 1 - inYfrac, inYfrac };
                    T opVal = 0;

                    #pragma unroll
                    for (int64_t row = 0; row < 2; ++row) {
                        #pragma unroll
                        for (int64_t col = 0; col < 2; ++col) {
                            T Tpx = utils::get_pixel_clamped(input, inputIdx, c, inXint + col, inYint + row);
                            opVal += rs[row] * ps[col] * Tpx;
                        }
                    }

                    output[i][c][outY][outX] = opVal;
                }
            }
        }
    }
}

torch::Tensor cpu_indirect_grid_sample_forward(torch::Tensor input, torch::Tensor grid,
                                               torch::Tensor inputIndices, const std::string &method)
{
    auto output = input.new_empty({ inputIndices.size(0), input.size(1), grid.size(1), grid.size(2) });

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "cpu_indirect_grid_sample_forward_impl",
        ([&] {
            typedef scalar_t T;
            if (method == "bilinear") {
                indirect_grid_sample_forward_bilinear(
                    input.accessor<T, 4>(),
                    grid.accessor<T, 4>(),
                    inputIndices.accessor<int64_t, 1>(),
                    output.accessor<T, 4>()
                );
            } else {
                throw std::runtime_error("Unsupported resample method: " + method);
            }
        })
    );

    return output;
}
