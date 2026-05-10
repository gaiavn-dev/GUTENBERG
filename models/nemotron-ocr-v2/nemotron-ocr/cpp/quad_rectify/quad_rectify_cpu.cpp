// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "quad_rectify_cpu.h"

#include <iostream>

#include "../geometry.h"
#include "quad_rectify_shared.h"

using namespace std;

template<typename quads_accessor_t, typename output_accessor_t, typename scalar_t>
void quad_rectify_calc_quad_width_impl(const quads_accessor_t &quads,
                                       output_accessor_t output,
                                       const scalar_t outputHeight,
                                       const scalar_t roundFactor,
                                       const scalar_t maxWidth)
{
    const int64_t numQuads = quads.size(0);

    for (int64_t quadIdx = 0; quadIdx < numQuads; ++quadIdx) {
        auto quadWidth = calc_quad_width(quads[quadIdx], outputHeight, roundFactor, maxWidth);

        output[quadIdx] = Convert<scalar_t, int64_t>::LeftToRight(quadWidth);
    }
}

template<typename quads_accessor_t, typename output_accessor_t, typename scalar_t>
void quad_rectify_cpu_forward_impl(const quads_accessor_t &quads,
                                   output_accessor_t output,
                                   const scalar_t imageHeight,
                                   const scalar_t imageWidth,
                                   bool isotropic)
{
    typedef Point_<scalar_t> Point_t;

    const int64_t numQuads = quads.size(0);
    const int64_t outputHeight = output.size(1);
    const int64_t outputWidth = output.size(2);

    for (int64_t quadIdx = 0; quadIdx < numQuads; ++quadIdx) {
        auto currQuad = quads[quadIdx];

        scalar_t quadWidth = isotropic ? calc_quad_width<scalar_t>(currQuad, outputHeight, 1, outputWidth) : scalar_t(outputWidth);

        for (int64_t row = 0; row < outputHeight; ++row) {
            for (int64_t col = 0; col < outputWidth; ++col) {
                Point_t outputPoint = calc_rect_value<scalar_t>(currQuad,
                                                                quadWidth,
                                                                outputHeight,
                                                                col,
                                                                row,
                                                                imageWidth,
                                                                imageHeight);

                auto currOutput = output[quadIdx][row][col];
                currOutput[0] = outputPoint.X;
                currOutput[1] = outputPoint.Y;
            }
        }
    }
}

/*template<typename scalar_t>
void quad_rectify_cpu_backward_impl(torch::Tensor quads,
                                    torch::Tensor gradOutput,
                                    torch::Tensor gradInput)
{
    const int64_t batchSize = gradOutput.size(0);
    const int64_t outputHeight = gradOutput.size(1);
    const int64_t outputWidth = gradOutput.size(2);

    auto gradInputAccess = gradInput.accessor<scalar_t, 3>();
    auto gradOutputAccess = gradOutput.accessor<scalar_t, 4>();

    for (int64_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
        auto batchInputAccess = gradInputAccess[batchIdx];
        auto batchOutputAccess = gradOutputAccess[batchIdx];

        for (int64_t rowIdx = 0; rowIdx < outputHeight; ++rowIdx) {
            for (int64_t colIdx = 0; colIdx < outputWidth; ++colIdx) {

                const scalar_t fRow = scalar_t(rowIdx) / outputHeight;
                const scalar_t fCol = scalar_t(colIdx) / outputWidth;
                const scalar_t fRowCol = fRow * fCol;

                for (int64_t dim = 0; dim < 2; ++dim) {
                    const scalar_t dOut = batchOutputAccess[rowIdx][colIdx][dim];

                    const scalar_t gradIns[] = {
                        dOut * (fRowCol - fCol - fRow + 1),
                        dOut * (fCol - fRowCol),
                        dOut * fRowCol,
                        dOut * (fRow - fRowCol)
                    };

                    for (int64_t quadIdx = 0; quadIdx < 4; ++quadIdx) {
                        batchInputAccess[quadIdx][dim] += 2.0f * gradIns[quadIdx];
                    }
                }
            }
        }
    }
}*/

torch::Tensor quad_rectify_cpu_calc_quad_width(torch::Tensor quads,
                                               int64_t outputHeight,
                                               int64_t roundFactor,
                                               float maxWidth)
{
    auto output = torch::empty({ quads.size(0) },
                               quads.options().dtype(torch::kInt64));

    AT_DISPATCH_FLOATING_TYPES(
        quads.scalar_type(),
        "quad_rectify_cpu_calc_quad_width",
        ([&] {
            quad_rectify_calc_quad_width_impl(
                quads.accessor<scalar_t, 3>(),
                output.accessor<int64_t, 1>(),
                Convert<scalar_t, int64_t>::RightToLeft(outputHeight),
                Convert<scalar_t, int64_t>::RightToLeft(roundFactor),
                Convert<scalar_t, float>::RightToLeft(maxWidth)
            );
        })
    );

    return output;
}

torch::Tensor quad_rectify_cpu_forward(torch::Tensor quads,
                                       int64_t imageHeight,
                                       int64_t imageWidth,
                                       int64_t outputHeight,
                                       int64_t outputWidth,
                                       bool isotropic)
{
    auto output = torch::empty({ quads.size(0), outputHeight, outputWidth, 2 },
                               quads.options());

    AT_DISPATCH_FLOATING_TYPES(
        quads.scalar_type(),
        "quad_rectify_cpu_forward",
        ([&] {
            quad_rectify_cpu_forward_impl(
                quads.accessor<scalar_t, 3>(),
                output.accessor<scalar_t, 4>(),
                Convert<scalar_t, int64_t>::RightToLeft(imageHeight),
                Convert<scalar_t, int64_t>::RightToLeft(imageWidth),
                isotropic
            );
        })
    );

    return output;
}

torch::Tensor quad_rectify_cpu_backward(torch::Tensor quads,
                                        torch::Tensor gradOutput,
                                        int64_t imageHeight,
                                        int64_t imageWidth,
                                        bool isotropic)
{
    auto gradInput = torch::zeros_like(quads);

    throw std::runtime_error("Calling backward, and it's not implemented!");

    /*AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        quads.scalar_type(),
        "quad_rectify_cpu_backward",
        ([&] {
            quad_rectify_cpu_backward_impl<scalar_t>(quads,
                                                     gradOutput,
                                                     gradInput);
        })
    );*/

    return gradInput;
}
