// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "quad_rectify_gpu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include "quad_rectify_shared.h"
#include "../half_ops.cuh"
#include "../geometry.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x)

template<typename scalar_t, typename quads_accessor_t, typename output_accessor_t>
__global__ void quad_rectify_device_calc_quad_width(quads_accessor_t quads,
                                                    output_accessor_t output,
                                                    const scalar_t outputHeight,
                                                    const scalar_t roundFactor,
                                                    const scalar_t maxWidth)
{
    const unsigned int quadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numQuads = quads.size(0);

    if (quadIdx >= numQuads) {
        return;
    }

    auto currQuad = quads[quadIdx];

    auto quadWidth = calc_quad_width<scalar_t>(currQuad, outputHeight, roundFactor, maxWidth);

    output[quadIdx] = Convert<scalar_t, int64_t>::LeftToRight(quadWidth);
}

template<typename scalar_t, typename quads_accessor_t, typename output_accessor_t>
__global__ void quad_rectify_device_forward(quads_accessor_t quads,
                                            output_accessor_t outputs,
                                            const scalar_t imageHeight,
                                            const scalar_t imageWidth,
                                            bool isotropic)
{
    typedef Point_<scalar_t> Point_t;

    const unsigned int quadIdx = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int numQuads = quads.size(0);

    if (quadIdx >= numQuads) {
        return;
    }

    const unsigned int outputHeight = outputs.size(1);
    const unsigned int outputWidth = outputs.size(2);

    const unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int x = offset % outputWidth;
    const unsigned int y = offset / outputWidth;

    if (y >= outputHeight) {
        return;
    }

    auto quad = quads[quadIdx];
    auto output = outputs[quadIdx][y][x];

    auto scOutputHeight = Convert<scalar_t, unsigned int>::RightToLeft(outputHeight);
    auto scOutputWidth = Convert<scalar_t, unsigned int>::RightToLeft(outputWidth);
    auto scOne = Convert<scalar_t, float>::RightToLeft(1);

    scalar_t quadWidth = isotropic ? calc_quad_width<scalar_t>(quad, scOutputHeight, scOne, scOutputWidth) : scOutputWidth;

    Point_t outputPoint = calc_rect_value<scalar_t>(quad,
                                                    quadWidth,
                                                    scOutputHeight,
                                                    x,
                                                    y,
                                                    imageWidth,
                                                    imageHeight);

    output[0] = outputPoint.X;
    output[1] = outputPoint.Y;
}

template<typename scalar_t, typename quads_accessor_t, typename output_accessor_t>
__global__ void quad_rectify_device_backward(quads_accessor_t quads,
                                             output_accessor_t gradOutput,
                                             quads_accessor_t gradInput,
                                             const scalar_t imageHeight,
                                             const scalar_t imageWidth,
                                             bool isotropic)
{
    const unsigned int numQuads = quads.size(0);
    int64_t quadIdx = blockIdx.y * blockDim.y + threadIdx.y;

    int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;

    const int64_t outputHeight = gradOutput.size(1);
    const int64_t outputWidth = gradOutput.size(2);

    int64_t x = offset % outputWidth;
    int64_t y = offset / outputWidth;

    auto scOutputHeight = Convert<scalar_t, unsigned int>::RightToLeft(outputHeight);
    auto scOutputWidth = Convert<scalar_t, unsigned int>::RightToLeft(outputWidth);
    auto scOne = Convert<scalar_t, float>::RightToLeft(1);
    const scalar_t scHalf = Convert<scalar_t, float>::RightToLeft(0.5);

    auto currQuad = quads[quadIdx];
    scalar_t quadWidth = isotropic ? calc_quad_width<scalar_t>(currQuad, scOutputHeight, scOne, scOutputWidth) : scOutputWidth;

    __shared__ scalar_t sharedFloats[32][8];

    scalar_t scale[2] = { Convert<scalar_t, float>::RightToLeft(2.0f) / imageWidth,
                          Convert<scalar_t, float>::RightToLeft(2.0f) / imageHeight };

    bool valid = false;
    if (quadIdx < numQuads && y < outputHeight) {
        auto fRow = (scalar_t(y) + scHalf) / outputHeight;
        auto fCol = (scalar_t(x) + scHalf) / quadWidth;
        // auto fRow = scalar_t(y) / (outputHeight - scOne);
        // auto fCol = scalar_t(x) / (quadWidth - scOne);
        auto fRowCol = fRow * fCol;

        if (fCol <= 1) {
            #pragma unroll 2
            for (int64_t i = 0; i < 2; ++i) {
                auto currGradOutput = gradOutput[quadIdx][y][x][i] * scale[i];

                sharedFloats[threadIdx.x][0 + i] = currGradOutput * (fRowCol - fCol - fRow + 1);
                sharedFloats[threadIdx.x][2 + i] = currGradOutput * (fCol - fRowCol);
                sharedFloats[threadIdx.x][4 + i] = currGradOutput * fRowCol;
                sharedFloats[threadIdx.x][6 + i] = currGradOutput * (fRow - fRowCol);
            }
            valid = true;
        }
    }

    if (! valid) {
        #pragma unroll 8
        for (int64_t i = 0; i < 8; ++i) {
            sharedFloats[threadIdx.x][i] = 0;
        }
    }

    __syncthreads();

    // Now accumulate over the shared memory
    for (unsigned int i = 16; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            #pragma unroll 8
            for (unsigned int k = 0; k < 8; ++k) {
                sharedFloats[threadIdx.x][k] += sharedFloats[threadIdx.x + i][k];
            }
        }
        __syncthreads();
    }

    auto pGradInput = gradInput[quadIdx].data();

    // Finally, write the values
    if (threadIdx.x == 0) {
        #pragma unroll 8
        for (int64_t i = 0; i < 8; ++i) {
            atomicAdd(pGradInput + i, sharedFloats[0][i]);
        }
    }
}

torch::Tensor quad_rectify_gpu_calc_quad_width(torch::Tensor quads,
                                               int64_t outputHeight,
                                               int64_t roundFactor,
                                               float maxWidth)
{
    CHECK_INPUT(quads);

    const int64_t numQuads = quads.size(0);

    dim3 dimBlock(32);
    dim3 dimGrid(div_up(numQuads, dimBlock.x));

    auto output = torch::empty({ numQuads },
                               quads.options().dtype(torch::kInt64));

    if (numQuads > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            quads.scalar_type(),
            "quad_rectify_calc_quad_width",
            ([&] {
                typedef typename remap_half<scalar_t>::type T;
                quad_rectify_device_calc_quad_width<T> KERNEL_ARG2(dimGrid, dimBlock) (
                    quads.packed_accessor64<T, 3>(),
                    output.packed_accessor64<int64_t, 1>(),
                    Convert<T, int64_t>::RightToLeft(outputHeight),
                    Convert<T, int64_t>::RightToLeft(roundFactor),
                    Convert<T, float>::RightToLeft(maxWidth)
                );
            })
        );
    }

    return output;
}

torch::Tensor quad_rectify_gpu_forward(torch::Tensor quads,
                                       int64_t imageHeight,
                                       int64_t imageWidth,
                                       int64_t outputHeight,
                                       int64_t outputWidth,
                                       bool isotropic)
{
    CHECK_INPUT(quads);

    const int64_t numQuads = quads.size(0);
    const int64_t numCells = outputHeight * outputWidth;

    dim3 dimBlock(32);
    dim3 dimGrid(div_up(numCells, dimBlock.x),
                 numQuads);

    auto output = torch::empty({ numQuads, outputHeight, outputWidth, 2 },
                               quads.options());

    if (numQuads > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            quads.scalar_type(),
            "quad_rectify_device_forward",
            ([&] {
                typedef typename remap_half<scalar_t>::type T;
                quad_rectify_device_forward<T> KERNEL_ARG2(dimGrid, dimBlock) (
                    quads.packed_accessor64<T, 3>(),
                    output.packed_accessor64<T, 4>(),
                    Convert<T, int64_t>::RightToLeft(imageHeight),
                    Convert<T, int64_t>::RightToLeft(imageWidth),
                    isotropic
                );
            })
        );
    }

    return output;
}

torch::Tensor quad_rectify_gpu_backward(torch::Tensor quads,
                                        torch::Tensor gradOutput,
                                        int64_t imageHeight,
                                        int64_t imageWidth,
                                        bool isotropic)
{
    CHECK_INPUT(quads);
    CHECK_INPUT(gradOutput);

    const int64_t numQuads = quads.size(0);
    const int64_t outputHeight = gradOutput.size(1);
    const int64_t outputWidth = gradOutput.size(2);

    const int64_t numCells = outputHeight * outputWidth;

    dim3 dimBlock(32);
    dim3 dimGrid(div_up(numCells, dimBlock.x),
                 numQuads);

    auto gradInput = torch::zeros_like(quads);

    if (numQuads > 0) {
        AT_DISPATCH_FLOATING_TYPES(
            quads.scalar_type(),
            "quad_rectify_device_backward",
            ([&] {
                typedef typename remap_half<scalar_t>::type T;
                quad_rectify_device_backward<T> KERNEL_ARG2(dimGrid, dimBlock) (
                    quads.packed_accessor64<T, 3>(),
                    gradOutput.packed_accessor64<T, 4>(),
                    gradInput.packed_accessor64<T, 3>(),
                    Convert<T, int64_t>::RightToLeft(imageHeight),
                    Convert<T, int64_t>::RightToLeft(imageWidth),
                    isotropic
                );
            })
        );
    }

    return gradInput;
}
