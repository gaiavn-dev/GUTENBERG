// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "grid_sample.h"

#include "../cuda_intellisense.cuh"
#include "../half_ops.cuh"
#include "gpu_grid_sample_utils.cuh"

using namespace std;

template<typename accessor_t, typename index_t>
__device__ __lib_inline__
auto &my_get_pixel_clamped(accessor_t &inputs, index_t x, index_t y)
{
    x = utils::clamp(x, 0, inputs.size(1) - 1);
    y = utils::clamp(y, 0, inputs.size(0) - 1);

    return inputs[y][x];
}

__global__
void single_ex_grid_sample_bilinear_kernel(const float *pInputImage,
                                           uint32_t imgHeight, uint32_t imgWidth, uint32_t numChannels,
                                           const float2 *pGrid,
                                           uint32_t numGridCells,
                                           float *pOutputImage)
{
    const uint32_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t c = blockDim.y * blockIdx.y + threadIdx.y;

    if (c >= numChannels || z >= numGridCells) {
        return;
    }

    const uint32_t g = blockIdx.z;

    const float2 uv = pGrid[g * numGridCells + z];

    float &outPx = pOutputImage[(g * numChannels + c) * numGridCells + z];
    if (abs(uv.x) > 1.0f || abs(uv.y) > 1.0f) {
        outPx = 0.0f;
    } else {
        const uint32_t maxX = imgWidth - 1;
        const uint32_t maxY = imgHeight - 1;

        const float u = (uv.x + 1.0f) * maxX * 0.5f;
        const float v = (uv.y + 1.0f) * maxY * 0.5f;

        // calculate coordinates
        const float inX = u;
        const uint32_t inXint = inX;
        const float inXfrac = inX - inXint;

        const float inY = v;
        const uint32_t inYint = inY;
        const float inYfrac = inY - inYint;

        const float *pChanImage = pInputImage + c * imgHeight * imgWidth;

        // By being in this conditional block, we know that u and v are >= 0, which means
        // that their truncated value is also >= 0. Instead of clamping the value to within the buffer,
        // we set the multiplication factor to be 0 if the interpolated value is outside the buffer
        const float ps[] = { 1.0f - inXfrac, inXfrac * (inXint < maxX) };
        const float rs[] = { 1.0f - inYfrac, inYfrac * (inYint < maxY) };
        float opVal = 0.0f;
        #pragma unroll
        for (uint32_t row = 0; row < 2; ++row) {
            const float *pRowImage = pChanImage + (inYint + row) * imgWidth;

            #pragma unroll
            for (uint32_t col = 0; col < 2; ++col) {
                const float px = pRowImage[inXint + col];
                opVal += rs[row] * ps[col] * px;
            }
        }

        outPx = opVal;
    }
}

template<typename T>
__global__
void indirect_grid_sample_forward_bilinear_kernel(torch::PackedTensorAccessor32<T, 4> inputs,
                                                  torch::PackedTensorAccessor32<T, 4> grid,
                                                  torch::PackedTensorAccessor32<int64_t, 1> inputIndices,
                                                  torch::PackedTensorAccessor32<T, 4> outputs)
{
    static_assert(std::is_same<T, float>::value, "Currently only float32 is supported!");
    //typedef typename fp_promote<T>::type accum_t;
    typedef float accum_t;
    constexpr T NEG_ONE = -1;
    constexpr T ONE = 1;
    constexpr T ZERO = 0;
    constexpr T TWO = 2;
    constexpr T ZERO_PT_5 = 0.5;
    typedef decltype(inputs.stride(0)) index_t;

    const index_t n = blockDim.z * blockIdx.z + threadIdx.z;

    if (n >= inputIndices.size(0)) return;

    const index_t c = blockDim.y * blockIdx.y + threadIdx.y;

    const index_t z = blockDim.x * blockIdx.x + threadIdx.x;

    const accum_t inputHeight = inputs.size(2);
    const accum_t inputWidth = inputs.size(3);
    const index_t outputHeight = outputs.size(2);
    const index_t outputWidth = outputs.size(3);

    const index_t outY = z / outputWidth;
    //const index_t outX = z % outputWidth;
    const index_t outX = z - (outY * outputWidth);

    if (outY >= outputHeight) return;

    index_t inputIdx = inputIndices[n];
    const float2 f2uv = *reinterpret_cast<const float2*>(grid[n][outY][outX].data());
    float u = f2uv.x;
    float v = f2uv.y;

    if (u < NEG_ONE || u > ONE || v < NEG_ONE || v > ONE) {
        outputs[n][c][outY][outX] = ZERO;
        return;
    }

    // Denormalize the coordinates
    u = (u + ONE) * ((inputWidth - ONE) * ZERO_PT_5);
    v = (v + ONE) * ((inputHeight - ONE) * ZERO_PT_5);

    // calculate coordinates
    const accum_t inX = u;
    const index_t inXint = inX;
    const accum_t inXfrac = inX - inXint;

    const accum_t inY = v;
    const index_t inYint = inY;
    const accum_t inYfrac = inY - inYint;

    accum_t ps[] = { ONE - inXfrac, inXfrac };
    accum_t rs[] = { ONE - inYfrac, inYfrac };
    accum_t opVal = ZERO;

    auto localInputs = inputs[inputIdx][c];

    #pragma unroll
    for (index_t row = 0; row < 2; ++row) {
        #pragma unroll
        for (index_t col = 0; col < 2; ++col) {
            T Tpx = my_get_pixel_clamped(localInputs, inXint + col, inYint + row);
            opVal += rs[row] * ps[col] * Convert<T, accum_t>::LeftToRight(Tpx);
        }
    }

    outputs[n][c][outY][outX] = Convert<T, accum_t>::RightToLeft(opVal);
}

template<typename T>
__global__
void indirect_grid_sample_backward_bilinear_kernel(torch::PackedTensorAccessor64<T, 4> inputs,
                                                   torch::PackedTensorAccessor64<T, 4> grid,
                                                   torch::PackedTensorAccessor64<int64_t, 1> inputIndices,
                                                   torch::PackedTensorAccessor64<T, 4> gradOutput,
                                                   torch::PackedTensorAccessor64<T, 4> gradInput,
                                                   torch::PackedTensorAccessor64<T, 4> gradGrid)
{
    typedef typename fp_promote<T>::type accum_t;
    constexpr T NEG_ONE = -1;
    constexpr T ONE = 1;

    const int64_t n = blockDim.z * blockIdx.z + threadIdx.z;

    if (n >= inputIndices.size(0)) return;

    const int64_t c = blockDim.y * blockIdx.y + threadIdx.y;

    const int64_t z = blockDim.x * blockIdx.x + threadIdx.x;

    const accum_t inputHeight = inputs.size(2);
    const accum_t inputWidth = inputs.size(3);
    const int64_t outputHeight = gradOutput.size(2);
    const int64_t outputWidth = gradOutput.size(3);

    const int64_t outY = z / outputWidth;
    const int64_t outX = z % outputWidth;

    if (outY >= outputHeight) return;

    int64_t inputIdx = inputIndices[n];
    const float2 f2uv = *reinterpret_cast<const float2*>(grid[n][outY][outX].data());
    float u = f2uv.x;
    float v = f2uv.y;

    // No output gradient contribution from this position
    if (u < NEG_ONE || u > ONE || v < NEG_ONE || v > ONE) {
        return;
    }

    // Denormalize the coordinates
    u = (u + 1) * ((inputWidth - 1) / 2);
    v = (v + 1) * ((inputHeight - 1) / 2);

    // calculate coordinates
    const accum_t inX = u;
    const accum_t inXint = floor(inX);
    const accum_t inXfrac = inX - inXint;

    const accum_t inY = v;
    const accum_t inYint = floor(inY);
    const accum_t inYfrac = inY - inYint;

    accum_t ps[] = { 1 - inXfrac, inXfrac };
    accum_t rs[] = { 1 - inYfrac, inYfrac };

    const accum_t gOut = Convert<T, accum_t>::LeftToRight(gradOutput[n][c][outY][outX]);

    #pragma unroll
    for (size_t row = 0; row < 2; ++row) {
        #pragma unroll
        for (size_t col = 0; col < 2; ++col) {
            T &gIn = utils::get_pixel_clamped(gradInput, inputIdx, c, inXint + col, inYint + row);

            T gContrib = Convert<T, accum_t>::RightToLeft(rs[row] * ps[col] * gOut);

            atomicAdd(&gIn, gContrib);
        }
    }
}

torch::Tensor gpu_indirect_grid_sample_forward(torch::Tensor input, torch::Tensor grid, torch::Tensor inputIndices, const std::string &method)
{
    auto output = input.new_empty({ inputIndices.size(0), input.size(1), grid.size(1), grid.size(2) });


    if (method != "bilinear"s) {
        throw runtime_error("Only 'bilinear' sampling is currently supported!");
    }

    if (input.size(0) == 1 && input.is_contiguous() && grid.is_contiguous()) {
        uint32_t gridNumCells = grid.size(1) * grid.size(2);
        dim3 blockDim(32, 3, 1);
        dim3 gridDim(div_up(gridNumCells, blockDim.x),
                     div_up(input.size(1), blockDim.y),
                     div_up(grid.size(0), blockDim.z));
        single_ex_grid_sample_bilinear_kernel KERNEL_ARG2(gridDim, blockDim) (
            input.data_ptr<float>(),
            input.size(2), input.size(3), input.size(1),
            reinterpret_cast<const float2*>(grid.data_ptr()),
            gridNumCells,
            output.data_ptr<float>()
        );

    } else {
        // z is batch idx
        // y is channel
        // x is w*h
        dim3 blockDim(32, 1, 3);
        dim3 gridDim(div_up(grid.size(1) * grid.size(2), blockDim.x),
                        div_up(input.size(1), blockDim.y),
                        div_up(inputIndices.size(0), blockDim.z));
        indirect_grid_sample_forward_bilinear_kernel KERNEL_ARG2(gridDim, blockDim) (
            input.packed_accessor32<float, 4>(),
            grid.packed_accessor32<float, 4>(),
            inputIndices.packed_accessor32<int64_t, 1>(),
            output.packed_accessor32<float, 4>()
        );
    }

    //AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    //    input.scalar_type(),
    //    "gpu_indirect_grid_sample_forward",
    //    ([&] {
    //        typedef typename remap_half<scalar_t>::type T;
    //        // typedef scalar_t T;
    //        if (method == "bilinear") {
    //            indirect_grid_sample_forward_bilinear_kernel KERNEL_ARG2(gridDim, blockDim) (
    //                input.packed_accessor64<T, 4>(),
    //                grid.packed_accessor64<T, 4>(),
    //                inputIndices.packed_accessor64<int64_t, 1>(),
    //                output.packed_accessor64<T, 4>()
    //            );
    //        } else {
    //            throw runtime_error("Unsupported resample method: " + method);
    //        }
    //    })
    //);

    return output;
}

std::vector<torch::Tensor> gpu_indirect_grad_sample_backward(torch::Tensor gradOutput, torch::Tensor input, torch::Tensor grid, torch::Tensor inputIndices, const std::string &method)
{
    auto gradInput = torch::zeros_like(input);
    auto gradGrid = torch::zeros_like(grid);

    // z is batch idx
    // y is channel
    // x is w*h
    dim3 blockDim(32, 1, 1);
    dim3 gridDim(div_up(grid.size(1) * grid.size(2), blockDim.x),
                 div_up(input.size(1), blockDim.y),
                 div_up(inputIndices.size(0), blockDim.z));

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "gpu_indirect_grid_sample_backward",
        ([&] {
            typedef typename remap_half<scalar_t>::type T;
            // typedef scalar_t T;
            if (method == "bilinear") {
                indirect_grid_sample_backward_bilinear_kernel KERNEL_ARG2(gridDim, blockDim) (
                    input.packed_accessor64<T, 4>(),
                    grid.packed_accessor64<T, 4>(),
                    inputIndices.packed_accessor64<int64_t, 1>(),
                    gradOutput.packed_accessor64<T, 4>(),
                    gradInput.packed_accessor64<T, 4>(),
                    gradGrid.packed_accessor64<T, 4>()
                );
            } else {
                throw runtime_error("Unsupported resample method: " + method);
            }
        })
    );

    return { gradInput, gradGrid };
}
