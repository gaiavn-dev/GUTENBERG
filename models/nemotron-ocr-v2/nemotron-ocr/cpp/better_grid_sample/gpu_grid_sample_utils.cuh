// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include "../cuda_intellisense.cuh"

#ifndef __NVCC__
#include <algorithm>
#define __device__
#endif

namespace utils {

#ifdef __NVCC__

template<typename T>
__device__ __lib_inline__
T clamp(T val, T minVal, T maxVal)
{
    return max(minVal, min(val, maxVal));
}

#else
using std::clamp;
#endif

template<typename accessor_t>
__device__ __lib_inline__
auto &get_pixel_clamped(accessor_t &inputs,
                            int64_t n, int64_t c, int64_t x, int64_t y)
{
    x = clamp<decltype(x)>(x, 0, inputs.size(3) - 1);
    y = clamp<decltype(y)>(y, 0, inputs.size(2) - 1);

    return inputs[n][c][y][x];
}

}
