// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(__INTELLISENSE__) || !defined(__NVCC__)
#ifndef KERNEL_ARG2
#define KERNEL_ARG2(grid, block)
#define KERNEL_ARG3(grid, block, sh_mem)
#define KERNEL_ARG4(grid, block, sh_mem, stream)
#define __global__
#define __device__
#define __host__
#endif
#endif

#ifdef __INTELLISENSE__
#define __CUDACC__
#include <cuda_runtime.h>

void __syncthreads();  // workaround __syncthreads warning

dim3 threadIdx;
dim3 blockIdx;
dim3 blockDim;
dim3 gridDim;

#else
#ifndef KERNEL_ARG2
#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#endif
#endif

#define __any_device__ __host__ __device__

#ifdef __NVCC__
#define __lib_inline__ __forceinline__

#else
#define __lib_inline__ inline
#endif

template<typename T1, typename T2>
__any_device__
inline auto div_up(T1 n, T2 d)
{
    return (n + d - 1) / d;
}
