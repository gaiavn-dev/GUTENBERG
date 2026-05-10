// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include "cuda_intellisense.cuh"

#ifndef __CUDACC__
#pragma message("__CUDACC__ not defined!")
#else
#pragma message("__CUDACC__ defined!")
#endif

#ifdef __NVCC__
#define __qr_device__ __device__
#define __qr_host__ __host__
#define __qr_inline__ __forceinline__
#else
#define __qr_device__
#define __qr_host__
#define __qr_inline__ inline
#endif

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


__qr_inline__ __device__ __half operator-(__half v) {
    return __hneg(v);
}

__qr_inline__ __device__ __half operator+(__half a, __half b) {
    return __hadd(a, b);
}

__qr_inline__ __device__ __half operator-(__half a, __half b) {
    return __hsub(a, b);
}

__qr_inline__ __device__ __half operator*(__half a, __half b) {
    return __hmul(a, b);
}

__qr_inline__ __device__ __half operator/(__half a, __half b) {
    return __hdiv(a, b);
}

__qr_inline__ __device__ bool operator==(__half a, __half b) {
    return __heq(a, b);
}

__qr_inline__ __device__ bool operator<(__half a, __half b) {
    return __hlt(a, b);
}

__qr_inline__ __device__ bool operator>(__half a, __half b) {
    return __hgt(a, b);
}

__qr_inline__ __device__ __half sqrt(__half v) {
    return hsqrt(v);
}

__qr_inline__ __device__ __half floor(__half v) {
    return hfloor(v);
}

__qr_inline__ __device__ __half ceil(__half v) {
    return hceil(v);
}

__qr_inline__ __device__ __half max(__half a, __half b) {
    return a > b ? a : b;
}
#endif //__CUDACC__

template<typename Src, typename Dest>
struct Convert {
    __qr_inline__ static __qr_host__ __qr_device__ constexpr Dest From(Src value) { return static_cast<Dest>(value); }
    __qr_inline__ static __qr_host__ __qr_device__ constexpr Src To(Dest value) { return static_cast<Src>(value); }
    __qr_inline__ static __qr_host__ __qr_device__ constexpr Dest LeftToRight(Src value) { return static_cast<Dest>(value); }
    __qr_inline__ static __qr_host__ __qr_device__ constexpr Src RightToLeft(Dest value) { return static_cast<Src>(value); }
};

#ifdef __CUDACC__
template<>
struct Convert<__half, float> {
    __qr_inline__ static __host__ __device__ float From(__half value) { return __half2float(value); }
    __qr_inline__ static __host__ __device__ __half To(float value) { return __float2half(value); }
    __qr_inline__ static __host__ __device__ float LeftToRight(__half value) { return __half2float(value); }
    __qr_inline__ static __host__ __device__ __half RightToLeft(float value) { return __float2half(value); }
};

template<typename Dest>
struct Convert<__half, Dest> : Convert<__half, float> {

};

namespace at {

template<>
inline __half* TensorBase::mutable_data_ptr() const {
    TORCH_CHECK(scalar_type() == ScalarType::Half,
                "expected scalar type Half but found ",
                c10::toString(scalar_type()));
    return static_cast<__half*>(this->unsafeGetTensorImpl()->mutable_data());
}

template<>
inline __half* TensorBase::data_ptr() const {
    TORCH_CHECK(scalar_type() == ScalarType::Half,
                "expected scalar type Half but found ",
                c10::toString(scalar_type()));
    return static_cast<__half*>(this->unsafeGetTensorImpl()->mutable_data());
}

}

template<typename T>
struct remap_half {
    typedef T type;
};

template<>
struct remap_half<at::Half> {
    typedef __half type;
};

template<typename T>
__half to_half(T val) {
    return Convert<__half, T>::RightToLeft(val);
}

template<typename T>
struct fp_promote {
    typedef T type;
};

template<>
struct fp_promote<__half> {
    typedef float type;
};

#endif //__CUDACC__
