// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../geometry.h"
#include "../cuda_intellisense.cuh"


template<typename T>
struct StridedQuad_ : QuadBase_<T, StridedQuad_<T>>
{
    T *DataPtr = nullptr;
    int64_t Stride = 0;

    StridedQuad_() = default;
    __host__ __device__
    StridedQuad_(T *dataPtr, int64_t stride)
        : DataPtr(dataPtr), Stride(stride) {}

    __host__ __device__
    const Point_<T> operator[](int64_t offset) const {
        auto ptOffset = DataPtr + 2 * offset * Stride;
        return {
            *ptOffset,
            ptOffset[Stride]
        };
    }

    __host__ __device__
    InPlaceQuad_<T> ToIPQuad() const
    {
        InPlaceQuad_<T> ret;
        #pragma unroll
        for (int64_t i = 0; i < 4; ++i) {
            ret.Vertices[i] = (*this)[i];
        }
        return ret;
    }
};

template<typename T>
struct StridedEmbedQuad_ : StridedQuad_<T>
{
    using StridedQuad_<T>::StridedQuad_;

    __host__ __device__ T &Confidence() { return this->DataPtr[8 * this->Stride]; }
    __host__ __device__ const T Confidence() const { return this->DataPtr[8 * this->Stride]; }

    __host__ __device__ T &NumQuads() { return this->DataPtr[9 * this->Stride]; }
    __host__ __device__ const T NumQuads() const { return this->DataPtr[9 * this->Stride]; }
};
