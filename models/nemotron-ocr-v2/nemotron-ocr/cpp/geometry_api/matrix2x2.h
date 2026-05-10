// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../geometry.h"


struct contiguous_tag{};

struct transpose_tag{};

template<typename layout_t, uint32_t R, uint32_t C>
struct Matrix2x2_Offset;

template<uint32_t R, uint32_t C>
struct Matrix2x2_Offset<contiguous_tag, R, C>
{
    static const uint32_t OFFSET = R * 2 + C;
};

template<uint32_t R, uint32_t C>
struct Matrix2x2_Offset<transpose_tag, R, C>
{
    static const uint32_t OFFSET = C * 2 + R;
};


template<typename T, typename layout_t, uint32_t R, uint32_t C>
struct Matrix2x2_Indexor
{
    static const uint32_t OFFSET = Matrix2x2_Offset<layout_t, R, C>::OFFSET;

    static T &get(T *data) { return data[OFFSET]; }
    static const T get(const T *data) { return data[OFFSET]; }
};


template<typename T>
struct Matrix2x2
{
    Matrix2x2() = default;
    Matrix2x2(T r0c0, T r0c1, T r1c0, T r1c1)
        : m_data{ r0c0, r0c1, r1c0, r1c1 }
    {
    }
    Matrix2x2(const Point_<T> &r0, const Point_<T> &r1)
        : m_data{ r0.X, r0.Y, r1.X, r1.Y }
    {
    }
    Matrix2x2(const Point_<T> &r0, const Point_<T> &r1, transpose_tag)
        : m_data{ r0.X, r1.X, r0.Y, r1.Y }
    {
    }

    inline T &operator[](uint32_t i) { return m_data[i]; }
    inline const T operator[](uint32_t i) const { return m_data[i]; }

    T m_data[4];
};

template<typename T, typename layout_t>
struct Matrix2x2_View
{
    Matrix2x2_View(const Matrix2x2<T> &m) : m_data(m.m_data) {}

    const T *m_data;
};

template<uint32_t R, uint32_t C, typename T, typename layout_t>
const T get(const Matrix2x2_View<T, layout_t> &m)
{
    return Matrix2x2_Indexor<T, layout_t, R, C>::get(m.m_data);
}

template<typename T, typename get_pt_t, typename callback_t, typename layout_t = contiguous_tag>
inline
void matmul_fn(int64_t N, const get_pt_t &get_fn, const Matrix2x2<T> &mat, const callback_t &callback,
               layout_t lt = layout_t{})
{
    Matrix2x2_View<T, layout_t> m{ mat };

    #pragma omp simd
    for (int64_t i = 0; i < N; ++i) {
        Point_<T> pt = get_fn(i);

        T x = pt.X * get<0, 0>(m) + pt.Y * get<1, 0>(m);
        T y = pt.X * get<0, 1>(m) + pt.Y * get<1, 1>(m);

        callback(i, Point_<T>{ x, y });
    }
}
