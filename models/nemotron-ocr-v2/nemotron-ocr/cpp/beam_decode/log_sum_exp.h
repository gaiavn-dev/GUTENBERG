// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <limits>
#include <algorithm>

typedef float float_t;
extern const float_t NEG_INF;

template<typename T>
inline T max_val(T v)
{
    return v;
}

template<typename T, typename ...Args>
inline T max_val(T v, Args... rest)
{
    auto restMax = max_val(rest...);

    return std::max(v, restMax);
}

template<typename T>
inline T sum_exp(T maxVal, T v)
{
    return std::exp(v - maxVal);
}

template<typename T, typename ...Args>
inline T sum_exp(T maxVal, T v, Args... rest)
{
    auto restSum = sum_exp(maxVal, rest...);

    return sum_exp(maxVal, v) + restSum;
}

template<typename T, typename ...Args>
inline T log_sum_exp(T v, Args ...args)
{
    auto maxVal = max_val(v, args...);

    if (maxVal == -std::numeric_limits<T>::infinity()) {
        return -std::numeric_limits<T>::infinity();
    }

    auto sumExp = sum_exp(maxVal, v, args...);

    return maxVal + std::log(sumExp);
}
