// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include "../cuda_intellisense.cuh"
#include "../geometry.h"

#if defined(__NVCC__)
#include <math_constants.h>
#define GEO_PI CUDART_PI_F
#else
#include <math.h>
#define GEO_PI M_PI
#endif


template<typename access_t, typename point_t>
__device__
inline
void pt_assign(access_t acc, const point_t &p) {
    acc[0] = p.X;
    acc[1] = p.Y;
}

template<typename T, typename rrect_access_t>
__device__ __lib_inline__
InPlaceQuad_<T> cvt_rrect_to_quad(const rrect_access_t &rrect, T cellSize, T cellOff, T x, T y)
{
    typedef Point_<T> Pointf;

    Pointf prior{
        x * cellSize + cellOff,
        y * cellSize + cellOff
    };

    T dTop = rrect[0];
    T dRight = rrect[1];
    T dBottom = rrect[2];
    T dLeft = rrect[3];
    T theta = rrect[4];

    T piOver2{GEO_PI / 2.0f};
    Pointf vX{ cos(theta), sin(theta) };
    Pointf vY{ cos(theta - piOver2), sin(theta - piOver2) };

    InPlaceQuad_<T> ret;

    ret[0] = prior - vX * dLeft + vY * dTop;
    ret[1] = prior + vX * dRight + vY * dTop;
    ret[2] = prior + vX * dRight - vY * dBottom;
    ret[3] = prior - vX * dLeft - vY * dBottom;

    return ret;
}

template<typename rrect_access_t, typename quad_access_t, typename T>
__device__ __lib_inline__
void assign_rrect_to_quad(const rrect_access_t &rrect, quad_access_t &quad,
                          T cellSize, T cellOff, T x, T y)
{
    const InPlaceQuad_<T> cvQuad = cvt_rrect_to_quad<T>(rrect, cellSize, cellOff, x, y);

    const T *pInQuad = reinterpret_cast<const T*>(&cvQuad);
    T *pOutQuad = reinterpret_cast<T*>(quad.data());

    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        pOutQuad[i] = pInQuad[i];
    }
}

template<typename T, typename rrect_access_t, typename quad_access_t>
__device__
inline
void assign_grad_rrect_to_quad(const rrect_access_t &rrect,
                               const quad_access_t &gradOutput,
                               rrect_access_t gradInput)
{
    typedef Point_<T> Pointf;

    T Top = rrect[0];
    T Right = rrect[1];
    T Bottom = rrect[2];
    T Left = rrect[3];
    T theta = rrect[4];

    T piOver2{GEO_PI / 2.0f};
    Pointf vX{ cos(theta), sin(theta) };
    Pointf vY{ cos(theta - piOver2), sin(theta - piOver2) };

    Pointf dVX{ -vX.Y, vX.X };
    Pointf dVY{ -vY.Y, vY.X };

    Pointf gP0 = gradOutput[0],
           gP1 = gradOutput[1],
           gP2 = gradOutput[2],
           gP3 = gradOutput[3];

    // Top
    gradInput[0] = (gP0 * vY + gP1 * vY).Sum();
    // Right
    gradInput[1] = (gP1 * vX + gP2 * vX).Sum();
    // Bottom
    gradInput[2] = -(gP2 * vY + gP3 * vY).Sum();
    // Left
    gradInput[3] = -(gP0 * vX + gP3 * vX).Sum();

    // Theta
    gradInput[4] = (
        gP0 * (-Left * dVX + Top * dVY) +
        gP1 * (Right * dVX + Top * dVY) +
        gP2 * (Right * dVX - Bottom * dVY) +
        gP3 * (-Left * dVX - Bottom * dVY)
    ).Sum();
}

#undef GEO_PI
