// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <type_traits>
#include "../half_ops.cuh"
#include "../geometry.h"

// template<typename Point_t>
// __qr_inline__ __qr_device__ auto dot(const Point_t &a, const Point_t &b) -> decltype(a.X) {
//     auto sq = a * b;
//     return sq.X + sq.Y;
// }

// template<typename Point_t>
// __qr_inline__ __qr_device__ auto dot(const Point_t &a) -> decltype(a.X) {
//     return dot(a, a);
// }

template<typename scalar_t>
__qr_inline__ __qr_device__ scalar_t square(scalar_t v) {
    return v * v;
}

template<typename accessor_t>
__qr_inline__ __qr_device__ auto sub_accessor(const accessor_t &a, const accessor_t &b) -> Point_<typename std::remove_pointer<typename accessor_t::PtrType>::type> {
    return { a[0] - b[0], a[1] - b[1] };
}

template<typename scalar_t, typename accessor_t>
__qr_device__ scalar_t calc_quad_width(const accessor_t &quad,
                                       scalar_t outputHeight,
                                       scalar_t roundFactor,
                                       scalar_t maxWidth)
{
    using std::max;
    using std::ceil;
    using std::floor;
    typedef Point_<scalar_t> Point_t;

    Point_t vecWidth = sub_accessor(quad[1], quad[0]);
    Point_t vecHeight = sub_accessor(quad[3], quad[0]);
    Point_t vecHeight2 = sub_accessor(quad[2], quad[1]);

    scalar_t quadWidth = sqrt(dot(vecWidth));
    scalar_t quadHeight = sqrt(dot(vecHeight));
    scalar_t quadHeight2 = sqrt(dot(vecHeight2));

    const scalar_t sc2 = Convert<scalar_t, float>::To(2);
    quadHeight = (quadHeight + quadHeight2) / sc2;

    if (quadHeight < sc2) {
        quadHeight = sc2;
    }

    scalar_t growthRatio = outputHeight / quadHeight;
    quadWidth = growthRatio * quadWidth;

    quadWidth = max(roundFactor, ceil(quadWidth / roundFactor) * roundFactor);

    if (maxWidth > Convert<scalar_t, float>::To(0) && quadWidth > maxWidth) {
        quadWidth = maxWidth;
    }

    return max(sc2, floor(quadWidth));
}

template<typename scalar_t, typename accessor_t>
__qr_inline__
__qr_device__ Point_<scalar_t> calc_rect_value(const accessor_t &quad,
                                               const scalar_t quadWidth,
                                               const scalar_t outputHeight,
                                               const unsigned int x,
                                               const unsigned int y,
                                               const scalar_t imageWidth,
                                               const scalar_t imageHeight)
{
    typedef Point_<scalar_t> Point_t;

    const Point_t pts[4] = {
        quad[0], quad[1], quad[2], quad[3]
    };


    const scalar_t scX = Convert<scalar_t, unsigned int>::RightToLeft(x);
    const scalar_t sc1 = Convert<scalar_t, float>::RightToLeft(1);
    const scalar_t scHalf = Convert<scalar_t, float>::RightToLeft(0.5);

    const scalar_t fRow = (Convert<scalar_t, unsigned int>::RightToLeft(y) + scHalf) / outputHeight;
    const scalar_t fCol = (scX + scHalf) / quadWidth;

    // const scalar_t fRow = Convert<scalar_t, unsigned int>::RightToLeft(y) / (outputHeight - sc1);
    // const scalar_t fCol = scX / (quadWidth - sc1);

    Point_t outputPoint;
    if (scX < quadWidth) {
        const Point_t &q0 = pts[0];
        const Point_t A = pts[1] - q0;
        const Point_t B = pts[3] - q0;
        const Point_t C = pts[2] - pts[1];

        outputPoint = q0
                    + fCol * A
                    + fRow * B
                    + (fCol * fRow) * (C - B);
    }
    else {
        outputPoint = { -sc1, -sc1 };
    }

    outputPoint /= Point_t{ imageWidth, imageHeight };

    // Remap from [0, 1] -> [-1, 1]
    outputPoint = (Convert<scalar_t, float>::RightToLeft(2) * outputPoint) - sc1;

    return outputPoint;
}
