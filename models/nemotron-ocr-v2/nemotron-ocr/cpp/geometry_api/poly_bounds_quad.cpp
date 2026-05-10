// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_api.h"

using namespace std;


template<typename T>
void pt_assign(torch::TensorAccessor<T, 1> acc, T x, T y)
{
    acc[0] = x;
    acc[1] = y;
}


template<typename T>
void poly_bounds_quad_impl(torch::TensorAccessor<T, 2> poly, torch::TensorAccessor<T, 2> outBounds)
{
    T minX = poly[0][0],
      minY = poly[0][1],
      maxX = poly[0][0],
      maxY = poly[0][1];

    const int64_t numVertices = poly.size(0);

    for (int64_t i = 0; i < numVertices; ++i) {
        auto vert = poly[i];

        minX = min(minX, vert[0]);
        maxX = max(maxX, vert[0]);

        minY = min(minY, vert[1]);
        maxY = max(maxY, vert[1]);
    }

    pt_assign(outBounds[0], minX, minY);
    pt_assign(outBounds[1], maxX, minY);
    pt_assign(outBounds[2], maxX, maxY);
    pt_assign(outBounds[3], minX, maxY);
}


torch::Tensor get_poly_bounds_quad(torch::Tensor poly)
{
    auto ret = torch::empty({ 4, 2 }, poly.options());

    AT_DISPATCH_FLOATING_TYPES(
        poly.scalar_type(),
        "poly_bounds_quad_impl",
        ([&] {
            poly_bounds_quad_impl(
                poly.accessor<scalar_t, 2>(),
                ret.accessor<scalar_t, 2>()
            );
        })
    );

    return ret;
}
