// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_api.h"

#include "../graph_detection/encode_util.h"

#include "../geometry.h"
#include "matrix2x2.h"

using namespace std;

template<typename T>
void _calc_poly_min_rrect(const torch::TensorAccessor<T, 2> vertices, torch::TensorAccessor<T, 2> outRRect);
template<typename T>
void _calc_quad_min_rrect(const torch::TensorAccessor<T, 2> vertices, torch::TensorAccessor<T, 2> outRRect);

torch::Tensor calc_poly_min_rrect(torch::Tensor vertices)
{
    if (vertices.size(0) < 3) {
        throw runtime_error("Invalid polygon! Expected >= 3 vertices, got " + to_string(vertices.size(0)));
    }

    auto ret = torch::empty({ 4, 2 }, vertices.options());

    auto retAcc = ret.accessor<float, 2>();

    if (vertices.size(0) != 4) {
        // OpenCV requires this to be a contiguous buffer
        vertices = vertices.contiguous();
        _calc_poly_min_rrect(vertices.accessor<float, 2>(), retAcc);
    } else {
        _calc_quad_min_rrect(vertices.accessor<float, 2>(), retAcc);
    }

    return ret;
}


template<typename T>
void _calc_bounds(const torch::TensorAccessor<T, 2> &vertices, torch::TensorAccessor<T, 2> &outRRect,
                  const Point_<T> &leftCenter, const Point_<T> &rightCenter)
{
    typedef Point_<T> Pointf;

    Pointf vecAlong = rightCenter - leftCenter;
    auto alongMag = length(vecAlong);

    if (alongMag == 0.0f) {
        throw runtime_error("Invalid polygon!");
    }

    vecAlong /= alongMag;

    Pointf dOrtho{ -vecAlong.Y, vecAlong.X };

    Pointf center = (leftCenter + rightCenter) / 2.0f;

    Matrix2x2<T> rotMat{ vecAlong, dOrtho };

    auto get_fn = [&vertices, &center] (int64_t i) {
        return Pointf{ vertices[i] } - center;
    };

    // All we care about it getting the bounds in the normalized space, so this saves
    // us from having to do any memory allocation
    Pointf minPt{ 0, 0 }, maxPt{ 0, 0 };
    auto tx_fn = [&minPt, &maxPt] (int64_t i, const Pointf &pt) {
        minPt = min(minPt, pt);
        maxPt = max(maxPt, pt);
    };

    matmul_fn(vertices.size(0), get_fn, rotMat, tx_fn, transpose_tag{});

    Pointf rotBox[4] = {
        minPt,
        { maxPt.X, minPt.Y },
        maxPt,
        { minPt.X, maxPt.Y }
    };

    auto get_fn2 = [&rotBox] (int64_t i) {
        return rotBox[i];
    };

    auto assign_fn = [&center, &outRRect] (int64_t i, const Pointf &pt) {
        outRRect[i][0] = pt.X + center.X;
        outRRect[i][1] = pt.Y + center.Y;
    };

    matmul_fn(4, get_fn2, rotMat, assign_fn, contiguous_tag{});
}


template<typename T>
void _calc_poly_min_rrect(const torch::TensorAccessor<T, 2> vertices, torch::TensorAccessor<T, 2> outRRect)
{
    typedef Point_<T> Pointf;
    typedef Polygon_<T> Polygonf;

    Polygonf poly{ vertices.data(), vertices.size(0) };

    vector<graph_detection::Edge> bottoms = graph_detection::find_bottom(poly, false);

    if (bottoms.size() != 2) {
        throw runtime_error("Invalid polygon!");
    }

    vector<graph_detection::Edge> longEdges[2];
    graph_detection::find_long_edges(poly, bottoms.data(), longEdges[0], longEdges[1]);

    ////
    // Determine which edge is above the other
    Pointf cpts[2];
    for (size_t i = 0; i < 2; ++i) {
        auto &pedge = longEdges[i];

        cpts[i] = Pointf{0.0f, 0.0f};
        float ct = 0;
        for (size_t z = 0; z < pedge.size(); ++z) {
            auto edge = pedge[z];
            Pointf p1 = poly[edge.A];
            Pointf p2 = poly[edge.B];
            cpts[i] += (p1 + p2) / 2.0f;
            ct += 1.0f;
        }

        if (ct < 1.0f) {
            throw runtime_error("Edge was empty!");
        }
        cpts[i] /= ct;
    }

    float vpp = graph_detection::vector_sin(cpts[0] - cpts[1]);
    if (vpp >= 0) {
        swap(bottoms[0], bottoms[1]);
    }
    ////

    Pointf edge1[2] = { poly[bottoms[0].A], poly[bottoms[0].B] };
    Pointf edge2[2] = { poly[bottoms[1].A], poly[bottoms[1].B] };

    Pointf c0 = (edge1[0] + edge1[1]) / 2.0f;
    Pointf c1 = (edge2[0] + edge2[1]) / 2.0f;

    _calc_bounds(vertices, outRRect, c0, c1);
}

template<typename T>
void _calc_quad_min_rrect(const torch::TensorAccessor<T, 2> vertices, torch::TensorAccessor<T, 2> outRRect)
{
    typedef Point_<T> Pointf;

    // Instead of finding an arbitrary rotated box, find a reasonable
    // fit for the quadrangle
    Pointf pts[4] = {
        vertices[0], vertices[1], vertices[2], vertices[3]
    };

    Pointf c0 = (pts[0] + pts[3]) / 2.0f;
    Pointf c1 = (pts[1] + pts[2]) / 2.0f;

    _calc_bounds(vertices, outRRect, c0, c1);
}
