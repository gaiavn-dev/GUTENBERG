// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <random>
#include <algorithm>

#include "../geometry.h"

namespace graph_detection {



struct Edge {
    int32_t A;
    int32_t B;

    Edge() = default;
    Edge(int32_t a, int32_t b) : A(a), B(b) {}

    bool Touches(int32_t idx) const { return A == idx || B == idx; }
    bool Touches(const Edge &other) const;
};

inline
bool edge_touches(const Edge &edge, int32_t vertex) {
    return edge.A == vertex || edge.B == vertex;
}

inline
bool Edge::Touches(const Edge &other) const {
    return edge_touches(other, A) || edge_touches(other, B);
}

typedef Point_<float> Pointf;
typedef AABB_<float> AABBf;
typedef Polygon_<float> Polyf;
typedef std::vector<Pointf> Polyline;

std::vector<Edge> find_bottom(const Polygon_<float> &poly, bool useVertexOrder);

void find_long_edges(const Polygon_<float> &poly, Edge *bottoms, std::vector<Edge> &outLongEdge1, std::vector<Edge> &outLongEdge2);

void split_edge_sequence_by_step(const Polygon_<float> &poly, const std::vector<Edge> &longEdge1, const std::vector<Edge> &longEdge2,
                                 float step, std::vector<Pointf> &outInnerPoints1, std::vector<Pointf> &outInnerPoints2);

std::string print_poly(const Polyf &poly);

template<typename T>
inline
std::vector<T> vec_cumsum(const std::vector<T> &v)
{
    std::vector<T> ret;
    ret.reserve(v.size() + 1);
    ret.push_back(0);
    for (T val : v) {
        ret.push_back(ret.back() + val);
    }
    return ret;
}

template<typename RandEng, typename Fn>
inline
void n_choose_k(size_t n, size_t k, RandEng &randEng, Fn fn)
{
    if (k == 0) return;

    // TODO(mranzinger): This algorithm can be replaced with sampling from a geometric
    // distribution, which drastically reduces the runtime complexity
    for (size_t i = 0; i < n; ++i) {
        size_t leftover = n - i;
        if (leftover <= k) {
            fn(i);
            --k;
        } else {
            float p = std::uniform_real_distribution<float>(0.0f, 1.0f)(randEng);
            float probSample = float{k} / float{leftover};
            if (p < probSample) {
                fn(i);
                --k;
            }
        }
    }
}

template<typename T>
inline T clamp(T val, T minVal, T maxVal) {
    return std::max(std::min(val, maxVal), minVal);
}

inline
Pointf avg_point(const std::vector<Pointf> &points)
{
    return std::accumulate(std::begin(points), std::end(points), Pointf(0,0)) / float(points.size());
}

inline
float vector_sin(const Pointf &pt)
{
    // sin = y / len(pt)
    return pt.Y / (length(pt) + 1e-8);
}

inline
float vector_cos(const Pointf &pt)
{
    // cos = x / len(pt)
    return pt.X / (length(pt) + 1e-8);
}

inline
void vector_cos_sin(const Pointf & pt, float &outCos, float &outSin)
{
    float len = length(pt) + 1e-8;
    outCos = pt.X / len;
    outSin = pt.Y / len;
}

inline
float point_dist_to_line(const Pointf &l1, const Pointf &l2, const Pointf &pt)
{
    auto d = l2 - l1;

    auto lineLen = length(d);

    if (lineLen > 0) {
        float distance = abs(
              d.Y * pt.X
            - d.X * pt.Y
            + l2.X * l1.Y
            - l2.Y * l1.X
        ) / lineLen;
        return distance;
    } else {
        return length(pt - l1);
    }
}

template<typename T>
T find_mode(std::vector<T> &inputs) {
    using std::sort;
    using std::begin;
    using std::end;

    if (inputs.empty()) {
        throw std::runtime_error("Cannot find mode of empty distribution!");
    }

    sort(begin(inputs), end(inputs));

    T currVal = inputs[0];
    size_t currCount = 1;

    T modeVal = inputs[0];
    size_t modeCount = 1;

    auto commitCurr = [&] () {
        if (currCount > modeCount) {
            modeCount = currCount;
            modeVal = currVal;
        }
    };

    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i] == currVal) {
            ++currCount;
        } else {
            // Start of a new value
            commitCurr();

            currCount = 1;
            currVal = inputs[i];
        }
    }

    commitCurr();

    return modeVal;
}

} // namespace graph_detection
