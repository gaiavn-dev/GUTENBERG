// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>
#include <unordered_set>

#include "../geometry.h"
#include "../cuda_intellisense.cuh"
#include "strided_quad.h"



std::vector<torch::Tensor> quad_nms_from_adjacency(
    torch::Tensor quads, torch::Tensor probs, torch::Tensor adjacency,
    float probThreshold, float iouThreshold,
    int64_t maxRegions);

template<typename T>
struct EmbedQuad_ : public QuadBase_<T, EmbedQuad_<T> > {
    Point_<T> Vertices[4];
    T Confidence;
    T NumQuads = 0;

    __device__
    EmbedQuad_(T confidence = 0)
    {
        Reset();
        Confidence = confidence;
    }
    __device__
    EmbedQuad_(const EmbedQuad_ &other) = default;

    __device__
    void swap(EmbedQuad_ &other) noexcept {
        using std::swap;

        for (size_t i = 0; i < 4; ++i) {
            swap(Vertices[i], other.Vertices[i]);
        }

        SWAP(Confidence, other.Confidence);
        SWAP(NumQuads, other.NumQuads);
    }

    __device__
    EmbedQuad_(EmbedQuad_ &&other) : EmbedQuad_() {
        other.swap(*this);
    }

    __device__
    EmbedQuad_ &operator=(EmbedQuad_ other) {
        other.swap(*this);
        return *this;
    }

    __device__
    void Append(const EmbedQuad_ &other) {
        Append(other, other.Confidence, other.NumQuads);
    }

    template<typename Derived>
    __device__
    void Append(const QuadBase_<T, Derived> &q, T conf, T numQuads = 1) {
        Confidence *= NumQuads;

        if (Confidence > 0) {
            for (size_t i = 0; i < 4; ++i) {
                Vertices[i] *= Confidence;
            }
        }

        Confidence += conf * numQuads;

        auto qVertices = static_cast<const Derived *>(&q)->Vertices;
        for (size_t i = 0; i < 4; ++i) {
            Vertices[i] += conf * numQuads * qVertices[i];
            Vertices[i] /= Confidence;
        }

        NumQuads += numQuads;
        Confidence /= NumQuads;
    }

    __device__
    void Prepare() {
        // T factor = 1.0 / Confidence;
        // for (size_t i = 0; i < 4; ++i) {
        //     Vertices[i] *= factor;
        // }
        // Confidence /= numQuads;
    }

    __device__
    void Reset() {
        for (size_t i = 0; i < 4; ++i) {
            Vertices[i] = Point_<T>{0, 0};
        }
        Confidence = 0.0f;
        NumQuads = 0;
    }

    __device__
    const Point_<T> &operator[](size_t v) const { return Vertices[v]; }
    __device__
    Point_<T> &operator[](size_t v) { return Vertices[v]; }
};

struct ZeroInitTag {};

template<typename T>
struct MergeQuad_ : public QuadBase_<T, MergeQuad_<T>> {
    Point_<T> Vertices[4];
    T Confidence;
    T NumQuads;

    MergeQuad_() = default;

    __device__
    MergeQuad_(ZeroInitTag) : Confidence(0), NumQuads(0) {
        for (size_t i = 0; i < 4; ++i) {
            Vertices[i] = Point_<T>{0, 0};
        }
    }

    template<typename Derived>
    __device__
    void Append(const QuadBase_<T, Derived> &q, T conf) {
        Confidence += conf;
        ++NumQuads;

        auto &d = static_cast<const Derived&>(q);
        for (size_t i = 0; i < 4; ++i) {
            Vertices[i] += conf * d[i];
        }
    }
    __device__
    void Append(const EmbedQuad_<T> &q) {
        T qConf = q.NumQuads * q.Confidence;

        Confidence += qConf;
        NumQuads += q.NumQuads;
        for (size_t i = 0; i < 4; ++i) {
            Vertices[i] += qConf * q.Vertices[i];
        }
    }
    __device__
    void Append(const StridedEmbedQuad_<T> &q) {
        const T numQuads = q.NumQuads();
        const T qConf = numQuads * q.Confidence();

        Confidence += qConf;
        NumQuads += numQuads;
        for (size_t i = 0; i < 4; ++i) {
            Vertices[i] += qConf * q[i];
        }
    }

    __device__
    EmbedQuad_<T> Commit() {
        EmbedQuad_<T> ret;
        for (size_t i = 0; i < 4; ++i) {
            ret.Vertices[i] = Vertices[i] / Confidence;
        }
        ret.Confidence = Confidence / NumQuads;
        ret.NumQuads = NumQuads;

        return ret;
    }

    __device__
    const Point_<T> &operator[](size_t v) const { return Vertices[v]; }
    __device__
    Point_<T> &operator[](size_t v) { return Vertices[v]; }
};

template<typename T, typename Intermediate=float>
__device__
inline T triangle_root(T val)
{
    // It's easier to visualize this algorithm for a lower triangular matrix
    // What we're trying to find is the `row` of a lower triangular matrix that a given `val` resides in.
    // e.g. 0->0, 2->1, 4->2, etc.
    //
    // 0: 0
    // 1: 1 2
    // 2: 3 4 5
    // 3: 6 7 8 9
    //
    // See https://math.stackexchange.com/questions/698961/finding-the-triangular-root-of-a-number for explanation
    Intermediate numer = Intermediate(-1) + sqrt(Intermediate(1) + Intermediate(8) * Intermediate(val));
    Intermediate denom = Intermediate(2);

    Intermediate ret = floor(numer / denom);
    return T(ret);
}

template<typename T>
void visit_node(const std::vector<EmbedQuad_<T>> &allQuads, size_t quadIdx,
                const std::vector<std::vector<size_t>> &adjIdxs, EmbedQuad_<T> &currQuad,
                std::unordered_set<size_t> &visited)
{
    if (visited.count(quadIdx) > 0) return;

    const EmbedQuad_<T> &vQuad = allQuads[quadIdx];

    currQuad.Append(vQuad);
    visited.insert(quadIdx);

    for (size_t childIdx : adjIdxs[quadIdx]) {
        visit_node(allQuads, childIdx, adjIdxs, currQuad, visited);
    }
}

template<typename T, typename Derived, typename scalar_t>
void copy_quad(const QuadBase_<T, Derived> &srcQuad, scalar_t *pDest)
{
    auto vertices = static_cast<const Derived*>(&srcQuad)->Vertices;
    for (size_t i = 0; i < 4; ++i) {
        const Point_<T> &v = vertices[i];
        *pDest++ = v.X;
        *pDest++ = v.Y;
    }
}
