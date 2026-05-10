// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

#include "nms_common.h"
#include "../geometry.h"

/*
* \brief Result type for non-maximal suppression.
*
* The results are flattened across the batch, use the third value (region counts) to determine which
* example a quad is associated with.
*
* N - Total number of quads for the entire batch
* B - Batch size
*
* 0 - quads - Nx4x2
* 1 - confidence - N
* 2 - regionCounts - B (s.t. sum(regionCounts) == N)
*/
typedef std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> nms_result_t;

nms_result_t quad_non_maximal_suppression(
    torch::Tensor quads, torch::Tensor probs,
    float probThreshold, float iouThreshold,
    int64_t kernelHeight, int64_t kernelWidth,
    int64_t maxRegions,
    bool verbose = false);


template<typename T>
struct TrackedInPlaceQuad_ : InPlaceQuad_<T> {
    Point_<int64_t> ImgCoords;

    TrackedInPlaceQuad_(Point_<int64_t> imgCoords) : ImgCoords(std::move(imgCoords)) {}
    TrackedInPlaceQuad_(int64_t row, int64_t col) : ImgCoords(col, row) {}
};

template<typename T>
struct TrackedEmbedQuad_ : EmbedQuad_<T> {
    std::vector<Point_<int64_t>> ImgCoords;

    TrackedEmbedQuad_(T confidence = 0): EmbedQuad_<T>(confidence) {}
    TrackedEmbedQuad_(const TrackedEmbedQuad_ &other) = default;

    void swap(TrackedEmbedQuad_ &other) noexcept {
        using std::swap;

        swap(ImgCoords, other.ImgCoords);

        EmbedQuad_<T>::swap(other);
    }

    TrackedEmbedQuad_(TrackedEmbedQuad_ &&other) : TrackedEmbedQuad_() {
        other.swap(*this);
    }

    TrackedEmbedQuad_ &operator=(TrackedEmbedQuad_ other) {
        other.swap(*this);
        return *this;
    }

    void Append(const TrackedInPlaceQuad_<T> &q, T conf, T numQuads = 1) {
        ImgCoords.push_back(q.ImgCoords);

        EmbedQuad_<T>::Append(q, conf, numQuads);
    }

    void Append(const TrackedEmbedQuad_<T> &other) {
        ImgCoords.insert(end(ImgCoords), begin(other.ImgCoords), end(other.ImgCoords));

        EmbedQuad_<T>::Append(other);
    }

    void Reset() {
        ImgCoords.clear();

        EmbedQuad_<T>::Reset();
    }
};

typedef TrackedInPlaceQuad_<float> TIPQuad;
typedef TrackedEmbedQuad_<float> TEFQuad;


std::vector<TEFQuad> reduced_quad_non_maximal_suppression(
    const std::vector<TIPQuad> &rowQuads, float iouThreshold, int64_t imageHeight, int64_t imageWidth);

std::vector<torch::Tensor> quad_non_maximal_suppression_backward(
    torch::Tensor quads, torch::Tensor probs,
    torch::Tensor gradOutQuads, torch::Tensor gradOutProbs);

nms_result_t cuda_quad_non_maximal_suppression(
    torch::Tensor quads, torch::Tensor probs,
    float probThreshold, float iouThreshold,
    int64_t kernelHeight, int64_t kernelWidth,
    int64_t maxRegions,
    bool verbose);
