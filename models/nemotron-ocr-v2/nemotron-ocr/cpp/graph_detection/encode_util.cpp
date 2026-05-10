// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "encode_util.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "../third_party/clipper/clipper.hpp"

using namespace std;

namespace graph_detection {

template<typename T>
struct Candidate : Edge {
    T C;

    Candidate() = default;
    Candidate(int32_t a, int32_t b, T c) : Edge(a, b), C(c) {}
};

struct DistStruct {
    Candidate<Pointf> A;
    Candidate<Pointf> B;
    float Dist;

    DistStruct() = default;
    DistStruct(Candidate<Pointf> a, Candidate<Pointf> b, float dist) : A(a), B(b), Dist(dist) {}
};

template<typename T>
float vec_cos(const Point_<T> &a, const Point_<T> &b)
{
    return dot(a, b) / (length(a) * length(b) + 1e-8);
}

template<typename T, typename Fn = std::less<T>>
vector<size_t> arg_sort(const vector<T> &vec, Fn comp = Fn())
{
    vector<size_t> ret;
    ret.reserve(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        ret.push_back(i);
    }

    sort(begin(ret), end(ret),
        [&vec, &comp] (size_t idxA, size_t idxB) {
            return comp(vec[idxA], vec[idxB]);
        }
    );

    return ret;
}


float edge_length(const Polygon_<float> &poly, const vector<Edge> &edges);

vector<Edge> find_bottom(const Polygon_<float> &poly, bool useVertexOrder)
{
    if (poly.Count < 4) {
        throw runtime_error("Invalid polygon. Fewer than 4 vertices!");
    }

    // If we trust the source of the geometries, then this saves us both computation,
    // but can also be more reliable since we won't reorder the vertices
    if (useVertexOrder) {
        if ((poly.Count % 2) == 1) {
            throw runtime_error("Can't use trusted vertex order when the vertex count is odd!");
        }
        int32_t halfCt = poly.Count / 2;
        return { { halfCt - 1, halfCt },
                 { static_cast<int32_t>(poly.Count) - 1, 0 } };
    }

    if (poly.Count == 4) {
        float d1 = length(poly[1] - poly[0]) + length(poly[2] - poly[3]);
        float d2 = length(poly[2] - poly[1]) + length(poly[0] - poly[3]);

        if (4 * d1 < d2) {
            return { { 0, 1 }, { 2, 3 } };
        } else {
            return { { 1, 2 }, { 3, 0 } };
        }
    }

    auto idx_wrap = [&poly] (size_t idx) {
        return poly[idx % poly.Count];
    };

    vector<Candidate<float>> candidates;
    for (size_t i = 1; i < (poly.Count + 1); ++i) {
        auto vPrev = idx_wrap(i) - idx_wrap(i - 1);
        auto vNext = idx_wrap(i + 2) - idx_wrap(i + 1);

        // We're looking for the segment where the preceding and following segment
        // essentially travel in opposite directions
        if (vec_cos(vPrev, vNext) < -0.875f) {
            auto currSeg = idx_wrap(i) - idx_wrap(i + 1);
            candidates.emplace_back(i % poly.Count, (i + 1) % poly.Count, length(currSeg));
        }
    }

    if (candidates.size() != 2 || candidates[0].A == candidates[1].B || candidates[0].B == candidates[1].A) {
        // If candidate number < 2, or two bottom are joined, select 2 farthest edge
        vector<Candidate<Pointf>> midList;
        for (size_t i = 0; i < poly.Count; ++i) {
            Pointf midPoint = (idx_wrap(i) + idx_wrap(i + 1)) / 2.0f;
            midList.emplace_back(i, (i + 1) % poly.Count, midPoint);
        }

        vector<DistStruct> distList;

        // Only found one good candidate, so search for the edge that's the furthest from this candidate
        if (candidates.size() == 1) {
            auto idx1a = candidates.back().A;
            auto idx1b = candidates.back().B;
            Candidate<Pointf> cand1{ idx1a, idx1b, (idx_wrap(idx1a) + idx_wrap(idx1b)) / 2.0f };
            for (size_t j = 0; j < poly.Count; ++j) {
                auto &cand2 = midList[j];

                if (cand1.Touches(cand2)) continue;

                float dist = length(cand1.C - cand2.C);
                distList.emplace_back(cand1, cand2, dist);
            }
        } else {
            for (size_t i = 0; i < poly.Count; ++i) {
                for (size_t j = i + 1; j < poly.Count; ++j) {
                    auto &cand1 = midList[i];
                    auto &cand2 = midList[j];

                    if (cand1.Touches(cand2)) continue;

                    float dist = length(cand1.C - cand2.C);
                    distList.emplace_back(cand1, cand2, dist);
                }
            }
        }
        sort(begin(distList), end(distList), [] (auto a, auto b) { return a.Dist < b.Dist; });

        if (distList.empty()) {
            throw runtime_error("No valid bottom candidates found for this polygon!");
        }

        auto &bEdge = distList.back();
        return vector<Edge>{ bEdge.A, bEdge.B };

    } else {
        return vector<Edge>{ candidates[0], candidates[1] };
    }
}

void find_long_edges(const Polygon_<float> &poly, Edge *bottoms, vector<Edge> &outLongEdge1, vector<Edge> &outLongEdge2)
{
    int32_t b1End = bottoms[0].B;
    int32_t b2End = bottoms[1].B;

    int32_t nPoints = poly.Count;

    auto accum_into = [nPoints] (int32_t end1, int32_t end2, vector<Edge> &outEdge) {
        int32_t i = (end1 + 1) % nPoints;
        while ((i % nPoints) != end2) {
            int32_t start = i > 0 ? i - 1 : nPoints - 1;
            int32_t end = i % nPoints;
            outEdge.emplace_back(start, end);
            i = (i + 1) % nPoints;
        }
    };

    accum_into(b1End, b2End, outLongEdge1);
    accum_into(b2End, b1End, outLongEdge2);
}

float edge_length(const Polygon_<float> &poly, const vector<Edge> &edges)
{
    float ret = 0.0f;
    for (const Edge &e : edges) {
        ret += length(poly[e.B] - poly[e.A]);
    }
    return ret;
}

vector<float> edge_lengths(const Polygon_<float> &poly, const vector<Edge> &edges)
{
    if (edges.empty()) {
        throw runtime_error("Found an empty edge!");
    }

    vector<float> ret;
    ret.reserve(edges.size());

    for (const Edge &e : edges) {
        ret.push_back(length(poly[e.B] - poly[e.A]));
    }

    return ret;
}

void split_edge_sequence(const Polygon_<float> &poly, const vector<Edge> &edges,
                         const vector<float> &edgeLengths, float nParts,
                         vector<Pointf> &outPts);

void split_edge_sequence_by_step(const Polygon_<float> &poly, const vector<Edge> &longEdge1, const vector<Edge> &longEdge2,
                                 float step, vector<Pointf> &outInnerPoints1, vector<Pointf> &outInnerPoints2)
{
    auto edgeLengths1 = edge_lengths(poly, longEdge1);
    auto edgeLengths2 = edge_lengths(poly, longEdge2);

    float totalLength = (accumulate(begin(edgeLengths1), end(edgeLengths1), 0.0f) + accumulate(begin(edgeLengths2), end(edgeLengths2), 0.0f)) / 2;

    float nParts = max<float>(ceil(totalLength / step), 2);

    split_edge_sequence(poly, longEdge1, edgeLengths1, nParts, outInnerPoints1);
    split_edge_sequence(poly, longEdge2, edgeLengths2, nParts, outInnerPoints2);
}

void split_edge_sequence(const Polygon_<float> &poly, const vector<Edge> &edges,
                         const vector<float> &edgeLengths, float nParts,
                         vector<Pointf> &outPts)
{
    vector<float> elCumSum = vec_cumsum(edgeLengths);

    float totalLength = elCumSum.back();
    float lengthPerPart = totalLength / (nParts - 1);

    size_t iNumParts = nParts;
    size_t currNode = 0;
    size_t ctr = 0;
    for (float i = 0.0f; ctr < iNumParts; i += 1.0f, ++ctr) {
        float t = min(i * lengthPerPart, totalLength);

        while (t > elCumSum[currNode + 1]) {
            ++currNode;
        }

        Edge currEdge = edges[currNode];
        Pointf e1 = poly[currEdge.A];
        Pointf e2 = poly[currEdge.B];

        float currLen = edgeLengths[currNode];

        Pointf sampledPt;

        if (currLen > 0) {
            float deltaT = t - elCumSum[currNode];
            float ratio = deltaT / currLen;
            sampledPt = e1 + ratio * (e2 - e1);
        } else {
            sampledPt = e1;
        }

        outPts.push_back(sampledPt);
    }
}

string print_poly(const Polyf &poly) {
    ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < poly.Count; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << "(" << poly[i].X << ", " << poly[i].Y << ")";
    }
    oss << "]";
    return oss.str();
}

} // namespace graph_detection
