// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "text_region_grouping.h"

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <stack>

#include "../geometry.h"
#include "../common.h"
#include "../scope_timer.h"
#include "../non_maximal_suppression/nms_kd_tree.h"

using namespace std;


vector<vector<int64_t>> relations_to_clusters(const unordered_map<int64_t, int64_t> &lineRelations, int64_t numQuads)
{
    unordered_map<int64_t, int64_t> reverseLookup;
    for (auto &kv : lineRelations) {
        reverseLookup.emplace(kv.second, kv.first);
    }

    vector<TextLine> ret;

    unordered_set<int64_t> visited;
    for (auto &kv : lineRelations) {
        int64_t root = kv.first;
        if (visited.count(root)) continue;

        // Find the root
        bool bad = false;
        auto rlIter = reverseLookup.find(root);
        while (rlIter != reverseLookup.end()) {
            root = rlIter->second;
            rlIter = reverseLookup.find(root);
            if (visited.count(root)) {
                bad = true;
                break;
            }
            visited.insert(root);
        }

        // It could be bad either because this node was already visited, or if there's a cycle in the graph (somehow)
        if (bad) continue;

        // Now walk the chain
        TextLine line;
        auto iter = lineRelations.end();
        do
        {
            line.push_back(root);
            visited.insert(root);
            iter = lineRelations.find(root);
            if (iter != lineRelations.end()) {
                root = iter->second;
            }
        } while (iter != lineRelations.end());

        ret.push_back(move(line));
    }

    // Add in all of the stragglers
    for (int64_t i = 0; i < numQuads; ++i) {
        if (! visited.count(i)) {
            TextLine line;
            line.push_back(i);

            ret.push_back(move(line));
        }
    }

    return ret;
}

template<typename T>
inline T default_match(const Quad_<T> &a, const Quad_<T> &query, const Quad_<T> &b)
{
    return std::max<T>(intersection_area(query, b), 0);
}

template<typename T>
inline T height_match(const Quad_<T> &a, const Quad_<T> &query, const Quad_<T> &b)
{
    T aHeight = a.Height();
    T bHeight = b.Height();

    T ratio = aHeight / bHeight;
    if (ratio > 1) {
        ratio = 1 / ratio;
    }

    // Don't combine words that have very different heights
    if (ratio < 0.5) {
        return 0;
    }

    T dfMatch = default_match(a, query, b);
    return dfMatch * ratio;
}

template<typename T, typename CtorFn, typename MatchFn>
vector<vector<int64_t>> cluster_quads(const vector<Quad_<T>> &vQuads, CtorFn queryConstructor, MatchFn matchFn)
{
    torch::Tensor tAllIxAreas = torch::zeros({ (int)vQuads.size(), (int)vQuads.size() }, torch::kFloat32);
    auto accAllIxAreas = tAllIxAreas.accessor<float, 2>();

    NMS_KDTree<Quad_<T>> kdTree;
    kdTree.Build(vQuads);

    for (int64_t i = 0; i < vQuads.size(); ++i) {
        for (int64_t direction = 0; direction < 2; ++direction) {
            auto queryPts = queryConstructor(i, direction);
            Quad_<T> queryQuad{ queryPts.data() };

            kdTree.FindIntersections(queryQuad,
                [i, &accAllIxAreas, &vQuads, &queryQuad, &matchFn, direction]
                    (int64_t k, float pctN, float pctM, float bdsIOU)
                {
                    if (i == k) return;

                    auto oI = i, oK = k;
                    if (direction == 1) {
                        swap(oI, oK);
                    }

                    float matchVal = matchFn(vQuads[oI], queryQuad, vQuads[oK]);
                    accAllIxAreas[oI][oK] = max(accAllIxAreas[oI][oK], matchVal);
                }
            );
        }
    }

    torch::Tensor tAllIxIdxs;
    tie(tAllIxAreas, tAllIxIdxs) = torch::sort(tAllIxAreas, /*dim=*/1, /*descending=*/true);

    accAllIxAreas = tAllIxAreas.accessor<float, 2>();
    auto accAllIxIdxs = tAllIxIdxs.accessor<int64_t, 2>();

    stack<tuple<int64_t, int64_t>> idxsToProcess;
    for (int64_t i = 0; i < vQuads.size(); ++i) {
        idxsToProcess.emplace(i, 0);
    }

    unordered_map<int64_t, tuple<int64_t, T, int64_t>> ownerLookup;

    while (! idxsToProcess.empty()) {
        int64_t i, k;
        tie(i, k) = idxsToProcess.top();
        idxsToProcess.pop();

        for (; k < vQuads.size(); ++k) {
            T ixArea = accAllIxAreas[i][k];

            // There will never be a better match, so just stop processing this quad
            if (ixArea == 0) break;

            int64_t oIdx = accAllIxIdxs[i][k];
            auto ownerIter = ownerLookup.find(oIdx);
            // There is no owner for this region yet!
            if (ownerIter == ownerLookup.end()) {
                ownerLookup.emplace(oIdx, make_tuple(i, ixArea, k));
                break;
            } else {
                int64_t exI, exK;
                T exIxArea;
                tie(exI, exIxArea, exK) = ownerIter->second;

                // This quad is a better match, so boot the other one and add it to the stack
                if (ixArea > exIxArea) {
                    ownerIter->second = make_tuple(i, ixArea, k);
                    // Increment the counter for the quad we just booted
                    idxsToProcess.emplace(exI, exK + 1);
                    break;
                }

                // Otherwise, move to the next best match
            }
        }
    }

    unordered_map<int64_t, int64_t> bijection;
    for (auto &kv : ownerLookup) {
        bijection.emplace(get<0>(kv.second), kv.first);
    }

    return relations_to_clusters(bijection, vQuads.size());
}

template<typename T>
vector<TextLine> quads_to_lines(const vector<Quad_<T>> &vQuads, T horizontalTolerance)
{
    auto queryCtor = [&] (int64_t i, int64_t direction) {
        const Quad_<T> &currQuad = vQuads[i];

        // Direction == 0: Box to the right of the word
        // Direction == 1: Box to the left of the word

        Point_<T> d1 = currQuad[1] - currQuad[0];
        Point_<T> d2 = currQuad[2] - currQuad[3];
        Point_<T> dEnd = direction == 0 ? (currQuad[2] - currQuad[1]) : (currQuad[3] - currQuad[0]);

        T w1 = length(d1);
        T w2 = length(d2);
        T endHeight = length(dEnd);
        T width = (w1 + w2) / 2;

        d1 /= w1;
        d2 /= w2;
        dEnd /= endHeight;

        T avgCharWidth = std::max<T>(endHeight * 0.75f, 1.0f);

        Point_<T> endPt = direction == 0 ? currQuad[1] : currQuad[0];

        Point_<T> rp0 = endPt + (T(0.1) * endHeight * dEnd);
        Point_<T> rp1 = endPt + (T(0.9) * endHeight * dEnd);

        if (direction == 1) {
            d1 *= -1.0f;
            d2 *= -1.0f;
        }

        Point_<T> qp1 = rp0 + (avgCharWidth * horizontalTolerance * d1);
        Point_<T> qp2 = rp1 + (avgCharWidth * horizontalTolerance * d2);

        if (direction == 0) {
            // Create an extension of this quad outward horizontally
            array<Point_<T>, 4> pts{ rp0, qp1, qp2, rp1 };

            return pts;
        } else {
            array<Point_<T>, 4> pts{ qp1, rp0, rp1, qp2 };

            return pts;
        }
    };

    return cluster_quads(vQuads, queryCtor, height_match<T>);
}

template<typename T>
PhraseList lines_to_phrases(const vector<Quad_<T>> &vQuads, const vector<TextLine> &lines,
                            T verticalTolerance)
{
    vector<array<Point_<T>, 4>> linesPts;
    for (const TextLine &line : lines) {
        const Quad_<T> &leftQuad = vQuads[line.front()];
        const Quad_<T> &rightQuad = vQuads[line.back()];

        linesPts.push_back({leftQuad[0], rightQuad[1], rightQuad[2], leftQuad[3]});
    }

    vector<Quad_<T>> vLines;
    for (auto &line : linesPts) {
        vLines.emplace_back(line.data());
    }

    auto queryCtor = [&] (int64_t i, int64_t direction) {
        const Quad_<T> &currQuad = vLines[i];

        Point_<T> d1 = currQuad[3] - currQuad[0];
        Point_<T> d2 = currQuad[2] - currQuad[1];

        if (direction == 0) {
            Point_<T> qp1 = currQuad[3] + (verticalTolerance * d1);
            Point_<T> qp2 = currQuad[2] + (verticalTolerance * d2);

            array<Point_<T>, 4> pts{ currQuad[3], currQuad[2], qp2, qp1 };

            return pts;
        } else {
            Point_<T> qp1 = currQuad[0] - (verticalTolerance * d1);
            Point_<T> qp2 = currQuad[1] - (verticalTolerance * d2);

            array<Point_<T>, 4> pts{ qp1, qp2, currQuad[1], currQuad[0] };

            return pts;
        }
    };

    vector<vector<int64_t>> phraseClusters = cluster_quads(vLines, queryCtor, height_match<T>);

    PhraseList phrases;
    for (const vector<int64_t> &lineIdxs : phraseClusters) {
        Phrase phrase;
        for (int64_t lineIdx : lineIdxs) {
            phrase.push_back(lines[lineIdx]);
        }
        phrases.push_back(move(phrase));
    }

    return phrases;
}


template<typename T>
PhraseList process_image(torch::Tensor quads,
                         T horizontalTolerance, T verticalTolerance, bool verbose)
{
    static bool s_timerEnabled = true;

    if (verbose) {
        cout << "Text Grouper - Processing Image..." << endl;
    }

    auto quadsAccess = quads.accessor<T, 3>();

    vector<Quad_<T>> vQuads;
    for (int64_t i = 0; i < quadsAccess.size(0); ++i) {
        vQuads.emplace_back(quadsAccess[i].data());
    }

    double tQuadsToLines, tLinesToPhrases;
    vector<TextLine> lines;
    PhraseList phrases;

    {
        // Step 1: Construct Lines
        CudaStoreTimer t(tQuadsToLines, s_timerEnabled && verbose, false);
        lines = quads_to_lines(vQuads, horizontalTolerance);
    }

    {
        // Step 2: Construct the phrases
        CudaStoreTimer t(tLinesToPhrases, s_timerEnabled && verbose, false);
        phrases = lines_to_phrases(vQuads, lines, verticalTolerance);
    }

    if (s_timerEnabled && verbose) {
        cout << "Text Grouper " << quads.size(0)
            << " - To Lines: " << tQuadsToLines << "ms"
            << ", To Phrases: " << tLinesToPhrases << "ms"
            << endl;
    }

    return phrases;
}


std::vector<PhraseList> text_region_grouping(torch::Tensor sparseQuads, torch::Tensor sparseCounts,
                                             float horizontalTolerance,
                                             float verticalTolerance,
                                             bool verbose)
{
    sparseQuads = sparseQuads.to(torch::kFloat32);
    sparseCounts = sparseCounts.to(torch::kInt64);

    auto countsAccess = sparseCounts.accessor<int64_t, 1>();

    vector<PhraseList> ret;

    int64_t offset = 0, ct = 0;
    for (int64_t i = 0; i < countsAccess.size(0); ++i, offset += ct) {
        ct = countsAccess[i];

        auto currQuads = sparseQuads.slice(0, offset, offset + ct);

        ret.push_back(process_image<float>(currQuads, horizontalTolerance, verticalTolerance, verbose));
    }

    return ret;
}
