// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "non_maximal_suppression.h"

#include <algorithm>
#include "../geometry.h"

using namespace std;


template<typename scalar_t>
void visit_node(
    const torch::TensorAccessor<scalar_t, 4> &quads,
    const torch::TensorAccessor<scalar_t, 2> &probs,
    const torch::TensorAccessor<int32_t, 3> &adjacency,
    MergeQuad_<scalar_t> &mQuad,
    unordered_set<int32_t> &visited,
    int64_t r, int64_t c, int32_t vIdx)
{
    if (visited.count(vIdx)) {
        return;
    }
    visited.insert(vIdx);

    int32_t *pAdj = adjacency[r][c].data();

    int32_t adjCt = pAdj[0];
    assert(adjCt > 0);

    mQuad.Append(Quad_<scalar_t>(quads[r][c].data()), probs[r][c]);

    int32_t *pOff = pAdj + 2;
    int32_t *pEnd = pAdj + adjCt + 1;

    const int32_t W = quads.size(1);

    for (; pOff != pEnd; ++pOff) {
        int32_t vIdx2 = *pOff;
        int32_t r2 = vIdx2 / W;
        int32_t c2 = vIdx2 % W;

        visit_node(quads, probs, adjacency, mQuad, visited, r2, c2, vIdx2);
    }
}

template<typename scalar_t>
std::vector<torch::Tensor> quad_nms_from_adjacency_impl(
    const torch::TensorAccessor<scalar_t, 5> &quads,
    const torch::TensorAccessor<scalar_t, 3> &probs,
    const torch::TensorAccessor<int32_t, 4> &adjacency,
    scalar_t probThreshold, scalar_t iouThreshold,
    int64_t maxRegions)
{
    const uint64_t B = quads.size((int)0);
    const int64_t H = quads.size((int)1);
    const int64_t W = quads.size((int)2);

    typedef MergeQuad_<scalar_t> MQuad;
    typedef EmbedQuad_<scalar_t> EFQuad;

    vector<vector<EFQuad>> batchQuads{ static_cast< const unsigned int >( B ) };
    vector<vector<EFQuad>> allQuads{ static_cast< const unsigned int >( B ) };
    vector<vector<vector<size_t>>> batchAdjIdxs{ static_cast< const unsigned int >( B ) };

    #pragma omp parallel num_threads (8)
    {
        #pragma omp for
        for (int64_t b = 0; b < B; ++b) {
            unordered_set<int32_t> visited;

            for (int64_t r = 0; r < H; ++r) {
                for (int64_t c = 0; c < W; ++c) {
                    auto currProb = probs[b][r][c];

                    if (currProb < probThreshold) {
                        continue;
                    }

                    int32_t vIdx = r * W + c;

                    // Ensure that this quad hasn't already been merged
                    if (visited.count(vIdx)) {
                        continue;
                    }

                    MQuad mQuad{ZeroInitTag{}};
                    visit_node(quads[b], probs[b], adjacency[b], mQuad, visited, r, c, vIdx);

                    batchQuads[b].push_back(mQuad.Commit());
                }
            }
        }

        #pragma omp single
        {
            for (size_t b = 0; b < B; ++b) {
                size_t numQuads = batchQuads[b].size();
                batchAdjIdxs[b].resize(numQuads);
                for (int64_t n = 0; n < numQuads; ++n) {
                    #pragma omp task default(none) shared(batchAdjIdxs, batchQuads, iouThreshold) firstprivate(b, numQuads, n)
                    {
                        for (int64_t m = n + 1; m < numQuads; ++m) {
                            vector<size_t> &adjIdxs = batchAdjIdxs[b][n];
                            vector<EFQuad> &quads = batchQuads[b];
                            auto iou = quads[n].IOU(quads[m]);

                            if (iou > iouThreshold) {
                                adjIdxs.push_back(m);
                            }
                        }
                    }
                }
            }

            #pragma omp taskwait
        }

        #pragma omp for
        for (int64_t batchIdx = 0; batchIdx < B; ++batchIdx) {
            vector<vector<size_t>> &adjIdxs = batchAdjIdxs[batchIdx];
            vector<EFQuad> &quads = batchQuads[batchIdx];
            vector<EFQuad> &finalQuads = allQuads[batchIdx];

            // Step 3: Using depth first search, merge the regions
            unordered_set<size_t> visited;
            for (int64_t n = 0; n < quads.size(); ++n) {
                EFQuad currQuad;
                visit_node(quads, n, adjIdxs, currQuad, visited);

                if (currQuad.NumQuads > 0) {
                    currQuad.Prepare();

                    finalQuads.push_back(currQuad);
                }
            }

            // Only sort the part that we want to keep
            partial_sort(begin(finalQuads),
                        begin(finalQuads) + std::min<int64_t>(finalQuads.size(), maxRegions),
                        end(finalQuads),
                [] (auto a, auto b) {
                    return a.Confidence > b.Confidence;
                }
            );

            // Truncate the low confidence regions
            if (finalQuads.size() > maxRegions) {
                finalQuads.resize(maxRegions);
            }

            //cout << "Ex " << batchIdx << " quads:" << endl << finalQuads << endl << endl;
        }

    } // End parallel

    int64_t numOutQuads = 0;
    for (int64_t batchIdx = 0; batchIdx < B; ++batchIdx) {
        numOutQuads += allQuads[batchIdx].size();
    }

    // Step 4: Convert the quads into tensor representation
    auto outQuadTensor = torch::empty({ numOutQuads, 4, 2 }, torch::kFloat32);
    auto outConfTensor = torch::empty({ numOutQuads }, torch::kFloat32);
    torch::Tensor outCountTensor = torch::empty({ static_cast<int64_t>( allQuads.size() ) }, torch::kInt64);

    auto outQuadAccess = outQuadTensor.accessor<float, 3>();
    auto outConfAccess = outConfTensor.accessor<float, 1>();
    auto outCountAccess = outCountTensor.accessor<int64_t, 1>();

    int64_t offset = 0;
    for (int64_t batchIdx = 0; batchIdx < allQuads.size(); ++batchIdx) {
        vector<EFQuad> &exQuads = allQuads[batchIdx];

        outCountAccess[batchIdx] = exQuads.size();

        for (int64_t qIdx = 0; qIdx < exQuads.size(); ++qIdx, ++offset) {
            copy_quad(exQuads[qIdx], outQuadAccess[offset].data());
            outConfAccess[offset] = exQuads[qIdx].Confidence;
        }
    }

    return { outQuadTensor, outConfTensor, outCountTensor };
}

std::vector<torch::Tensor> quad_nms_from_adjacency(
    torch::Tensor quads, torch::Tensor probs, torch::Tensor adjacency,
    float probThreshold, float iouThreshold,
    int64_t maxRegions)
{
    std::vector<torch::Tensor> ret;

    AT_DISPATCH_FLOATING_TYPES(
        quads.scalar_type(),
        "quad_nms_from_adjacency",
        ([&] {
            ret = quad_nms_from_adjacency_impl<scalar_t>(
                quads.accessor<scalar_t, 5>(),
                probs.accessor<scalar_t, 3>(),
                adjacency.accessor<int32_t, 4>(),
                probThreshold, iouThreshold,
                maxRegions
            );
        })
    );

    return ret;
}
