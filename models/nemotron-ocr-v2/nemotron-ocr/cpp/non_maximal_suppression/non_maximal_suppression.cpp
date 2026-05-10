// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "non_maximal_suppression.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>

#include "../geometry.h"
#include "../common.h"
#include "nms_kd_tree.h"

using namespace std;
namespace ix = torch::indexing;

typedef EmbedQuad_<float> EFQuad;

nms_result_t quad_non_maximal_suppression_cpu_impl(
    torch::Tensor tQuads, torch::Tensor tProbs,
    float probThreshold, float iouThreshold,
    int64_t kernelHeight, int64_t kernelWidth,
    int64_t maxRegions,
    bool verbose)
{
    tQuads = tQuads.to(torch::kFloat32).to(torch::kCPU, /*non_blocking=*/ true);
    tProbs = tProbs.to(torch::kFloat32).to(torch::kCPU, /*non_blocking=*/ true);

    auto tStart = chrono::high_resolution_clock::now();

    cudaDeviceSynchronize();

    auto tData = chrono::high_resolution_clock::now();

    if (maxRegions == 0) {
        maxRegions = numeric_limits<int64_t>::max();
    }

    // B,H,W,4,2
    auto quadsAccess = tQuads.accessor<float, 5>();
    // B,H,W
    auto probsAccess = tProbs.accessor<float, 3>();


    const int64_t B = probsAccess.size(0);

    vector<vector<EFQuad>> allQuads{ (unsigned int)B };
    vector<vector<EFQuad>> batchQuads{ (unsigned int)B };
    vector<vector<vector<size_t>>> batchAdjIdxs{ (unsigned int)B };
    vector<unordered_set<size_t>> batchVisited{ (unsigned int)B };

    vector<NMS_KDTree<EFQuad>> batchKDTrees{ (unsigned int)B };

    decltype(tData) tRowSpan, tBuildKD, tAdjacent;

    // Only enable parallelism if release mode
    #ifndef NDEBUG
    #pragma omp parallel num_threads (8)
    #endif
    {
        // Step 1: Combine quads by row
        // Parallelize on both batch and rows
        #ifndef NDEBUG
        #pragma omp for collapse (2)
        #endif
        for (int64_t batchIdx = 0; batchIdx < B; ++batchIdx) {
            for (int64_t row = 0; row < probsAccess.size(1); ++row) {
                vector<EFQuad>& quads = batchQuads[batchIdx];
                EFQuad currQuad;

                auto commitQuad = [&]() {
                    if (currQuad.NumQuads > 0) {
                        #pragma omp critical
                        {
                            if (quads.size() < maxRegions) {
                                quads.push_back(currQuad);
                            }
                        }
                        currQuad.Reset();
                    }
                };

                for (int64_t col = 0; col < probsAccess.size(2); ++col) {
                    Quad_<float> predQuad{ quadsAccess[batchIdx][row][col].data() };
                    float predConf = probsAccess[batchIdx][row][col];

                    // If we're currently in a span, then merge
                    if (predConf >= probThreshold) {
                        auto iou = currQuad.NumQuads > 0 ? predQuad.IOU_UpperBound(currQuad) : 0;

                        // These two regions aren't mergable. Finalize the current quad, and start a new one
                        if (iou < iouThreshold) {
                            commitQuad();
                        }

                        currQuad.Append(predQuad, predConf);
                    }
                    // Otherwise, commit it if valid
                    else {
                        commitQuad();
                    }
                }

                // Capture any dangling span
                commitQuad();
            }
        }

        #ifndef NDEBUG
        #pragma omp single
        #endif
        {
            tRowSpan = chrono::high_resolution_clock::now();
        }

        #ifndef NDEBUG
        #pragma omp for
        #endif
        for (int64_t batchIdx = 0; batchIdx < B; ++batchIdx) {
            batchKDTrees[batchIdx].Build(batchQuads[batchIdx]);
        }

        static const int64_t TASK_SIZE = 2;

        // Step 2: At this point, we have the set of row-merged quads, so now we
        // apply the real merge algorithm. For this, we start with an adjacency matrix.
        //
        // OMP note: "single" means that only one of the threads in the parallel group will execute this block.
        // We're using tasking here to add a bunch of work to the thread pool that will be processed concurrently.
        #ifndef NDEBUG
        #pragma omp single
        #endif
        {
            tBuildKD = chrono::high_resolution_clock::now();

            for (int64_t batchIdx = 0; batchIdx < B; ++batchIdx) {
                int64_t numQuads = batchQuads[batchIdx].size();
                batchAdjIdxs[batchIdx].resize(numQuads);

                for (int64_t q = 0; q < numQuads; q += TASK_SIZE) {
                    // This defines a task that will be executed in parallel by the pool
                    // OMP note:
                    // "shared" essentially means that we're capturing these variables by reference
                    // "firstprivate" means that we're capturing these variables by value
                    #ifndef NDEBUG
                    #pragma omp task default(none) shared(batchAdjIdxs, batchQuads, batchKDTrees, batchVisited, iouThreshold) firstprivate(batchIdx, numQuads, q)
                    #endif
                    {
                        vector<EFQuad>& quads = batchQuads[batchIdx];
                        auto& kdTree = batchKDTrees[batchIdx];
                        unordered_set<size_t>& visited = batchVisited[batchIdx];

                        for (int64_t n = q, nend = min(numQuads, q + TASK_SIZE); n < nend; ++n) {
                            vector<size_t>& adjIdxs = batchAdjIdxs[batchIdx][n];

                            kdTree.FindIntersections(n,
                                [n, iouThreshold, &quads, &visited, &adjIdxs](size_t m, float bdsPctN, float bdsPctM, float bdsIOU) {
                                    float pctN, pctM, iou;
                                    tie(pctN, pctM, iou) = geometry_region_sizes(quads[n], quads[m]);

                                    // Merge
                                    if (iou >= iouThreshold) {
                                        adjIdxs.push_back(m);
                                        // The next two cases are when one region envelops the other. In this case, take the larger region.
                                        // If iou > 0, then they overlap at least somewhat
                                    }
                                    else if (pctN > 0.8 || pctM > 0.8) {
                                        float nHeight = quads[n].Height();
                                        float mHeight = quads[m].Height();

                                        float ratio = nHeight > mHeight ? mHeight / nHeight : nHeight / mHeight;
                                        // If the two quads are roughly the same height (within 90% of each other), then eliminate the smaller region
                                        if (ratio > 0.9) {
                                            if (pctN > 0.8) {
                                                // M envelops N
                                                #pragma omp critical
                                                // Marking a node as visited will prevent it from being processed during the adjacency collapse phase
                                                visited.insert(n);
                                            }
                                            else if (pctM > 0.8) {
                                                // N envelops M
                                                #pragma omp critical
                                                visited.insert(m);
                                            }
                                        }
                                    }
                                }
                            );
                        }
                    }
                }
            }

            #ifndef NDEBUG
            #pragma omp taskwait
            #endif

            tAdjacent = chrono::high_resolution_clock::now();
        }

        #ifndef NDEBUG
        #pragma omp for
        #endif
        for (int64_t batchIdx = 0; batchIdx < B; ++batchIdx) {
            vector<vector<size_t>>& adjIdxs = batchAdjIdxs[batchIdx];
            vector<EFQuad>& quads = batchQuads[batchIdx];
            vector<EFQuad>& finalQuads = allQuads[batchIdx];
            unordered_set<size_t>& visited = batchVisited[batchIdx];

            // Step 3: Using depth first search, merge the regions
            for (int64_t n = 0; n < quads.size(); ++n) {
                EFQuad currQuad;
                visit_node(quads, n, adjIdxs, currQuad, visited);

                if (currQuad.NumQuads > 0) {
                    currQuad.Prepare();

                    finalQuads.push_back(currQuad);
                }
            }
        }

    } // End parallel

    auto tMerge = chrono::high_resolution_clock::now();

    int64_t numOutQuads = 0;
    for (int64_t batchIdx = 0; batchIdx < B; ++batchIdx) {
        numOutQuads += allQuads[batchIdx].size();
    }

    // Allocate the output tensors in pinned memory because they will be immediately sent back to the GPU
    auto pinnedOpt = torch::TensorOptions().pinned_memory(true);

    // Step 4: Convert the quads into tensor representation
    auto outQuadTensor = torch::empty({ numOutQuads, 4, 2 }, pinnedOpt.dtype(torch::kFloat32));
    auto outConfTensor = torch::empty({ numOutQuads }, pinnedOpt.dtype(torch::kFloat32));
    auto outCountTensor = torch::empty({ (int)allQuads.size() }, pinnedOpt.dtype(torch::kInt64));

    auto outQuadAccess = outQuadTensor.accessor<float, 3>();
    auto outConfAccess = outConfTensor.accessor<float, 1>();
    auto outCountAccess = outCountTensor.accessor<int64_t, 1>();

    int64_t offset = 0;
    for (int64_t batchIdx = 0; batchIdx < allQuads.size(); ++batchIdx) {
        vector<EFQuad>& exQuads = allQuads[batchIdx];

        outCountAccess[batchIdx] = exQuads.size();

        for (int64_t qIdx = 0; qIdx < exQuads.size(); ++qIdx, ++offset) {
            copy_quad(exQuads[qIdx], outQuadAccess[offset].data());
            outConfAccess[offset] = exQuads[qIdx].Confidence;
        }
    }

    if (verbose) {
        auto tWrite = chrono::high_resolution_clock::now();

        typedef chrono::duration<double, std::milli> tp_t;
        tp_t dataElapsed = tData - tStart;
        tp_t rowSpanElapsed = tRowSpan - tData;
        tp_t buildKDElapsed = tBuildKD - tRowSpan;
        tp_t adjacentElapsed = tAdjacent - tBuildKD;
        tp_t mergeElapsed = tMerge - tAdjacent;
        tp_t writeElapsed = tWrite - tMerge;
        tp_t totalElapsed = tWrite - tStart;

        // print_tensor(outCountTensor);
        cout << "NMS " << numOutQuads
            << " - Wait for data: " << dataElapsed.count() << "ms"
            << ", Row Span: " << rowSpanElapsed.count() << "ms"
            << ", Build KD: " << buildKDElapsed.count() << "ms"
            << ", Adjacency: " << adjacentElapsed.count() << "ms"
            << ", Merge: " << mergeElapsed.count() << "ms"
            << ", Write: " << writeElapsed.count() << "ms"
            << ", Total: " << totalElapsed.count() << "ms"
            << endl;
    }

    return { outQuadTensor, outConfTensor, outCountTensor };
}

nms_result_t quad_non_maximal_suppression(
    torch::Tensor tQuads, torch::Tensor tProbs,
    float probThreshold, float iouThreshold,
    int64_t kernelHeight, int64_t kernelWidth,
    int64_t maxRegions,
    bool verbose)
{
    auto nmsFn = tQuads.is_cuda() ?
        cuda_quad_non_maximal_suppression :
        quad_non_maximal_suppression_cpu_impl;

    torch::Tensor quads, confidence, regionCounts;
    tie(quads, confidence, regionCounts) = nmsFn(
        tQuads, tProbs,
        probThreshold, iouThreshold,
        kernelHeight, kernelWidth,
        maxRegions, verbose
    );

#ifndef NDEBUG
    // In debug mode, do cell sorting so that it's easier to see where the quads are
    auto cells = get<0>(quads.min(1)).div_(10).floor_();
    auto maxX = cells.index({ ix::Slice(), 0 }).max();

    cells = maxX * cells.select(1, 1) + cells.select(1, 0);

    // Ensure that we keep them ordered by example
    auto regionIdxs = torch::arange(regionCounts.size(0), cells.options()).repeat_interleave(regionCounts);
    auto cellMax = cells.max();
    cells += cellMax * regionIdxs;

    auto order = torch::argsort(cells);

    quads = quads.index({ order });
    confidence = confidence.index({ order });
#endif

    return { quads, confidence, regionCounts };
}

vector<TEFQuad> reduced_quad_non_maximal_suppression(
    const vector<TIPQuad> &rowQuads, float iouThreshold, int64_t imageHeight, int64_t imageWidth)
{
    // auto tStart = chrono::high_resolution_clock::now();

    vector<TEFQuad> allQuads;

    TEFQuad currQuad;

    auto commitQuad = [&] () {
        if (currQuad.NumQuads > 0) {
            allQuads.push_back(move(currQuad));
        }
        currQuad.Reset();
    };

    for (const auto &thisQuad : rowQuads) {
        auto iou = currQuad.NumQuads > 0 ? thisQuad.IOU_UpperBound(currQuad) : 0;

        // These two regions aren't mergeable. Finalize the current quad, and start a new one
        if (iou < iouThreshold) {
            commitQuad();
        }

        currQuad.Append(thisQuad, 1);
    }

    // Capture any dangling span
    commitQuad();

    const int64_t numQuads = allQuads.size();
    vector<TEFQuad> mergeQuads;
    vector<bool> visited;
    visited.resize(numQuads, false);

    NMS_KDTree<TEFQuad> kdTree;
    kdTree.Build(allQuads);

    for (int64_t row = 0; row < numQuads; ++row) {
        if (visited[row]) continue;

        TEFQuad &rowQuad = allQuads[row];

        kdTree.FindIntersections(row,
            [row, iouThreshold, &rowQuad, &allQuads, &visited] (size_t col, float pctN, float pctM, float iou) {
                if (iou >= iouThreshold && ! visited[col]) {
                    rowQuad.Append(move(allQuads[col]));
                    visited[col] = true;
                }
            }
        );

        mergeQuads.push_back(move(rowQuad));
    }

    // auto tEnd = chrono::high_resolution_clock::now();

    // chrono::duration<double, std::milli> totalElapsed = tEnd - tStart;
    // cout << "Row NMS " << mergeQuads.size() << " - Time: " << totalElapsed.count() << "ms" << endl;

    return mergeQuads;
}
