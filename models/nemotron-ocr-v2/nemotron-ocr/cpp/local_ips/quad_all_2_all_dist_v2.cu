// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include <iostream>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "local_ips.h"
#include "../cuda_intellisense.cuh"
#include "../common.h"
#include "../geometry.h"

using namespace std;
namespace cg = cooperative_groups;

typedef Point_<float> Pointf;

__device__ inline
float square(float val) { return val * val; }

__global__
void device_quad_all_2_all_distance_v2(torch::PackedTensorAccessor64<float, 4> allEmbedQuads,
                                       torch::PackedTensorAccessor64<int64_t, 1> allRegionCounts,
                                       torch::PackedTensorAccessor64<int64_t, 1> csWorkPerExample,
                                       torch::PackedTensorAccessor64<float, 3> outDistances,
                                       float xFactor, float yFactor,
                                       bool allowSelfDistance)
{
    // Note that the blockIdx.x is on purpose here
    int64_t workIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if (workIdx >= csWorkPerExample[csWorkPerExample.size(0) - 1]) return;

    auto exIter = thrust::upper_bound(thrust::seq,
                                      csWorkPerExample.data(), csWorkPerExample.data() + csWorkPerExample.size(0),
                                      workIdx);

    const int64_t exIdx = exIter - csWorkPerExample.data();

    const int64_t workStart = exIdx == 0 ? 0 : csWorkPerExample[exIdx - 1];
    const int64_t workOff = workIdx - workStart;

    const int64_t row = workOff / allRegionCounts[exIdx];
    const int64_t col = workOff % allRegionCounts[exIdx];

    auto taRowQuad = allEmbedQuads[exIdx][row];
    auto taColQuad = allEmbedQuads[exIdx][col];

    Quad_<float> rowQuad(taRowQuad.data()),
                 colQuad(taColQuad.data());

    auto p1 = (rowQuad[0] + rowQuad[3]) / 2.0f;
    auto p2 = (rowQuad[1] + rowQuad[2]) / 2.0f;

    auto vX = p2 - p1;
    auto lenVX = length(vX);
    if (lenVX > 0) {
        vX = vX / max(lenVX, 1e-8f);
    } else {
        vX = { 1, 0 };
    }

    Pointf vY{ -vX.Y, vX.X };

    auto reproj = [&vX, &vY, xFactor, yFactor] (const Pointf &pt) {
        auto dX = dot(pt, vX);
        if (dX >= 0) {
            dX *= xFactor;
        }
        auto dY = dot(pt, vY);
        if (dY >= 0) {
            dY *= yFactor;
        }

        return Pointf{ dX, dY };
    };

    auto tile16 = cg::tiled_partition<16>(cg::this_thread_block());

    // Figure out which vertices this thread is processing
    const int64_t rowVertexIdx = tile16.thread_rank() / 4;
    const int64_t colVertexIdx = tile16.thread_rank() % 4;

    float dist;
    if (row != col) {
        Segment_<float> rowSeg{ rowQuad[rowVertexIdx], rowQuad[(rowVertexIdx + 1) % 4] };
        Segment_<float> colSeg{ colQuad[colVertexIdx], colQuad[(colVertexIdx + 1) % 4] };

        Segment_<float> minSeg = shortest_line_between_segments(rowSeg, colSeg);

        Point_<float> vSeg = minSeg.B - minSeg.A;

        vSeg = reproj(vSeg);

        dist = length(vSeg);
    } else if (allowSelfDistance) {
        dist = 0;
    } else {
        dist = numeric_limits<float>::infinity();
    }

    // Now find the minimum distance across the group
    int lane = tile16.thread_rank();
    // Each iteration halves the number of active threads
    // Each thread gets the partial min[i] to min[lane+i]
    #pragma unroll
    for (uint32_t i = 1; i < 16; i <<= 1) {
        auto otherDist = tile16.shfl_down(dist, i);
        dist = min(dist, otherDist);
    }

#ifndef NDEBUG
    float lowestDist = tile16.shfl(dist, 0);
    assert(dist >= lowestDist);
#endif

    if (lane == 0) {
        outDistances[exIdx][row][col] = dist;
    }
}

torch::Tensor ragged_quad_all_2_all_distance_v2(torch::Tensor embedQuads, torch::Tensor regionCounts,
                                                float xFactor, float yFactor,
                                                bool allowSelfDistance)
{
    if (!embedQuads.is_contiguous()) {
        throw std::runtime_error("Expected `embedQuads` to be contiguous!");
    }

    auto outDistances = torch::zeros({ embedQuads.size(0), embedQuads.size(1), embedQuads.size(1) },
                                     embedQuads.options());

    if (embedQuads.numel() == 0) {
        return outDistances;
    }

    auto workPerExample = regionCounts * regionCounts;

    auto csWorkPerExample = torch::cumsum(workPerExample, 0);

    int64_t totalWork = csWorkPerExample[-1].item<int64_t>();

    dim3 blockSize(16, 2);
    dim3 gridSize(div_up(totalWork, blockSize.y), 1);

    device_quad_all_2_all_distance_v2 KERNEL_ARG2(gridSize, blockSize) (
        embedQuads.packed_accessor64<float, 4>(),
        regionCounts.packed_accessor64<int64_t, 1>(),
        csWorkPerExample.packed_accessor64<int64_t, 1>(),
        outDistances.packed_accessor64<float, 3>(),
        xFactor, yFactor,
        allowSelfDistance
    );

    return outDistances;
}
