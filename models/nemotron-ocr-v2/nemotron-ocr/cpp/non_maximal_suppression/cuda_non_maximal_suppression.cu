// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "non_maximal_suppression.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <trove/ptr.h>

#include "../cuda_intellisense.cuh"
#include "../geometry.h"
#include "../common.h"
#include "../scope_timer.h"
#include "strided_quad.h"

// If this flag is turned on, then a bunch of checks will be inserted to ensure that the same results are produced by
// successive calls to NMS. This means that it makes the library unusable outside of a debug context, so beware!
//#define NMS_VERIFY_CORRECTNESS

namespace cg = cooperative_groups;
namespace ix = torch::indexing;

inline
void print_tensor_stats2(const std::string &msg, const torch::Tensor& tensor) {

    auto fTensor = tensor.to(torch::kDouble).cpu();

    std::stringstream ss;
    if (tensor.numel() > 1) {
        ss << msg << " Size: " << tensor.sizes() << " Type: " << tensor.dtype() << " Device: " << tensor.device() << " Max: " << fTensor.max().item<double>() << " Min: " << fTensor.min().item<double>() << " Mean: " << fTensor.mean().item<double>() << " Std: " << fTensor.std().item<double>();
    }
    else if (tensor.numel() == 1) {
        ss << msg << " Size: " << tensor.sizes() << " Type: " << tensor.dtype() << " Device: " << tensor.device() << " Value: " << fTensor.item<double>() << std::endl;
    }
    else {
        ss << msg << " Size: " << tensor.sizes() << " Type: " << tensor.dtype() << " Device: " << tensor.device() << std::endl;
    }
    std::cout << ss.str() << std::endl;
}

inline
void print_tensor_vec_stats2(std::string msg, const std::vector<torch::Tensor>& tensorVec) {
    std::cout << msg << " Size: " << tensorVec.size() << std::endl;
    std::stringstream ss;
    msg = "     - ";
    for (int i = 0; i < tensorVec.size(); ++i) {
        ss << msg << "[" << i << "]:";
        auto tensor = tensorVec[i];
        print_tensor_stats2(ss.str(), tensor);
        ss.str("");
    }
}

std::ostream &operator<<(std::ostream &os, dim3 d)
{
    return os << "(" << d.x << ", " << d.y << ", " << d.z << ")";
}

#define ADD_OP2(vector2_t) __device__ \
    vector2_t operator+(const vector2_t &a, const vector2_t &b) { \
        return { a.x + b.x, a.y + b.y }; \
    }
ADD_OP2(float2);
ADD_OP2(double2);
#undef ADD_OP2

#define ADD_OP4(vector4_t) __device__ \
    vector4_t operator+(const vector4_t &a, const vector4_t &b) { \
        return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; \
    }
ADD_OP4(float4);
ADD_OP4(double4);
#undef ADD_OP4

template<typename T, size_t Size>
__device__
std::array<T, Size> operator+(const std::array<T, Size> &a, const std::array<T, Size> &b) {
    std::array<T, Size> ret;
    #pragma unroll
    for (size_t i = 0; i < Size; ++i) {
        ret._Elems[i] = a._Elems[i] + b._Elems[i];
    }
    return ret;
}

#if __CUDA_ARCH__ >= 800
#define __reduce_add_full_warp(val) __reduce_add_sync(0xFFFFFFFF, val)
#define __reduce_max_full_warp(val) __reduce_max_sync(0xFFFFFFFF, val)
#define __reduce_min_full_warp(val) __reduce_min_sync(0xFFFFFFFF, val)
#else
#define __reduce_add_full_warp(val) cg::reduce(cg::tiled_partition<32>(cg::this_thread_block()), val, cg::plus<decltype(val)>())
#define __reduce_max_full_warp(val) cg::reduce(cg::tiled_partition<32>(cg::this_thread_block()), val, cg::greater<decltype(val)>())
#define __reduce_min_full_warp(val) cg::reduce(cg::tiled_partition<32>(cg::this_thread_block()), val, cg::less<decltype(val)>())
#endif

template<typename T>
struct TToVec;
template<>
struct TToVec<float> { typedef float2 type2; typedef float4 type4; };
template<>
struct TToVec<double> { typedef double2 type2; typedef double4 type4; };

template<typename T, typename accessor_t>
__device__
void write_embed_quad(accessor_t &acc, const MergeQuad_<T> &quad, int64_t storeOff)
{
    constexpr auto EMBED_QUAD_SIZE = sizeof(EmbedQuad_<T>) / sizeof(T);
    static_assert(EMBED_QUAD_SIZE == 10, "Unsupported embed quad size!");

    const T *mergeBuff = reinterpret_cast<const T*>(&quad);

    const T confidence = quad.Confidence;
    const auto i = threadIdx.x;

    if (i >= 10) {
        return;
    }

    T outVal;
    // Coordinates
    if (i < 8) {
        outVal = mergeBuff[i] / confidence;
    // Confidence
    } else if (i == 8) {
        outVal = confidence / mergeBuff[9];
    // NumQuads
    } else {
        outVal = mergeBuff[9];
    }

    acc[i][storeOff] = outVal;
}


template<typename group_t, typename ...Args>
__device__
void ordered_print(group_t &group, const char *const fmt, const Args& ...args)
{
    for (uint32_t i = 0; i < group.size(); ++i) {
        if (group.thread_rank() == i) {
            printf(fmt, args...);
        }
        group.sync();
    }
}

template<typename T>
__global__
void device_row_collapse(torch::PackedTensorAccessor64<T, 5> allQuads,
    torch::PackedTensorAccessor64<T, 3> allConfs,
    T confThreshold, T iouThreshold,
    torch::PackedTensorAccessor64<int32_t, 1> allOutCounts,
    torch::PackedTensorAccessor64<T, 3> allOutEmbedQuads,
    torch::PackedTensorAccessor64<int32_t, 2> allOutIds)
{
    typedef InPlaceQuad_<T> Quadf;
    static_assert(sizeof(Quadf) == sizeof(T) * 8, "Invalid QuadMem size!");

    constexpr uint32_t ALL_MASK = 0xFFFFFFFF;
    constexpr uint32_t WARP_SIZE = 32;
    constexpr T MIN_VALID_AREA = 8;

    const uint32_t B = allQuads.size(0);
    const uint32_t H = allQuads.size(1);

    const uint32_t b = blockIdx.z;
    const uint32_t r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= H) {
        return;
    }

    #define threadRank threadIdx.x

    auto rawQuads = reinterpret_cast<Quadf*>(allQuads[b][r].data());
#if defined(NDEBUG)
    trove::coalesced_ptr<Quadf> quads(rawQuads);
#else
    auto quads = rawQuads;
#endif

    auto confs = allConfs[b][r];

    T conf = confs[threadRank];

    bool quadValid = conf >= confThreshold;
    uint32_t ballot = __ballot_sync(ALL_MASK, quadValid);

    // No valid quads in this window, so we're done!
    if (ballot == 0) {
        return;
    }

    const Quadf currQuad = quads[threadRank];

    const T qArea = currQuad.Area();

    quadValid = quadValid && qArea > MIN_VALID_AREA;
    ballot = __ballot_sync(ALL_MASK, quadValid);
    if (ballot == 0) {
        return;
    }
    if (! quadValid) {
        conf = 0;
    }

    MergeQuad_<T> qAccum{ZeroInitTag{}};

    Quadf prevQuad;
    auto pCurrQuad = reinterpret_cast<const T*>(&currQuad);
    auto pPrevQuad = reinterpret_cast<T*>(&prevQuad);
    #pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        pPrevQuad[i] = __shfl_up_sync(ALL_MASK, pCurrQuad[i], 1);
    }
    T prevConf = __shfl_up_sync(ALL_MASK, conf, 1);

    if (threadRank == 0) {
        prevConf = 0;
    }

    bool iouValid = false;
    T iou = 0;
    if (quadValid) {
        qAccum.Append(currQuad, conf);

        if (prevConf >= confThreshold) {
            iou = prevQuad.IOU_UpperBound(currQuad);
            if (iou >= iouThreshold) {
                iouValid = true;
            }
        }
    }

    // This is the start of a span if the current confidence is above threshold, but the quad to the left is either below threshold,
    // or the IOU between the quads is below threshold
    const bool isStartOfSpan = quadValid && !iouValid;

    uint32_t label = isStartOfSpan;
    // All labels start out as 0 or 1, and we'll then do a cumsum over the warp, which gives each thread an assigned label
    // We also know that the final thread also contains the number of labels.
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        auto inc = __shfl_up_sync(ALL_MASK, label, offset);
        if (threadRank >= offset) {
            label += inc;
        }
    }

    // Before we zero out invalid labels, get the total number of labels
    const uint32_t numLabels = __shfl_sync(ALL_MASK, label, WARP_SIZE - 1);

    // Zero out the label if the current quad isn't valid
    label = quadValid ? label : 0;

    T* accumPtr = reinterpret_cast<T*>(&qAccum);
    // Reduce all of the quads s.t. the left-most position in the span contains the full quad.
    // We use `label` to decide whether to do the accumulation
    #pragma unroll
    for (uint32_t offset = 1; offset < 32; offset <<= 1) {
        const auto otherLabel = __shfl_down_sync(ALL_MASK, label, offset);

        // Regardless of whether the labels match, all threads in the warp must make the shfl_down
        // call. So we use factor to modulate whether the given merge is valid
        const T factor = otherLabel == label && offset + threadRank < WARP_SIZE ? 1.0f : 0.0f;

        #pragma unroll
        for (uint32_t i = 0; i < 10; ++i) {
            accumPtr[i] += factor * __shfl_down_sync(ALL_MASK, accumPtr[i], offset);
        }
    }

    // Elect thread-0 to figure out where to store the results
    uint32_t storeOff = 0;
    if (threadRank == 0) {
        storeOff = atomicAdd(&allOutCounts[b], numLabels);
    }
    // Broadcast that offset to the whole warp
    storeOff = __shfl_sync(ALL_MASK, storeOff, 0);

    auto outEmbedQuads = allOutEmbedQuads[b];
    // Now write out each quad, but collectively
    for (uint32_t procLabel = 1; procLabel <= numLabels; ++procLabel) {
        // Discover the index of the start of each label span
        ballot = __ballot_sync(ALL_MASK, procLabel == label);
        // ffs will find the (1-based) index of the least significant bit in ballot.
        // This just so happens to be the start of the span for the current label
        uint32_t startIdx = __ffs(ballot) - 1;

        const T* inT = reinterpret_cast<T*>(&qAccum);
        MergeQuad_<T> outQuad;
        T* outT = reinterpret_cast<T*>(&outQuad);
        #pragma unroll
        for (uint32_t i = 0; i < 10; ++i) {
            outT[i] = __shfl_sync(ALL_MASK, inT[i], startIdx);
        }

        write_embed_quad(outEmbedQuads, outQuad, storeOff + procLabel - 1);
        if (threadRank == 0) {
            allOutIds[b][storeOff + procLabel - 1] = r * 32 + startIdx;
        }
    }

    if (threadRank == 0) {
        // Increment the total number of quads by the number encountered on this row
        atomicAdd(&allOutCounts[B], numLabels);
    }

#undef threadRank
}

template<typename T>
__global__
void device_a2a_adjacency_sparse(const int32_t *ptrQuadCts,
                                 T iouThreshold,
                                 torch::PackedTensorAccessor64<T, 3> embedQuads,
                                 torch::PackedTensorAccessor64<bool, 2> outIsStart,
                                 torch::PackedTensorAccessor64<int32_t, 2> outAdjCounts,
                                 torch::PackedTensorAccessor64<int32_t, 3> outSparseAdj)
{
    const uint32_t b = blockIdx.y;

    const int32_t quadCt = ptrQuadCts[b];

    if (quadCt == 0) {
        return;
    }

    const int32_t jobIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t row = jobIdx / quadCt;
    const int32_t col = jobIdx % quadCt;

    // Only compute the upper triangular portion of the matrix
    if (row >= quadCt || col < row) {
        return;
    }

    T* exData = embedQuads[b].data();

    const auto qRow = StridedEmbedQuad_<T>{ exData + row * embedQuads.stride(2), embedQuads.stride(1) }.Bounds(),
               qCol = StridedEmbedQuad_<T>{ exData + col * embedQuads.stride(2), embedQuads.stride(1) }.Bounds();

    T pctRow, pctCol, iou;
    thrust::tie(pctRow, pctCol, iou) = geometry_region_sizes(qRow, qCol);

    auto warpGroup = cg::tiled_partition<32>(cg::this_thread_block());

    auto rowGroup = cg::labeled_partition(warpGroup, row);

    const bool isValid = iou >= iouThreshold;

    const uint32_t ballot = rowGroup.ballot(isValid);
    const uint32_t numValid = __popc(ballot);

    auto exAdjCounts = outAdjCounts[b].data();

    int32_t storeOff = 0;
    if (numValid > 0 && rowGroup.thread_rank() == 0) {
        storeOff = atomicAdd(exAdjCounts + row, numValid);
    }
    storeOff = rowGroup.shfl(storeOff, 0);

    if (isValid) {
        // This will set all of the bits to the left of this one to 1, otherwise 0.
        // We can use this to count the number of bits that are set, and are less significant than this one,
        // to get the local storage offset
        uint32_t lowerMask = (1 << rowGroup.thread_rank()) - 1;

        storeOff += __popc(ballot & lowerMask);

        outSparseAdj[b][row][storeOff] = col;
        if (row != col) {
            // Because `col` gets merged into `row`, we mark it as inactive for reduction purposes.
            // All of the quads that `col` is adjacent to will be absorbed by `row`.
            outIsStart[b][col] = false;

            // Also store the transposed relation
            storeOff = atomicAdd(exAdjCounts + col, 1);
            outSparseAdj[b][col][storeOff] = row;
        }
    } else if (pctRow > 0.8f || pctCol > 0.8f) {
        T anchorHeight = qRow.Height();
        T otherHeight = qCol.Height();

        T ratio = anchorHeight > otherHeight ?
            otherHeight / anchorHeight :
            anchorHeight / otherHeight;
        if (ratio > 0.9f) {
            if (pctRow > 0.8f) {
                // Other envelops anchor
                outIsStart[b][row] = false;
            }
            else {
                outIsStart[b][col] = false;
            }
        }
    }
}

template<uint32_t NumWarps, typename T, int32_t I_CELL_SIZE>
__global__
void device_a2a_adjacency_build_grid(const int32_t *ptrQuadCts,
                                     torch::PackedTensorAccessor64<T, 3> embedQuads,
                                     torch::PackedTensorAccessor64<int32_t, 4> outGridCells,
                                     torch::PackedTensorAccessor64<int32_t, 3> outQuadCells)
{
    constexpr T MIN_T = std::numeric_limits<T>::min();
    constexpr T MAX_T = std::numeric_limits<T>::max();
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t BLOCK_SIZE = NumWarps * WARP_SIZE;
    constexpr uint32_t FULL_WARP = 0xFFFFFFFF;
    constexpr uint32_t FIRST_16_THREADS = 0x0FFFF;
    constexpr T CELL_SIZE = I_CELL_SIZE;
    constexpr T INV_CELL_SIZE = 1 / CELL_SIZE;

    const uint32_t b = blockIdx.z;

    const uint32_t quadCt = ptrQuadCts[b];
    const uint32_t quadIdx = blockIdx.y;

    if (quadIdx >= quadCt) {
        return;
    }

    const uint32_t threadRank = threadIdx.x;
    const uint32_t localThreadRank = threadRank & 0x1F;

    auto exQuads = embedQuads[b];

    const uint32_t numCells[2] = { outGridCells.size(2), outGridCells.size(1) };

    const uint32_t numRows = outGridCells.size(1);
    const uint32_t numCols = outGridCells.size(2);

    // We use flip so that we can compute min and max simultaneously.
    // First 4 threads compute the min, next 4 compute the max
    T sign = localThreadRank < 8 ? 1.0f : -1.0f;
    T myVal = sign * (localThreadRank < 16 ? exQuads[localThreadRank & 0x7][quadIdx] : MIN_T);
    #pragma unroll
    for (uint32_t offset = 2; offset < 8; offset <<= 1) {
        T nextVal = __shfl_down_sync(FIRST_16_THREADS, myVal, offset);
        myVal = min(myVal, nextVal);
    }
    const uint32_t cellVal = max(0.0f, sign * INV_CELL_SIZE * myVal);

    uint32_t minCell[2] = { __shfl_sync(FULL_WARP, cellVal, 0), __shfl_sync(FULL_WARP, cellVal, 1) },
             maxCell[2] = { __shfl_sync(FULL_WARP, cellVal, 8), __shfl_sync(FULL_WARP, cellVal, 9) };

    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        maxCell[i] = min(numCells[i] - 1, maxCell[i]);
    }

    const uint32_t sizes[2] = { maxCell[0] - minCell[0] + 1, maxCell[1] - minCell[1] + 1 };

    const uint32_t totalCells = sizes[0] * sizes[1];

    auto exGridCells = outGridCells[b];

    for (uint32_t i = threadRank; i < totalCells; i += BLOCK_SIZE) {
        uint32_t row = minCell[1] + i / sizes[0];
        uint32_t col = minCell[0] + i % sizes[0];

        int32_t *pCell = exGridCells[row][col].data();

        // The first value in the array is the count, and the rest are the quad indices
        int32_t storeOff = atomicAdd(pCell, 1) + 1;
        pCell[storeOff] = quadIdx;
    }

    if (threadRank < 2) {
        outQuadCells[b][quadIdx][threadRank] = minCell[threadRank];
    } else if (threadRank < 4) {
        outQuadCells[b][quadIdx][threadRank] = maxCell[threadRank - 2];
    }
}

typedef uint8_t visit_mask_t;

template<uint32_t NumWarps, typename T>
__global__
void device_a2a_adjacency_with_grid(const int32_t *ptrQuadCts,
                                    T iouThreshold,
                                    torch::PackedTensorAccessor64<T, 3> allEmbedQuads,
                                    torch::PackedTensorAccessor64<int32_t, 4> allCells,
                                    torch::PackedTensorAccessor64<int32_t, 3> allQuadExtents,
                                    torch::PackedTensorAccessor64<bool, 2> outIsStart,
                                    torch::PackedTensorAccessor64<int32_t, 2> outAdjCounts,
                                    torch::PackedTensorAccessor64<int32_t, 3> outSparseAdj)
{
    constexpr T MIN_T = std::numeric_limits<T>::min();
    constexpr T MAX_T = std::numeric_limits<T>::max();
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t BLOCK_SIZE = NumWarps * WARP_SIZE;

    const uint32_t b = blockIdx.z;

    const uint32_t quadCt = ptrQuadCts[b];
    const uint32_t quadIdx = blockIdx.y;

    if (quadIdx >= quadCt) {
        return;
    }

    const uint32_t threadRank = threadIdx.x;

    auto exQuads = allEmbedQuads[b];

    __shared__ T s_quadVerts[8];
    __shared__ uint32_t s_quadExtent[4];
    extern __shared__ uint32_t s_alreadyVisited[];

    if (threadRank < 8) {
        s_quadVerts[threadRank] = exQuads[threadRank][quadIdx];
    } else if (threadRank < 12) {
        s_quadExtent[threadRank - 8] = reinterpret_cast<uint32_t*>(allQuadExtents[b][quadIdx].data())[threadRank - 8];
    }

    uint32_t zeroTerm = (quadCt + 31u) >> 5u; // Fast version of div_up(quadCt, 32)
    for (uint32_t col = threadRank; col < zeroTerm; col += BLOCK_SIZE) {
        s_alreadyVisited[col] = 0;
    }

    __syncthreads();

    auto exCells = allCells[b];
    auto exAdjCounts = reinterpret_cast<uint32_t*>(outAdjCounts[b].data());
    auto exAdjValues = outSparseAdj[b][quadIdx].data();

    T *exData = allEmbedQuads[b].data();

    const auto bdsAnchor = Quad_<T>{ s_quadVerts }.Bounds();

    const uint32_t startCol = s_quadExtent[0],
                   endCol = s_quadExtent[2];
    for (uint32_t row = s_quadExtent[1], endRow = s_quadExtent[3]; row <= endRow; ++row) {
        auto rowCells = exCells[row];

        for (uint32_t col = startCol; col <= endCol; ++col) {
            auto colCells = reinterpret_cast<const uint32_t*>(rowCells[col].data());

            const uint32_t ct = colCells[0];

            for (uint32_t i = threadRank + 1; i <= ct; i += BLOCK_SIZE) {
                const uint32_t otherIdx = colCells[i];

                const uint32_t maskIdx = otherIdx >> 5; // Divide by 32, since there are 32 bits per mask slot
                const uint32_t maskBit = 1 << (otherIdx & 0x1F); // Set the relevant bit for this mask ID

                const bool alreadyVisited = atomicOr(s_alreadyVisited + maskIdx, maskBit) & maskBit;

                if (!alreadyVisited) {
                    const auto bdsOther = StridedEmbedQuad_<T>{ exData + otherIdx * allEmbedQuads.stride(2), allEmbedQuads.stride(1) }.Bounds();

                    T pctAnchor, pctOther, iou;
                    thrust::tie(pctAnchor, pctOther, iou) = geometry_region_sizes(bdsAnchor, bdsOther);

                    if (iou >= iouThreshold) {
                        auto validGroup = cg::coalesced_threads();

                        uint32_t storeOff = 0;
                        if (validGroup.thread_rank() == 0) {
                            storeOff = atomicAdd(exAdjCounts + quadIdx, validGroup.size());
                        }
                        storeOff = validGroup.shfl(storeOff, 0) + validGroup.thread_rank();

                        exAdjValues[storeOff] = otherIdx;

                        if (otherIdx > quadIdx) {
                            outIsStart[b][otherIdx] = false;
                        }
                    } else if (pctAnchor > 0.8f || pctOther > 0.8f) {
                        T anchorHeight = bdsAnchor.Height();
                        T otherHeight = bdsOther.Height();

                        T ratio = anchorHeight > otherHeight ?
                                        otherHeight / anchorHeight :
                                        anchorHeight / otherHeight;
                        if (ratio > 0.9f) {
                            if (pctAnchor > 0.8f) {
                                // Other envelops anchor
                                outIsStart[b][quadIdx] = false;
                            } else {
                                outIsStart[b][otherIdx] = false;
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__
void device_flatten_graph_iterative(const int32_t *ptrQuadCts,
                                    torch::PackedTensorAccessor64<bool, 2> allIsStart,
                                    volatile uint32_t *allAdjCounts,
                                    volatile uint32_t *allAdjValues
#ifdef NMS_VERIFY_CORRECTNESS
                                  , int32_t *maxDepth
#endif
                                    )
{
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t VISIT_STACK_SIZE = 9;
    constexpr uint32_t TERM_VALUE = std::numeric_limits<uint32_t>::max();

    constexpr visit_mask_t VISITED_MASK = 0b001;
    constexpr visit_mask_t ADDED_MASK = 0b010;
    constexpr visit_mask_t QUEUED_MASK = 0b100;
    constexpr visit_mask_t QUEUED_OR_VISITED_MASK = VISITED_MASK | QUEUED_MASK;

    const uint32_t b = blockIdx.z;
    const uint32_t anchorRow = blockIdx.y;

    const uint32_t quadCt = ptrQuadCts[b];

    // Only need to check this if there are multiple examples, since in the case of a single example,
    // the grid is precisely sized to that quadCt
    if (anchorRow >= quadCt) {
        return;
    }

    auto isStart = allIsStart[b].data();

    const uint32_t threadRank = threadIdx.x;

    extern __shared__ visit_mask_t s_visitedMask[];

#ifndef NMS_VERIFY_CORRECTNESS
    // Only need to process the anchor rows, since they're the only ones
    // that will make it through the full NMS operation.
    // NOTE: There's a race condition where some rows may be marked as anchor,
    // but they'll later be marked non-anchor over the course of this kernel.
    // That's fine. It's a bit of extra work, but there's no real way around it.
    const bool anchorIsStart = isStart[anchorRow];
    if (!anchorIsStart) {
        return;
    }
#endif

    uint32_t *pIntVisitedMask = reinterpret_cast<uint32_t*>(s_visitedMask);
    uint32_t zeroTerm = (quadCt + 3) >> 2; // Fast version of div_up(quadCt, 4)
    for (uint32_t col = threadRank; col < zeroTerm; col += blockDim.x) {
        pIntVisitedMask[col] = 0;
    }

    __syncthreads();

    const uint32_t maxExCount = allIsStart.size(1);
    auto adjCounts = allAdjCounts + (b * maxExCount);
    auto adjValues = allAdjValues + (b * maxExCount * maxExCount);

    auto adjAnchorValues = adjValues + (anchorRow * maxExCount);
    // For the anchor row, set the visited mask to 0b10, which will signify that we haven't visited it yet,
    // but that the value is already in the adjacency vector.
    // 0bx1 signifies that the value has been visited
    for (uint32_t i = threadRank, ct = adjCounts[anchorRow]; i < ct; i += blockDim.x) {
        const auto adjCol = adjAnchorValues[i];
        s_visitedMask[adjCol] = ADDED_MASK;
    }

    __syncthreads();

    if (threadRank == 0) {
        s_visitedMask[anchorRow] |= QUEUED_MASK;
    }

    __syncthreads();

    // TODO(mranzinger): Is it worth incorporating these other threads?
    // It seems like the vast majority of adjacency counts is <32
    if (threadRank >= WARP_SIZE) {
        return;
    }

    uint32_t visitStack[VISIT_STACK_SIZE];
    visitStack[0] = TERM_VALUE;
    visitStack[1] = anchorRow;
#ifndef NDEBUG
    for (uint32_t i = 2; i < VISIT_STACK_SIZE; ++i) {
        visitStack[i] = TERM_VALUE;
    }
#endif
    int32_t visitPtr = 1;

    // NOTE: This loop is actually terminated by the `if (warpNextCol == TERM_VALUE)` check below
    for (uint32_t dfsIter = 0; true; ++dfsIter) {
#ifdef NMS_VERIFY_CORRECTNESS
        assert(visitPtr >= 0 && visitPtr < VISIT_STACK_SIZE);
#endif
        const uint32_t threadNextCol = visitStack[visitPtr];
        const uint32_t warpNextCol = __reduce_min_full_warp(threadNextCol);

        // Check to see if this thread got chosen.
        // If so, decrement the stack counter
        if (threadNextCol == warpNextCol) {
#ifndef NDEBUG
            // This makes it easier to debug where the pointer is
            visitStack[visitPtr] = TERM_VALUE;
#endif
            --visitPtr;
        }

        // If the maximum value encountered is -1, that means that none of the threads
        // had another value to process
        if (warpNextCol == TERM_VALUE) {
            break;
        }

        const uint32_t procRow = warpNextCol;

        __syncthreads();

        bool isAlreadyVisited = s_visitedMask[procRow] & VISITED_MASK;

        if (isAlreadyVisited) {
            continue;
        }

        const uint32_t procAdjCount = adjCounts[procRow];
        auto procAdjValues = adjValues + (procRow * maxExCount);

        for (uint32_t i = threadRank; i < procAdjCount; i += WARP_SIZE) {
            uint32_t adjCol = procAdjValues[i];

            auto group = cg::coalesced_threads();
            // Offsetting by the iteration number will help balance out the maximum depth of any stack in the warp.
            // The reason behind this is due to how otherwise, warp-0 will always get a new element, warp-1 iff the adj graph
            // has more than one element, warp-2 iff the adj graph has more than two elements, and so on. Basically,
            // the warps have decreasing pressure. With the rotation mechanism, it helps to balance out stack usage.
            adjCol = group.shfl(adjCol, (group.thread_rank() + dfsIter) % group.size());

            // This will set the queued flag for this column, if it's not already set.
            // It also returns the old state. In our case, we only want to add this value to the
            // stack iff it hasn't already been visited, and hasn't been queued elsewhere
            // NOTE: CUDA doesn't support atomicOr on uint8_t :(, but it's not necessary that
            // the operation be absolutely atomic, so the poor man's version is probably okay
            const auto oldMask = s_visitedMask[adjCol];
            auto newMask = oldMask;

            bool alreadyAdded = oldMask & ADDED_MASK;

            const uint32_t gThreadRank = group.thread_rank();
            uint32_t notAddedBallot = group.ballot(!alreadyAdded);
            if (notAddedBallot) {
                // Only one warp will ever be adding values to a given row, which means
                // that we don't need atomics. However, other warps may be reading data
                // from anchorRow, which means that we need to add the values first,
                // followed by incrementing the count. This order makes things
                // concurrency safe.
                const uint32_t globalStoreOff = adjCounts[anchorRow];
                // Gets the count of the bits to the left of this thread
                const uint32_t localStoreOff = __popc(notAddedBallot & ((1 << gThreadRank) - 1));

                if (!alreadyAdded) {
                    adjAnchorValues[globalStoreOff + localStoreOff] = adjCol;
                    if (adjCol > anchorRow) {
                        // Also, ensure that this quad is no longer marked as a starting quad
                        isStart[adjCol] = false;
                    }
                    newMask |= ADDED_MASK;
                }

                // Finally, commit the change by incrementing the counter
                if (gThreadRank == 0) {
                    adjCounts[anchorRow] += __popc(notAddedBallot);
                }
            }

            bool alreadyHandled = oldMask & QUEUED_OR_VISITED_MASK;

            if (!alreadyHandled) {
#ifdef NMS_VERIFY_CORRECTNESS
                newMask |= QUEUED_MASK;
                ++visitPtr;
                assert(visitPtr < VISIT_STACK_SIZE);
                atomicMax(maxDepth, visitPtr);
                visitStack[visitPtr] = adjCol;
#else
                // Prefer potentially inconsistent results over buffer overflow
                if (visitPtr < VISIT_STACK_SIZE - 1) {
                    newMask |= QUEUED_MASK;
                    ++visitPtr;
                    visitStack[visitPtr] = adjCol;
                }
#endif
            }

            if (newMask != oldMask) {
                s_visitedMask[adjCol] = newMask;
            }
        }

        // We actually rely on the `pop_next` function largely to handle recursing down into the next row
        __syncthreads();
    }
}

void add_to_set(const torch::TensorAccessor<int32_t, 1>& adjCounts,
    const torch::TensorAccessor<int32_t, 2>& adjValues,
    int32_t row,
    std::unordered_set<int32_t>& possible)
{
    if (possible.count(row)) {
        return;
    }

    possible.insert(row);

    const int32_t adjCount = adjCounts[row];
    auto values = adjValues[row].data();

    for (int32_t i = 0; i < adjCount; ++i) {
        const int32_t col = values[i];
        add_to_set(adjCounts, adjValues, col, possible);
    }
}

void cpu_flatten_graph(const int32_t *ptrQuadCts,
                       torch::Tensor isStartTensorGPU,
                       torch::Tensor adjCountsTensorGPU,
                       torch::Tensor adjValuesTensorGPU)
{
    auto isStartTensor = isStartTensorGPU.cpu();
    auto adjCountsTensor = adjCountsTensorGPU.cpu();
    auto adjValuesTensor = adjValuesTensorGPU.cpu();

    auto allIsStart = isStartTensor.accessor<bool, 2>();
    auto allAdjCounts = adjCountsTensor.accessor<int32_t, 2>();
    auto allAdjValues = adjValuesTensor.accessor<int32_t, 3>();

    for (int32_t b = 0; b < allAdjCounts.size(0); ++b) {
        const int32_t quadCt = ptrQuadCts[b];

        for (int32_t row = 0; row < quadCt; ++row) {
            std::unordered_set<int32_t> fullAdjSet;
            add_to_set(allAdjCounts[b], allAdjValues[b], row, fullAdjSet);

            int32_t &currCt = allAdjCounts[b][row];
            int32_t *currValues = allAdjValues[b][row].data();
            std::unordered_set<int32_t> existingSet{ currValues, currValues + currCt };

            for (int32_t adjCol : fullAdjSet) {
                if (existingSet.count(adjCol)) {
                    continue;
                }

                currValues[currCt] = adjCol;
                ++currCt;

                if (adjCol > row) {
                    allIsStart[b][adjCol] = false;
                }
            }
        }
    }

    isStartTensorGPU.copy_(isStartTensor);
    adjCountsTensorGPU.copy_(adjCountsTensor);
    adjValuesTensorGPU.copy_(adjValuesTensor);
}


__global__
void device_a2a_adj_cleanup(const int32_t *counts,
                            torch::PackedTensorAccessor64<uint8_t, 3> inOutAdjacency)
{
    const uint32_t b = blockIdx.y;
    const uint32_t jobIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t numQuads = counts[b];
    const uint32_t row = jobIdx / numQuads;
    const uint32_t col = jobIdx % numQuads;

    if (row >= numQuads) {
        return;
    }

    auto adjacency = inOutAdjacency[b];

    bool rowPivot = adjacency[row][row] > 0;
    bool colPivot = adjacency[col][col] > 0;

    if (!rowPivot || !colPivot) {
        adjacency[row][col] = 0;
    }
}

template<uint32_t NumWarps, typename T>
__global__
void device_a2a_collapse(torch::PackedTensorAccessor64<int32_t, 1> quadCounts,
                         torch::PackedTensorAccessor64<T, 3> allEmbedQuads,
                         torch::PackedTensorAccessor64<bool, 2> allIsLeadRow,
                         const int64_t *regionCounts,
                         torch::PackedTensorAccessor64<int32_t, 2> allAdjCounts,
                         torch::PackedTensorAccessor64<int32_t, 3> allAdjValues,
                         //torch::PackedTensorAccessor64<int32_t, 2> allOutPositions,
                         torch::PackedTensorAccessor64<T, 3> outQuads,
                         T *outConf)
{
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t FULL_WARP = 0xFFFFFFFF;
    constexpr uint32_t BLOCK_WIDTH = NumWarps * WARP_SIZE;
    constexpr size_t MERGE_QUAD_SIZE = sizeof(MergeQuad_<T>) / sizeof(T);

    static_assert(NumWarps < WARP_SIZE, "Only a single warp currently supported!");

    const uint32_t b = blockIdx.z;
    const uint32_t row = blockIdx.y;

    const int32_t quadCt = quadCounts[b];

    if (row >= quadCt) {
        return;
    }

    // Only process the lead rows
    const auto isLeadRow = allIsLeadRow[b].data();
    if (!isLeadRow[row]) {
        return;
    }

    const uint32_t threadRank = threadIdx.x;
    const uint32_t localThreadRank = threadRank & 0x1F;
    const uint32_t warpIdx = threadRank >> 5;

    __shared__ T s_mergeQuad[MERGE_QUAD_SIZE];

    if constexpr (NumWarps > 1) {
        if (threadRank < MERGE_QUAD_SIZE) {
            s_mergeQuad[threadRank] = 0.0f;
        }

        __syncthreads();
    }

    T *exData = allEmbedQuads[b].data();

    const int32_t adjCount = allAdjCounts[b][row];
    const int32_t *adjIdxs = allAdjValues[b][row].data();

    MergeQuad_<T> localMerge{ZeroInitTag{}};

    for (int32_t i = threadRank; i < adjCount; i += BLOCK_WIDTH) {
        const int32_t currQuadIdx = adjIdxs[i];
        const StridedEmbedQuad_<T> qCurr{ exData + currQuadIdx * allEmbedQuads.stride(2), allEmbedQuads.stride(1) };

        localMerge.Append(qCurr);
    }

    T *mqV = reinterpret_cast<T*>(&localMerge);
    #pragma unroll
    for (uint32_t offset = 1; offset < WARP_SIZE; offset <<= 1) {
        T mergeFactor = offset + localThreadRank < 32;
        #pragma unroll
        for (uint32_t i = 0; i < MERGE_QUAD_SIZE; ++i) {
            mqV[i] += mergeFactor * __shfl_down_sync(FULL_WARP, mqV[i], offset);
        }
    }
    #pragma unroll
    for (uint32_t i = 0; i < MERGE_QUAD_SIZE; ++i) {
        mqV[i] = __shfl_sync(FULL_WARP, mqV[i], 0);
    }

    // Only need to do a multi-warp merge if there are enough quads to justify it
    if (NumWarps > 1 && adjCount > WARP_SIZE) {
        if (localThreadRank < MERGE_QUAD_SIZE) {
            atomicAdd(s_mergeQuad + localThreadRank, mqV[localThreadRank]);
        }

        __syncthreads();

        mqV = s_mergeQuad;
    }

    // Figure out the output position
    uint32_t writePosition = 0;
    for (int32_t i = threadRank; i < b; i += BLOCK_WIDTH) {
        writePosition += regionCounts[i];
    }

    const uint8_t *pCurrIsLeadRow = reinterpret_cast<const uint8_t*>(isLeadRow);
    for (int32_t i = threadRank; i < row; i += BLOCK_WIDTH) {
        if (pCurrIsLeadRow[i]) {
            ++writePosition;
        }
    }
    // Sum all of the individual offsets over the warp
    writePosition = __reduce_add_full_warp(writePosition);
    // Reduce across warps, if applicable
    if constexpr (NumWarps > 1) {
        __shared__ uint32_t s_threadWritePositions[NumWarps];
        if (localThreadRank == 0) {
            s_threadWritePositions[warpIdx] = writePosition;
        }
        __syncthreads();
        writePosition = threadRank < NumWarps ? s_threadWritePositions[threadRank] : 0;
        writePosition = __reduce_add_full_warp(writePosition);
    }

    if (threadRank >= 9) {
        return;
    }

    const T sumConfidence = mqV[8];
    const T numQuads = mqV[9];
    const T divisor = threadRank < 8 ? sumConfidence : numQuads;

    const T myVal = mqV[threadRank] / divisor;

    auto writeVerts = outQuads[writePosition].data();

    if (threadRank < 8) {
        writeVerts[threadRank] = myVal;
    } else {
        outConf[writePosition] = myVal;
    }
}

struct CollapseRowsResult {
    torch::Tensor ExCounts;
    torch::Tensor StridedMergeQuads;
    int32_t TotalNumQuads;
    // NOTE: This will only be available in Debug builds
    torch::Tensor QuadIds;
    int32_t ImageWidth;
    int32_t ImageHeight;
};

template<typename scalar_t>
CollapseRowsResult collapse_rows(
    torch::Tensor quads, torch::Tensor probs, scalar_t probThreshold, scalar_t iouThreshold
)
{
    if (! quads.is_contiguous()) {
        throw std::runtime_error("Expected `quads` to be contiguous!");
    }

    if ((quads.size(2) % 32) != 0) {
        throw std::runtime_error("Expected the width of the `quads` buffer to be a multiple of 32!");
    }

    int32_t imageWidth = quads.size(2) * 4;
    int32_t imageHeight = quads.size(1) * 4;

    quads = quads.reshape({ quads.size(0), -1, 32, 4, 2 });
    probs = probs.reshape({ probs.size(0), -1, 32 });

    if (quads.size(0) != probs.size(0) || quads.size(1) != probs.size(1)) {
        throw std::runtime_error("Dimension mismatch between `quads` and `probs`");
    }

    // The final counter is for the total number of quads for the entire batch
    auto counts = torch::zeros({ quads.size(0) + 1 }, quads.options().dtype(torch::kInt32));

    int64_t embedSize = sizeof(EmbedQuad_<scalar_t>) / sizeof(scalar_t);
    auto rowMergeTensor = torch::empty({ quads.size(0), embedSize, quads.size(1) * quads.size(2) }, quads.options());

    auto idsTensor = torch::full({ quads.size(0), quads.size(1) * quads.size(2) },
                                 std::numeric_limits<int32_t>::max(),
                                 counts.options().dtype(torch::kInt32));

    dim3 blockSize(32, 3, 1);
    dim3 gridSize(1,
                  div_up(quads.size(1), blockSize.y),
                  quads.size(0));

    device_row_collapse KERNEL_ARG2(gridSize, blockSize) (
        quads.packed_accessor64<scalar_t, 5>(),
        probs.packed_accessor64<scalar_t, 3>(),
        probThreshold, iouThreshold,
        counts.packed_accessor64<int32_t, 1>(),
        rowMergeTensor.packed_accessor64<scalar_t, 3>(),
        idsTensor.packed_accessor64<int32_t, 2>()
    );

#ifdef NMS_VERIFY_CORRECTNESS
    static std::unordered_set<int32_t> s_quadIds;
    auto cpuIdsTensor = idsTensor.cpu();
    const int32_t *idsPtr = cpuIdsTensor.data_ptr<int32_t>();
    if (s_quadIds.empty()) {
        s_quadIds.insert(idsPtr, idsPtr + idsTensor.numel());
    } else {
        std::unordered_set<int32_t> otherIds{ idsPtr, idsPtr + idsTensor.numel() };

        if (s_quadIds != otherIds) {
            throw std::runtime_error("Inconsistent Ids!");
        }
    }
#endif

    // The final value in `counts` is actually to total number of quads for the entire batch
    int32_t totalQuads = counts[-1].item<int32_t>();

    counts = counts.slice(/*dim=*/ 0, 0, counts.size(0) - 1);

    int64_t maxExCount;
    if (counts.size(0) > 1) {
        maxExCount = counts.max().item<int32_t>();
    } else {
        maxExCount = totalQuads;
    }

    static bool s_sortOrder = false;

    rowMergeTensor = rowMergeTensor.slice(2, 0, maxExCount);
    idsTensor = idsTensor.slice(1, 0, maxExCount);
    auto order = torch::argsort(idsTensor, /*dim=*/ 1, s_sortOrder);

    auto embOrder = order.unsqueeze(1).expand_as(rowMergeTensor);

    rowMergeTensor = torch::gather(rowMergeTensor, /*dim=*/ 2, embOrder);
    idsTensor = torch::gather(idsTensor, /*dim=*/ 1, order);

    return { counts, rowMergeTensor, totalQuads, idsTensor, imageWidth, imageHeight };
}



void verify_row(const torch::TensorAccessor<int32_t, 1> &adjCounts,
                const torch::TensorAccessor<int32_t, 2> &adjValues,
                int32_t row)
{
    // Traverse the graph, and accumulate all set flags across all rows marked
    // adjacent by the current row. If the merge_up algorithm works correctly, then
    // `possible` will contain exactly the same set of values as the current row
    std::unordered_set<int32_t> possible;
    add_to_set(adjCounts, adjValues, row, possible);

    std::unordered_set<int32_t> thisRow{ row };
    const int32_t thisCount = adjCounts[row];
    auto thisValues = adjValues[row].data();
    thisRow.insert(thisValues, thisValues + thisCount);

    if (thisRow != possible) {
        throw std::runtime_error("The merge_up algorithm is not correct!");
    }
}

struct AdjacencyResult {
    // Shape: BxQ
    // Specifies whether the given row is a result row
    torch::Tensor IsLeadRow;
    // Shape: BxQ
    // The number of quads that need to be merged with the given quad
    torch::Tensor AdjCounts;
    // Shape: BxQx<Num Adjacent>
    // The indices of the adjacent quads.
    torch::Tensor AdjValues;
    int64_t MaxExCount;
};

template<typename T>
void cpu_a2a_adjacency_sparse(const int32_t *ptrQuadCts,
                              const T iouThreshold,
                              torch::Tensor embedQuadsTensor,
                              torch::Tensor outIsStartTensorGPU,
                              torch::Tensor outAdjCountsTensorGPU,
                              torch::Tensor outSparseAdjTensorGPU)
{
    embedQuadsTensor = embedQuadsTensor.cpu();
    auto outIsStartTensor = outIsStartTensorGPU.cpu();
    auto outAdjCountsTensor = outAdjCountsTensorGPU.cpu();
    auto outSparseAdjTensor = outSparseAdjTensorGPU.cpu();

    auto embedQuads = embedQuadsTensor.accessor<T, 3>();
    auto isStart = outIsStartTensor.accessor<bool, 2>();
    auto adjCounts = outAdjCountsTensor.accessor<int32_t, 2>();
    auto adjValues = outSparseAdjTensor.accessor<int32_t, 3>();

    for (int32_t b = 0; b < embedQuadsTensor.size(0); ++b) {
        const int32_t quadCt = ptrQuadCts[b];

        T *exData = embedQuads[b].data();

        for (int32_t row = 0; row < quadCt; ++row) {
            const auto qRow = StridedEmbedQuad_<T>{ exData + row, embedQuads.stride(1) }.Bounds();

            for (int32_t col = 0; col < quadCt; ++col) {
                const auto qCol = StridedEmbedQuad_<T>{ exData + col, embedQuads.stride(1) }.Bounds();

                T pctRow, pctCol, iou;
                thrust::tie(pctRow, pctCol, iou) = geometry_region_sizes(qRow, qCol);

                if (iou >= iouThreshold) {
                    int32_t &storeIdx = adjCounts[b][row];
                    adjValues[b][row][storeIdx] = col;
                    ++storeIdx;
                    if (row < col) {
                        isStart[b][col] = false;
                    }
                } else if (pctRow > 0.8f || pctCol > 0.8f) {
                    T anchorHeight = qRow.Height();
                    T otherHeight = qCol.Height();

                    T ratio = anchorHeight > otherHeight ?
                        otherHeight / anchorHeight :
                        anchorHeight / otherHeight;
                    if (ratio > 0.9f) {
                        if (pctRow > 0.8f) {
                            // Other envelops anchor
                            isStart[b][row] = false;
                        }
                        else {
                            isStart[b][col] = false;
                        }
                    }
                }
            }
        }
    }

    outIsStartTensorGPU.copy_(outIsStartTensor);
    outAdjCountsTensorGPU.copy_(outAdjCountsTensor);
    outSparseAdjTensorGPU.copy_(outSparseAdjTensor);
}

template<typename T>
std::string to_flat_string(torch::Tensor tensor) {
    tensor = tensor.flatten();

    auto acc = tensor.accessor<T, 1>();

    std::ostringstream oss;
    oss << "[";
    if (acc.size(0) > 0) {
        oss << acc[0];
        for (int64_t i = 1; i < acc.size(0); ++i) {
            oss << ", " << acc[i];
        }
    }
    oss << "]";
    return oss.str();
}

template<typename scalar_t>
AdjacencyResult compute_all_to_all_adjacency(
    const CollapseRowsResult &collapseResult,
    scalar_t iouThreshold)
{
    torch::Tensor counts = collapseResult.ExCounts;

    int64_t maxExCount;
    if (counts.size(0) > 1) {
        maxExCount = counts.max().item<int32_t>();
    } else {
        maxExCount = collapseResult.TotalNumQuads;
    }

    auto isStartTensor = torch::ones({ counts.size(0), maxExCount }, counts.options().dtype(torch::kBool));
    auto adjCountsTensor = torch::zeros({ counts.size(0), maxExCount }, counts.options().dtype(torch::kInt32));
#ifndef NMS_VERIFY_CORRECTNESS
    auto adjValuesTensor = torch::empty({ counts.size(0), maxExCount, maxExCount }, counts.options().dtype(torch::kInt32));
#else
    auto adjValuesTensor = torch::full({ counts.size(0), maxExCount, maxExCount },
                                       5000,
                                       counts.options().dtype(torch::kInt32));
#endif

#ifdef NMS_VERIFY_CORRECTNESS
    auto cpuAdjValuesTensor = adjValuesTensor.cpu();
    auto cpuAdjCountsTensor = adjCountsTensor.cpu();
    auto cpuIsStartTensor = isStartTensor.cpu();
#endif

    size_t smemSize;
    dim3 gridSize, blockSize;

    ///////////////////
    // NOTE(mranzinger): This algorithm uses a fixed sized grid to spatially subdivide the canvas. For virtually all test conditions
    //                   I ran this through, it was slightly slower than the brute force approach that parallelizes better.
    //                   It's possible that there is some number of words present (e.g. >500) where this algorithm becomes
    //                   faster.
    //
    //constexpr int32_t CELL_SIZE = 100;
    //constexpr int64_t NUM_BINS_PER_CELL = 200;
    //int32_t numXCells = div_up(collapseResult.ImageWidth, CELL_SIZE);
    //int32_t numYCells = div_up(collapseResult.ImageHeight, CELL_SIZE);
    //auto gridCellsTensor = torch::zeros({ counts.size(0), numYCells, numXCells, NUM_BINS_PER_CELL }, adjCountsTensor.options());
    //auto quadCellExtentsTensor = torch::empty({ counts.size(0), maxExCount, 4 }, gridCellsTensor.options());
    //smemSize = div_up(static_cast<uint32_t>(maxExCount), 32);

    //constexpr uint32_t GRID_NUM_WARPS = 3;
    //blockSize = dim3{ GRID_NUM_WARPS * 32, 1, 1 };
    //gridSize = dim3{ 1, static_cast<uint32_t>(maxExCount), static_cast<uint32_t>(counts.size(0)) };

    //device_a2a_adjacency_build_grid<GRID_NUM_WARPS, scalar_t, CELL_SIZE> KERNEL_ARG2(gridSize, blockSize) (
    //    counts.data_ptr<int32_t>(),
    //    collapseResult.StridedMergeQuads.packed_accessor64<scalar_t, 3>(),
    //    gridCellsTensor.packed_accessor64<int32_t, 4>(),
    //    quadCellExtentsTensor.packed_accessor64<int32_t, 3>()
    //);

    //device_a2a_adjacency_with_grid<GRID_NUM_WARPS, scalar_t> KERNEL_ARG3(gridSize, blockSize, smemSize) (
    //    counts.data_ptr<int32_t>(),
    //    iouThreshold,
    //    collapseResult.StridedMergeQuads.packed_accessor64<scalar_t, 3>(),
    //    gridCellsTensor.packed_accessor64<int32_t, 4>(),
    //    quadCellExtentsTensor.packed_accessor64<int32_t, 3>(),
    //    isStartTensor.packed_accessor64<bool, 2>(),
    //    adjCountsTensor.packed_accessor64<int32_t, 2>(),
    //    adjValuesTensor.packed_accessor64<int32_t, 3>()
    //);
    ///////////////////

    uint32_t totalWork = maxExCount * maxExCount;

    blockSize = dim3{96, 1};
    gridSize = dim3{div_up(totalWork, blockSize.x),
                    static_cast<uint32_t>(counts.size(0))};

    // This algorithm is O(n^2) with n being the current number of quads
    device_a2a_adjacency_sparse<scalar_t> KERNEL_ARG2(gridSize, blockSize) (
        counts.data_ptr<int32_t>(),
        iouThreshold,
        collapseResult.StridedMergeQuads.packed_accessor64<scalar_t, 3>(),
        isStartTensor.packed_accessor64<bool, 2>(),
        adjCountsTensor.packed_accessor64<int32_t, 2>(),
        adjValuesTensor.packed_accessor64<int32_t, 3>()
    );


#ifdef NMS_VERIFY_CORRECTNESS
    auto cpuCounts = counts.cpu();

    cpu_a2a_adjacency_sparse<scalar_t>(cpuCounts.data_ptr<int32_t>(), iouThreshold,
        collapseResult.StridedMergeQuads, cpuIsStartTensor, cpuAdjCountsTensor, cpuAdjValuesTensor);

    adjValuesTensor = std::get<0>(torch::sort(adjValuesTensor, /*dim=*/ 2));

    assert(torch::all(cpuAdjCountsTensor == adjCountsTensor.cpu()).item<bool>());
    assert(torch::all(cpuIsStartTensor == isStartTensor.cpu()).item<bool>());
    assert(torch::all(cpuAdjValuesTensor == adjValuesTensor.cpu()).item<bool>());

    std::cout << "\tA2A Is Start Count: " << isStartTensor.sum(torch::kInt32).item<int32_t>()
              << ", Most Adjacent: " << adjCountsTensor.max().item<int32_t>() << std::endl;

    auto maxDepthTensor = torch::tensor(0, adjCountsTensor.options());
#endif

    blockSize = dim3{ 128, 1, 1 };
    gridSize = dim3{ 1, static_cast<uint32_t>(maxExCount), static_cast<uint32_t>(counts.size(0)) };
    smemSize = div_up(maxExCount * sizeof(visit_mask_t), sizeof(uint32_t)) * sizeof(uint32_t);

    device_flatten_graph_iterative KERNEL_ARG3(gridSize, blockSize, smemSize) (
        counts.data_ptr<int32_t>(),
        isStartTensor.packed_accessor64<bool, 2>(),
        reinterpret_cast<uint32_t*>(adjCountsTensor.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(adjValuesTensor.data_ptr<int32_t>())
#ifdef NMS_VERIFY_CORRECTNESS
      , maxDepthTensor.data_ptr<int32_t>()
#endif
    );

#ifdef NMS_VERIFY_CORRECTNESS
    cpu_flatten_graph(cpuCounts.data_ptr<int32_t>(), cpuIsStartTensor, cpuAdjCountsTensor, cpuAdjValuesTensor);

    cpuAdjValuesTensor = std::get<0>(torch::sort(cpuAdjValuesTensor, /*dim=*/ 2));
    adjValuesTensor = std::get<0>(torch::sort(adjValuesTensor, /*dim=*/ 2));

    torch::Tensor diffStartIdxs = (cpuIsStartTensor != isStartTensor.cpu()).nonzero_numpy()[0];

    assert(diffStartIdxs.numel() == 0);

    torch::Tensor diffCountIdxs = (cpuAdjCountsTensor != adjCountsTensor.cpu()).nonzero_numpy()[0];

    assert(diffCountIdxs.numel() == 0);

    auto diffValuesTensor = torch::any(cpuAdjValuesTensor != adjValuesTensor.cpu(), /*dim=*/ 2, /*keepdim=*/ false).flatten().nonzero().flatten();

    std::cout << "\t\tDiff Indices: " << to_flat_string<int64_t>(diffValuesTensor) << std::endl;

    auto cpuDiffCountsTensor = cpuAdjCountsTensor.flatten().index({ diffValuesTensor });
    auto cpuDiffRowsTensor = cpuAdjValuesTensor.flatten(0, 1).index({ diffValuesTensor });
    auto gpuDiffRowsTensor = adjValuesTensor.cpu().flatten(0, 1).index({ diffValuesTensor });

    for (int64_t i = 0, ct = cpuDiffRowsTensor.size(0); i < ct; ++i) {
        auto z = cpuDiffCountsTensor[i].item<int32_t>();
        auto diffRow = diffValuesTensor[i].item<int64_t>();
        std::cout << "\t\tRow " << diffRow << std::endl;
        std::cout << "\t\t\tExpected: " << to_flat_string<int32_t>(cpuDiffRowsTensor[i].slice(0, 0, z + 1)) << std::endl;
        std::cout << "\t\t\t     GPU: " << to_flat_string<int32_t>(gpuDiffRowsTensor[i].slice(0, 0, z + 1)) << std::endl;
    }

    assert(diffValuesTensor.size(0) == 0);

    std::cout << "\tA2A - Flatten - Is Start Count: " << isStartTensor.sum(torch::kInt32).item<int32_t>()
         << ", Most Adjacent: " << adjCountsTensor.max().item<int32_t>()
         << ", Max Depth: " << maxDepthTensor.item<int32_t>() << std::endl;

    cpuIsStartTensor = isStartTensor.cpu();
    cpuAdjCountsTensor = adjCountsTensor.cpu();
    cpuAdjValuesTensor = adjValuesTensor.cpu();
    auto cpuCollapseIds = collapseResult.QuadIds.cpu();

    static std::vector<std::unordered_set<int32_t>> s_knownGroups;
    static std::unordered_map<int32_t, std::unordered_set<int32_t>> s_groupLookup;

    std::vector<std::unordered_set<int32_t>> idGroups;
    decltype(s_groupLookup) groupLookup;
    for (int64_t b = 0; b < counts.size(0); ++b) {
        int64_t quadCt = cpuCounts[b].item<int32_t>();
        for (int64_t row = 0; row < quadCt; ++row) {
            bool isLeadRow = cpuIsStartTensor[b][row].item<bool>();
            auto bCountsTensor = cpuAdjCountsTensor[b];
            auto bValuesTensor = cpuAdjValuesTensor[b];
            auto bCounts = bCountsTensor.accessor<int32_t, 1>();
            auto bValues = bValuesTensor.accessor<int32_t, 2>();

            auto bIdsTensor = cpuCollapseIds[b];
            auto bIds = bIdsTensor.accessor<int32_t, 1>();

            std::unordered_set<int32_t> sIds;
            for (int32_t i = 0, ct = bCounts[row]; i < ct; ++i) {
                int32_t col = bValues[row][i];
                int32_t id = bIds[col];
                sIds.insert(id);
            }

            if (sIds.empty()) {
                throw std::runtime_error("The ids tensor is empty!");
            }

            groupLookup[bIds[row]] = sIds;

            if (isLeadRow) {
                verify_row(bCounts, bValues, row);
                idGroups.push_back(move(sIds));
            }
        }
    }

    if (s_knownGroups.empty()) {
        s_knownGroups = move(idGroups);
        s_groupLookup = move(groupLookup);
    } else {
        // Make a copy
        auto remOrigGroups = s_knownGroups;
        auto remOrigGroupLookup = s_groupLookup;

        std::vector<int32_t> quadIds;
        for (auto &kv : remOrigGroupLookup) {
            quadIds.push_back(kv.first);
        }
        for (int32_t qId : quadIds) {
            assert(groupLookup.count(qId));
        }
        assert(groupLookup.size() == remOrigGroupLookup.size());

        for (int32_t qId : quadIds) {
            auto &oldGroup = remOrigGroupLookup[qId];
            auto &newGroup = groupLookup[qId];

            if (oldGroup == newGroup) {
                remOrigGroupLookup.erase(qId);
                groupLookup.erase(qId);
            } else {
                throw std::runtime_error("Group mismatch!");
            }
        }

        for (int i = idGroups.size() - 1; i >= 0; --i) {
            for (int j = remOrigGroups.size() - 1; j >= 0; --j) {
                auto &idGroup = idGroups[i];
                auto &knownGroup = remOrigGroups[j];

                if (idGroup == knownGroup) {
                    idGroups.erase(begin(idGroups) + i);
                    remOrigGroups.erase(begin(remOrigGroups) + j);
                    break;
                }
            }
        }

        if (!idGroups.empty() || !remOrigGroups.empty()) {
            auto group_str = [] (auto &group) {
                std::vector<int32_t> vGroup{ std::begin(group), std::end(group) };
                std::sort(std::begin(vGroup), std::end(vGroup));

                auto id_str = [] (int32_t id) {
                    std::ostringstream oss;
                    //oss << "(" << (id / 32) << ", " << (id % 32) << ")";
                    oss << id;
                    return oss.str();
                };

                std::ostringstream oss;
                oss << "[" << id_str(vGroup[0]);
                for (size_t i = 1; i < vGroup.size(); ++i) {
                    oss << ", " << id_str(vGroup[i]);
                }
                oss << "]";
                return oss.str();
            };

            std::cout << "\tEncountered a difference in groups!" << std::endl
                 << "\t\tOrig groups:" << std::endl;
            for (auto &group : remOrigGroups) {
                std::cout << "\t\t\t" << group_str(group) << std::endl;
            }
            std::cout << "\t\tNew groups:" << std::endl;
            for (auto &group : idGroups) {
                std::cout << "\t\t\t" << group_str(group) << std::endl;
            }
        }
    }
#endif

    return { isStartTensor, adjCountsTensor, adjValuesTensor, maxExCount };
}



template<typename scalar_t>
nms_result_t
    all_to_all_collapse(
        const CollapseRowsResult &collapseRowsRes,
        const AdjacencyResult &adjResult)
{
    auto counts = collapseRowsRes.ExCounts;
    auto embedQuads = collapseRowsRes.StridedMergeQuads;

    if (!embedQuads.is_contiguous()) {
        throw std::runtime_error("Input embed quads were not contiguous!");
    }

    torch::Tensor isLeadRow;
    if (counts.size(0) == 1) {
        isLeadRow = adjResult.IsLeadRow;
    } else {
        // For multiple examples: IsLeadRow will have true values beyond the extent of the number of quads
        // However, we know that Counts > 0 only happen within the extent, so the set intersection
        // tells us which rows are actually lead
        isLeadRow = torch::logical_and(adjResult.IsLeadRow, adjResult.AdjCounts > 0);
    }

    auto regionCounts = isLeadRow.sum(/*dim=*/ 1, /*keepdim=*/ false, torch::kInt64);

    const int64_t numOutQuads = counts.size(0) == 1 ? regionCounts.item<int64_t>() : regionCounts.sum().item<int64_t>();

    constexpr int32_t NUM_WARPS = 4;
    dim3 blockSize(NUM_WARPS * 32, 1, 1);
    dim3 gridSize(1, adjResult.MaxExCount, counts.size(0));

    torch::Tensor outQuads = torch::empty({ numOutQuads, 4, 2 }, embedQuads.options());
    torch::Tensor outConf = torch::empty({ numOutQuads }, embedQuads.options());

    device_a2a_collapse<NUM_WARPS, scalar_t>  KERNEL_ARG2(gridSize, blockSize) (
        counts.packed_accessor64<int32_t, 1>(),
        embedQuads.packed_accessor64<scalar_t, 3>(),
        isLeadRow.packed_accessor64<bool, 2>(),
        regionCounts.data_ptr<int64_t>(),
        adjResult.AdjCounts.packed_accessor64<int32_t, 2>(),
        adjResult.AdjValues.packed_accessor64<int32_t, 3>(),
        outQuads.packed_accessor64<scalar_t, 3>(),
        outConf.data_ptr<scalar_t>()
    );

    return { outQuads, outConf, regionCounts };
}

template<typename scalar_t>
nms_result_t cuda_quad_non_maximal_suppression_impl(
    torch::Tensor quads, torch::Tensor probs,
    scalar_t probThreshold, scalar_t iouThreshold,
    int64_t maxRegions, bool verbose)
{
    static const bool s_timerEnabled = true;
    static const bool s_verboseLevel2 = true;

    // Make sure there's a batch dimension
    if (quads.dim() == 4) {
        // B,H,W,V,2
        quads = quads.unsqueeze(0);
        // B,H,W
        probs = probs.unsqueeze(0);
    }

    //print_tensor_vec_stats2("NMS Input (quads, probs): ", { quads, probs });

    double msRowCollapse = -1,
           msAdjacency = -1,
           msA2ACollapse = -1,
           msTotal = -1;

    CollapseRowsResult collapseRows;
    AdjacencyResult adjacency;
    torch::Tensor retQuads, retConf, regionCounts;

    {
        CudaStoreTimer tTotal{msTotal, s_timerEnabled};
        {
            CudaStoreTimer t{msRowCollapse, s_timerEnabled && verbose && s_verboseLevel2};

            // First combine all of the quads in each row
            collapseRows = collapse_rows(quads, probs, probThreshold, iouThreshold);

            if (collapseRows.TotalNumQuads == 0) {
                return {
                    torch::empty({ 0, 4, 2 }, quads.options()),
                    torch::empty({ 0 }, probs.options()),
                    collapseRows.ExCounts.toType(torch::kInt64)
                };
            }
        }
        {
            CudaStoreTimer t{msAdjacency, s_timerEnabled && verbose && s_verboseLevel2};
            adjacency = compute_all_to_all_adjacency(collapseRows, iouThreshold);
        }
        {
            CudaStoreTimer t{msA2ACollapse, s_timerEnabled && verbose && s_verboseLevel2};
            std::tie(retQuads, retConf, regionCounts) = all_to_all_collapse<scalar_t>(collapseRows, adjacency);
        }
    }

#ifndef NDEBUG
    assert(regionCounts.sum().item<int64_t>() == retQuads.size(0));
#endif

    //print_tensor_vec_stats2("    Full NMS (quads, conf, counts): ", { retQuads, retConf, retCounts });

    if (s_timerEnabled && verbose) {
        std::cout << "NMS Cuda " << retQuads.size(0)
             << " - Row Collapse (" << quads.size(0) << ", " << quads.size(1) << ", " << quads.size(2) << ") - (" << collapseRows.TotalNumQuads << "): " << msRowCollapse << "ms"
             << ", Adjacency (" << adjacency.AdjCounts.sum(torch::kInt32).item<int32_t>() << "): " << msAdjacency << "ms"
             << ", A2A Collapse (" << retQuads.size(0) << "): " << msA2ACollapse << "ms"
             << ", Total: " << msTotal << "ms"
             << std::endl;
    }

    return { retQuads, retConf, regionCounts };
}

nms_result_t cuda_quad_non_maximal_suppression(
    torch::Tensor quads, torch::Tensor probs,
    float probThreshold, float iouThreshold,
    int64_t kernelHeight, int64_t kernelWidth,
    int64_t maxRegions, bool verbose)
{
    nms_result_t ret;

    ret = cuda_quad_non_maximal_suppression_impl<float>(
        quads.toType(torch::kFloat32), probs.toType(torch::kFloat32),
        probThreshold, iouThreshold,
        maxRegions, verbose
    );

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    //     quads.scalar_type(),
    //     "cuda_quad_non_maximal_suppression_impl",
    //     ([&] {
    //         ret = cuda_quad_non_maximal_suppression_impl<scalar_t>(
    //             move(quads), move(probs),
    //             probThreshold, iouThreshold,
    //             maxRegions
    //         );
    //     })
    // );

    return ret;
}
