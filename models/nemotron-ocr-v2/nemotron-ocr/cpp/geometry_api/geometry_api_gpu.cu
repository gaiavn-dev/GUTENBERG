// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "geometry_api.h"

#include "../geometry.h"
#include "../cuda_intellisense.cuh"
#include "geometry_api_common.h"

#include <trove/ptr.h>

using namespace std;


template<typename T>
struct RRect_ {
    T Data[5];

    template<typename index_t>
    __device__
    const T &operator[](index_t i) const { return Data[i]; }
    template<typename index_t>
    __device__
    T &operator[](index_t i) { return Data[i]; }
};

template<typename T>
__global__
void device_rrect_to_quads_gpu(torch::PackedTensorAccessor64<T, 2> rrectAccess,
                               torch::PackedTensorAccessor64<T, 3> quadsAccess,
                               int64_t numRows, int64_t numCols,
                               T cellSize)
{
    typedef Point_<T> Pointf;
    typedef RRect_<T> RRectf;
    typedef InPlaceQuad_<T> Quadf;
    constexpr T TWO = 2;

    const int64_t jobIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (jobIdx >= rrectAccess.size(0)) {
        return;
    }

    int64_t row = jobIdx / numCols;
    const int64_t col = jobIdx - (row * numCols);
    row = row % numRows;

    auto rawRRect = reinterpret_cast<RRectf*>(rrectAccess.data());
    auto rawQuad = reinterpret_cast<Quadf*>(quadsAccess.data());
#if defined(NDEBUG)
    trove::coalesced_ptr<RRectf> pRRect(rawRRect);
    trove::coalesced_ptr<Quadf> pQuad(rawQuad);
#else
    auto pRRect = rawRRect;
    auto pQuad = rawQuad;
#endif

    RRectf rrect = pRRect[jobIdx];

    T cellOff = cellSize / TWO;
    Quadf cvQuad = cvt_rrect_to_quad<T>(rrect, cellSize, cellOff, col, row);

    pQuad[jobIdx] = cvQuad;
}

torch::Tensor rrect_to_quads_gpu(torch::Tensor rrects, float cellSize)
{
    if (!rrects.is_contiguous()) {
        throw std::runtime_error("Expected the rrects to be contiguous!");
    }

    torch::Tensor quads = torch::empty({ rrects.size(0), rrects.size(1), rrects.size(2), 4, 2 }, rrects.options());

    auto rrFlat = rrects.flatten(0, 2);
    auto qFlat = quads.flatten(0, 2);

    dim3 blockSize(96);
    dim3 gridSize(div_up(qFlat.size(0), blockSize.x));

    if (quads.numel() > 0) {
        AT_DISPATCH_FLOATING_TYPES(
            quads.scalar_type(),
            "cuda_rrect_to_quads",
            ([&] {

                device_rrect_to_quads_gpu<scalar_t> KERNEL_ARG2(gridSize, blockSize) (
                    rrFlat.packed_accessor64<scalar_t, 2>(),
                    qFlat.packed_accessor64<scalar_t, 3>(),
                    rrects.size(1), rrects.size(2),
                    cellSize
                );

            })
        );
    }

    return quads;
}

template<typename scalar_t>
__global__
void device_rrect_to_quads_backward_gpu(torch::PackedTensorAccessor64<scalar_t, 2> rrect,
                                        torch::PackedTensorAccessor64<scalar_t, 3> gradOutput,
                                        torch::PackedTensorAccessor64<scalar_t, 2> gradInput)
{
    const int64_t jobIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (jobIdx >= rrect.size(0)) return;

    assign_grad_rrect_to_quad<scalar_t>(rrect[jobIdx], gradOutput[jobIdx], gradInput[jobIdx]);
}


torch::Tensor rrect_to_quads_backward_gpu(torch::Tensor rrects, torch::Tensor gradOutput)
{
    auto gradInput = torch::empty_like(rrects);

    auto flatRRects = rrects.reshape({ -1, 5 });
    auto flatGradOutput = gradOutput.reshape({ -1, 4, 2 });
    auto flatGradInput = gradInput.reshape({ -1, 5 });

    dim3 blockSize(32);
    dim3 gridSize(div_up(rrects.size(0) * rrects.size(1) * rrects.size(2), blockSize.x));

    if (rrects.numel() > 0) {
        AT_DISPATCH_FLOATING_TYPES(
            rrects.scalar_type(),
            "cuda_rrect_to_quads_backward",
            ([&] {
                device_rrect_to_quads_backward_gpu KERNEL_ARG2(gridSize, blockSize) (
                    flatRRects.packed_accessor64<scalar_t, 2>(),
                    flatGradOutput.packed_accessor64<scalar_t, 3>(),
                    flatGradInput.packed_accessor64<scalar_t, 2>()
                );
            })
        );
    }

    return gradInput;
}
