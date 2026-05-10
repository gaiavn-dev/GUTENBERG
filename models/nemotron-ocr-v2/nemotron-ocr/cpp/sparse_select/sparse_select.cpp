// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sparse_select.h"

#include <algorithm>

#include "../common.h"

using namespace std;

std::tuple<torch::Tensor, std::vector<torch::Tensor>> sparse_select(torch::Tensor sparseCounts,
                                                                    const std::vector<torch::Tensor> sparseTensors,
                                                                    torch::Tensor selectIndices)
{
    bool is_gpu = sparseCounts.is_cuda();

    auto sparseCountsCPU = sparseCounts.cpu();

    auto sortedSelect = get<0>(torch::sort(selectIndices));

    vector<torch::Tensor> retTensors;
    for (const torch::Tensor &t : sparseTensors) {
        retTensors.push_back(t.index({sortedSelect}));
    }

    vector<int64_t> offsets(1 + sparseCountsCPU.size(0));

    auto sparseCtAccess = sparseCountsCPU.accessor<int64_t, 1>();

    for (int64_t i = 0; i < sparseCountsCPU.size(0); ++i) {
        offsets[i + 1] = sparseCtAccess[i] + offsets[i];
    }

    // cout << "Offsets: " << offsets << endl;

    auto retCounts = torch::zeros_like(sparseCountsCPU);

    auto retCtAccess = retCounts.accessor<int64_t, 1>();
    auto idxAccess = sortedSelect.accessor<int64_t, 1>();

    for (int64_t i = 0; i < idxAccess.size(0); ++i) {
        int64_t idx = idxAccess[i];

        int64_t batchIdx = std::upper_bound(begin(offsets), end(offsets), idx) - begin(offsets) - 1;

        // cout << "Index: " << idx << ", Batch Index: " << batchIdx << endl;

        retCtAccess[batchIdx] += 1;
    }

    if (is_gpu) {
        retCounts = retCounts.to(sparseCounts);
    }

    return make_tuple(retCounts, retTensors);
}
