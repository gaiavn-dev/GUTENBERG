// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/torch.h>

inline
torch::Tensor region_counts_to_indices(torch::Tensor regionCounts, int64_t numOutputs)
{
    // If there's only one example, we can trivially return idx 0 for all
    if (regionCounts.size(0) == 1) {
        return torch::zeros({ numOutputs }, regionCounts.options().dtype(torch::kInt64));
    }

    // regionCounts will be some tensor like [ 5, 1, 10, 2 ] which means that the first 5 outputs
    // correspond to the first input, the next output to the second input, 10 to the third, and so on.

    // We want to convert this to instead have an entry for each output which specifies the index of the corresponding input.
    // To do this, we can count the number of times the output index exceeds the cumulative input counts.
    // e.g. the cumulative region count for the above tensor is [ 5, 6, 16, 18 ].
    // The output indices 0-4 are not greater than or equal to any cumulative count, so they get the input index of 0.
    // The output index 5 is equal to a single count, therefore index 1.
    // The outputs 6-15 are all greater than or equal to two cumulative counts, therefore index 2.
    // And so on.

    auto indices = torch::arange(regionCounts.size(0), regionCounts.options().dtype(torch::kInt64));

    auto outputIndices = torch::repeat_interleave(indices, regionCounts, /*dim=*/ 0, /*output_size=*/ numOutputs);

    return outputIndices;
}

torch::Tensor gpu_indirect_grid_sample_forward(torch::Tensor input, torch::Tensor grid, torch::Tensor inputIndices, const std::string &method);
torch::Tensor cpu_indirect_grid_sample_forward(torch::Tensor input, torch::Tensor grid, torch::Tensor inputIndices, const std::string &method);
std::vector<torch::Tensor> gpu_indirect_grad_sample_backward(torch::Tensor gradOutput, torch::Tensor input, torch::Tensor grid, torch::Tensor inputIndices, const std::string &method);

inline
torch::Tensor indirect_grid_sample_forward(torch::Tensor input, torch::Tensor grid, torch::Tensor inputIndices, const std::string &method)
{
    if (input.is_cuda() != grid.is_cuda() || input.is_cuda() != inputIndices.is_cuda()) {
        throw std::runtime_error("Input tensors must all be on the same device!");
    }
    if (inputIndices.size(0) != grid.size(0)) {
        throw std::runtime_error("The batch dimensions must match!");
    }
    if (grid.size(-1) != 2) {
        throw std::runtime_error("The final grid dimension must be 2.");
    }

    if (input.is_cuda()) {
        return gpu_indirect_grid_sample_forward(std::move(input), std::move(grid), std::move(inputIndices), method);
    } else {
        return cpu_indirect_grid_sample_forward(std::move(input), std::move(grid), std::move(inputIndices), method);
    }
}

inline
std::vector<torch::Tensor> indirect_grad_sample_backward(torch::Tensor gradOutput, torch::Tensor input, torch::Tensor grid, torch::Tensor inputIndices, const std::string &method)
{
    if (gradOutput.is_cuda()) {
        return gpu_indirect_grad_sample_backward(std::move(gradOutput), std::move(input), std::move(grid), std::move(inputIndices), method);
    } else {
        throw std::runtime_error("Not implemented!");
    }
}
