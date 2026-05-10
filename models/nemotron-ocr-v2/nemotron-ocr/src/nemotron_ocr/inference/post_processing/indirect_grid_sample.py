# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torch.autograd import Function
from nemotron_ocr_cpp import (
    indirect_grid_sample_forward,
    indirect_grad_sample_backward,
)

logger = logging.getLogger(__name__)


class IndirectGridSampleFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, input_indices, mode="bilinear"):
        val = indirect_grid_sample_forward(input, grid, input_indices, mode)

        ctx.mode = mode
        ctx.save_for_backward(input, grid, input_indices)

        return val

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, input_indices = ctx.saved_tensors

        grad_input, grad_grid = indirect_grad_sample_backward(
            grad_output, input, grid, input_indices, ctx.mode
        )

        return grad_input, grad_grid, None, None


def indirect_grid_sample(
    input: torch.Tensor, grid: torch.Tensor, input_indices: torch.Tensor, mode="bilinear"
):
    return IndirectGridSampleFunction.apply(input, grid, input_indices, mode)


class IndirectGridSample(torch.nn.Module):
    def __init__(self, mode="bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, input: torch.Tensor, grid: torch.Tensor, input_indices: torch.Tensor):
        return indirect_grid_sample(input, grid, input_indices, self.mode)
