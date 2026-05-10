# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.autograd import Function
from nemotron_ocr_cpp import (
    quad_rectify_backward,
    quad_rectify_calc_quad_width,
    quad_rectify_forward,
)


def _quad_wrap(fn, quads, *args, **kwargs):
    if quads.dtype == torch.float16:
        quads = quads.float()
    ret = fn(quads, *args, **kwargs)
    return ret


def _quad_rectify_forward(*args, **kwargs):
    return _quad_wrap(quad_rectify_forward, *args, **kwargs)


def _quad_rectify_backward(*args, **kwargs):
    return _quad_wrap(quad_rectify_backward, *args, **kwargs)


class QuadRectifyFunction(Function):
    @staticmethod
    def forward(
        ctx,
        quads,
        image_height,
        image_width,
        output_height,
        output_width,
        round_factor=16,
        isotropic=True,
    ):
        if output_width <= 0:
            widths = quad_rectify_calc_quad_width(
                quads.float(), output_height, round_factor, -output_width
            )
            output_width = widths.max().item()

        output = _quad_rectify_forward(
            quads,
            int(image_height),
            int(image_width),
            int(output_height),
            int(output_width),
            isotropic=isotropic,
        )
        ctx.save_for_backward(quads)
        ctx.image_shape = (int(image_height), int(image_width))

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = _quad_rectify_backward(
                ctx.saved_variables[0], grad_output, *ctx.image_shape
            )
        return grad_input, None, None, None, None, None, None


class QuadRectify(torch.nn.Module):
    def __init__(self, output_height, output_width, round_factor=16, isotropic=True):
        super().__init__()
        self.output_height = output_height
        self.output_width = output_width
        self.round_factor = round_factor
        self.isotropic = isotropic

    def forward(self, quads, image_height, image_width):
        return QuadRectifyFunction.apply(
            quads,
            image_height,
            image_width,
            self.output_height,
            self.output_width,
            self.round_factor,
            self.isotropic,
        )
