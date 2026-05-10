# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any

import torch


def is_named_tuple(obj):
    """
    Return where or not the specified instance is a namedtuple.

    NOTE: Not guaranteed to be correct, but close.

    Args:
        obj (object): Some object to test.
    """
    return isinstance(obj, tuple) and getattr(obj, "_fields", None) is not None


def cat(tensors, *rest_shape, dtype=torch.float32) -> torch.Tensor:
    if tensors:
        return torch.cat(tensors)
    else:
        return torch.empty(0, *rest_shape, dtype=dtype)


def options(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Returns as a dict the dtype and device options for a tensor. This allows you
    to construct a new tensor with a compatible format.

    e.g.
    new_tensor = torch.empty(<shape>, **options(other_tensor))
    """
    return {"dtype": tensor.dtype, "device": tensor.device}


def f_measure(*args):
    acc = 0
    for v in args:
        if torch.is_tensor(v):
            v = v.clamp_min(1e-8)
        elif v <= 0:
            v = 1e-8
        acc += 1.0 / v

    fmeasure = len(args) / acc

    return fmeasure
