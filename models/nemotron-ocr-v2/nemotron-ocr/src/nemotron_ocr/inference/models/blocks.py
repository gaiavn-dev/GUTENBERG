# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model blocks used by the recognizer and relational model."""

import torch.nn as nn


def get_activation(name):
    """Returns a pytorch activation layer of type 'name', where 'name' is a string."""
    if isinstance(name, nn.Module):
        return name

    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    if name == "none":
        return None
    raise ValueError(
        "Unsupported activation type: {}. " "Ensure activation name is all lower case.".format(name)
    )


def conv2d_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode="zeros",
    activation="relu",
    batch_norm=True,
):
    """Returns a Conv2d + optional BatchNorm + optional activation as nn.Sequential."""
    items = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        ),
    ]

    if batch_norm:
        items.append(nn.BatchNorm2d(out_channels))

    act = get_activation(activation)
    if act:
        items.append(act)

    return nn.Sequential(*items)
