# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atrous Spatial Pyramid Pooling implementation."""

import torch
from torch import nn


def _grow(rate, power):
    if isinstance(rate, (list, tuple)):
        return tuple(_grow(r, power) for r in rate)
    return int(rate**power)


class ASPP(nn.Module):
    """A class definining an ASPP module."""

    def __init__(self, in_channels, num_channels, dropout=0.0, growth_rate=2):
        """Initialize an ASPP.

        Args:
            in_channels (int): Number of input channels.
            num_channels (int): Number of channels in each branch of the ASPP.
            norm_type (str): Type of normalization layer, supported: 'batch_norm',
                'sync_batch_norm', 'group_norm'. Default: 'off'.
            norm_args (dict): Additional arguments given to the normalization layer. This
                includes for example:
                    - In case 'norm_type' == 'batch_norm' or 'sync_batch_norm', the
                    'momentum' parameter.
                    - In case 'norm_type' == 'group_norm', this includes the 'num_groups'
                    parameter.
        """
        super().__init__()

        norm_layer = nn.BatchNorm2d
        use_bias = False

        kernels = []
        num_kernels = 7
        for i in range(num_kernels):
            dilation = _grow(growth_rate, i)
            kernels.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_channels,
                        kernel_size=3,
                        stride=1,
                        dilation=dilation,
                        padding=dilation,
                        bias=use_bias,
                    ),
                    norm_layer(num_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.kernels = nn.ModuleList(kernels)

        # Global average pooling.
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        # Output convolution.
        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=(1 + num_kernels) * num_channels,
                out_channels=num_channels,
                kernel_size=1,
                stride=1,
                bias=use_bias,
            ),
            norm_layer(num_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        """The module forward function.

        Args:
            x (torch.tensor): The input tensor.

        Returns:
            torch.tensor: The output tensor.
        """
        outs = [kernel(x) for kernel in self.kernels]

        global_pool = self.global_pool(x).expand(-1, -1, *x.shape[2:])
        outs.append(global_pool)

        concatenated = torch.cat(outs, dim=1)

        out = self.final(concatenated)

        if x.shape == out.shape:
            return x + out
        return out
