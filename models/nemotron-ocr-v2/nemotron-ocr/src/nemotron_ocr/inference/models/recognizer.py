# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch
import torch.nn as nn

from nemotron_ocr.inference.models import blocks

logger = logging.getLogger(__name__)


class TransformerRecognizer(nn.Module):
    """
    Transformer-based text recognizer.
    
    Supports two variants:
    - standard: Original architecture without pre-norm layers
    - prenorm: Architecture with pre_norm layer and final tx.norm (more stable for training)
    
    The variant is auto-detected from model_config.json or state dict keys.
    """
    def __init__(
        self, 
        nic: int, 
        num_tokens: int, 
        max_width: int,
        use_pre_norm: bool = False,
        use_final_norm: bool = False,
        norm_first: bool = False,
        depth: int = 128,
        num_layers: int = 3,
        nhead: int = 8,
        dim_feedforward: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.fixed_width = max_width > 0
        self.inference_mode = False
        self.use_pre_norm = use_pre_norm
        self.use_final_norm = use_final_norm

        max_width = abs(max_width)

        final_depth = depth * 2
        if dim_feedforward is None:
            dim_feedforward = final_depth

        self.feature_depth = final_depth

        CONV_SHAPE = (3, 3)
        PAD_SHAPE = tuple((c - 1) // 2 for c in CONV_SHAPE)

        self.encoder = nn.Sequential(
            blocks.conv2d_block(nic, nic, 3, padding=1),
            blocks.conv2d_block(nic, nic * 2, 3, padding=1),
            nn.MaxPool2d((2, 1)),
            blocks.conv2d_block(nic * 2, nic * 2, CONV_SHAPE, padding=PAD_SHAPE),
            blocks.conv2d_block(nic * 2, depth * 2, CONV_SHAPE, padding=PAD_SHAPE),
            nn.MaxPool2d((2, 1)),
            blocks.conv2d_block(depth * 2, final_depth, CONV_SHAPE, padding=PAD_SHAPE),
            blocks.conv2d_block(final_depth, final_depth * 2, CONV_SHAPE, padding=PAD_SHAPE),
            nn.MaxPool2d((2, 1)),
            blocks.conv2d_block(final_depth * 2, final_depth, 1),
        )

        self.position_encoding = nn.Parameter(torch.randn(1, final_depth, max_width))

        # Optional standalone pre-norm layer (legacy variant)
        if use_pre_norm:
            self.pre_norm = nn.LayerNorm(final_depth)
        
        # norm_first: use pre-norm inside each TransformerEncoderLayer (no separate pre_norm layer)
        # use_pre_norm: legacy separate pre_norm layer before the transformer
        use_norm_first = norm_first or use_pre_norm
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=final_depth, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.0, 
            batch_first=True,
            norm_first=use_norm_first,
        )
        
        final_norm = nn.LayerNorm(final_depth) if (use_final_norm or norm_first) else None
        
        self.tx = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=final_norm,
        )
        self.classifier = nn.Linear(final_depth, num_tokens)

    def forward(self, x: torch.Tensor, cpu_region_counts: Optional[torch.Tensor] = None):
        x = self.encoder(x).squeeze(2)

        if self.fixed_width:
            pos = self.position_encoding
        else:
            pos = self.position_encoding[..., : x.shape[2]]
        x = x + pos

        # B,C,T -> B,T,C
        x = x.permute(0, 2, 1).contiguous()

        # Apply standalone pre-norm if present (legacy variant)
        if self.use_pre_norm:
            x = self.pre_norm(x)

        x = self.tx(x)

        y = self.classifier(x)

        return y, x
