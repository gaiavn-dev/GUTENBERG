# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F


def pad_to_square(img: torch.Tensor, target_length: int, how="bottom_right") -> torch.Tensor:
    """
    Pads the input image to a square shape with the specified size.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        size (int): The target size for both height and width.

    Returns:
        torch.Tensor: Padded image tensor of shape (C, size, size).
    """
    _, h, w = img.shape

    if how == "center":
        pad_h = (target_length - h) // 2
        pad_w = (target_length - w) // 2
        return F.pad(
            img, (pad_w, target_length - w - pad_w, pad_h, target_length - h - pad_h), value=1.0
        )
    elif how == "bottom_right":
        pad_h = target_length - h
        pad_w = target_length - w
        return F.pad(img, (0, pad_w, 0, pad_h), value=1.0)
    else:
        raise ValueError(f"Unsupported padding method: {how}")


def interpolate_and_pad(
    images: torch.Tensor, pad_color: torch.Tensor, infer_length: int
) -> torch.Tensor:
    """
    Interpolates the input images to a specified height and pads them to a specified width.

    Args:
        images (torch.Tensor): Input image tensor of shape (B, C, H, W).
        pad_color (torch.Tensor): Padding color tensor (should be on same device as images).
        infer_length (int): The target size for interpolation.

    Returns:
        torch.Tensor: Interpolated and padded image tensor of shape (B, C, infer_length, pad_infer_width).
    """
    pad_infer_width = int(math.ceil(infer_length / 128) * 128)

    device = pad_color.device if pad_color.is_cuda else 'cuda'
    images_gpu = images.to(device, non_blocking=True)

    rs_images = F.interpolate(
        images_gpu, size=(infer_length, infer_length), mode="bilinear", align_corners=True
    )

    pad_w = pad_infer_width - infer_length
    if pad_w == 0:
        return rs_images

    return F.pad(rs_images, (0, pad_w), value=0.0)
