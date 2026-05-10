# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target encoder base class (inference-only)."""

import torch
from nemotron_ocr.inference.post_processing.data.text_region import Batch, TextRegion


class TargetEncoderBase(object):
    """Base class for target encoders that convert model outputs back to structured text regions."""

    def __init__(self, input_size, amp_opt, verbose=False):
        self.input_size = input_size
        self.amp_opt = amp_opt
        self.verbose = verbose

    def convert_targets_to_labels(
        self, target_dict, image_size, limit_idxs=None, is_gt=True, **kwargs
    ) -> Batch:
        raise NotImplementedError("Subclasses must implement this function!")

    def is_recognition(self):
        return False

    def get_charset(self):
        raise ValueError("This target encoder does not support charsets!")

    def get_rrect_and_quad(self, region: TextRegion):
        verts = region.region.vertices
        if verts.shape[0] < 4:
            verts = torch.cat([verts, verts[:1].expand(4 - verts.shape[0], 2)])
        try:
            rrect = region.region.min_rrect
        except:  # noqa: E722
            rrect = region.region.bounds_quad
            region.valid = False
        vtx_count = verts.shape[0]

        quad = verts if vtx_count == 4 else rrect

        return rrect, quad
