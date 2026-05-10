# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from nemotron_ocr.inference.models.detector.aspp import ASPP
from nemotron_ocr.inference.models.detector import regnet

logger = logging.getLogger(__name__)


def get_prior_offsets(output_shape, downsample):
    """
    Returns the locations of the priors in normalized image space.

    Args:
        shape (tensor): Contains the output layer dimensions as [height, width]. This is used to
                        normalize the prior offsets.

    Returns:
        priors (HxWx2 tensor): contains prior offsets in normalized coordinates.
                               second dimension contains the x, y offsets.
    """
    x_priors = torch.arange(0, output_shape[1], dtype=torch.float16) * downsample
    y_priors = torch.arange(0, output_shape[0], dtype=torch.float16) * downsample

    x_priors += downsample / 2
    y_priors += downsample / 2

    x_priors = x_priors.reshape(1, -1, 1).repeat(output_shape[0], 1, 1)
    y_priors = y_priors.reshape(-1, 1, 1).repeat(1, output_shape[1], 1)

    priors = torch.cat((x_priors, y_priors), dim=2)

    return priors


class extractor(nn.Module):
    def __init__(self, backbone="regnet_y_8gf"):
        super().__init__()

        backbone = getattr(regnet, backbone)(pretrained=True)

        self.depths = backbone.channel_counts
        self.base = backbone.stem
        self.levels = nn.ModuleList(backbone.trunk_output)

        self.step = 0

        self.downsample = 4

    def set_current_and_total_steps(self, current_step, total_steps):
        self.step = current_step

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # logger.info(f'Input shape: {x.shape}')
        x = self.base(x)

        # out = [x]
        out = []
        for m in self.levels:
            x = m(x)

            out.append(x)

        # logger.info(f'Extraction levels: {[t.shape for t in out]}')

        if self.training:
            self.step += 1
        return tuple(out)


def conv_block(*args, **kwargs):
    conv = nn.Conv2d(*args, bias=False, **kwargs)
    bn = nn.BatchNorm2d(conv.out_channels)
    return nn.Sequential(
        conv,
        bn,
        nn.ReLU(inplace=True),
    )


class merge(nn.Module):
    def __init__(self, extractor_depths):
        super().__init__()

        # Go from deepest to most shallow
        extractor_depths = extractor_depths[::-1]

        pre_upsamples = []
        pre_sides = []
        post_upsamples = []
        next_depth = 512
        prev_depth = extractor_depths[0]
        num_features = [extractor_depths[0]]
        for i in range(1, len(extractor_depths)):
            ds_depth = min(prev_depth // 2, 512)
            # side_depth = extractor_depths[i] // 2
            side_depth = extractor_depths[i]
            depth = side_depth + ds_depth

            pre_upsamples.append(nn.Sequential(conv_block(prev_depth, ds_depth, 1)))
            pre_sides.append(
                nn.Sequential(
                    # conv_block(extractor_depths[i], side_depth, 1)
                    nn.Identity()
                )
            )
            post_upsamples.append(
                nn.Sequential(
                    conv_block(depth, next_depth, 1),
                    conv_block(next_depth, next_depth, 3, padding=1),
                )
            )
            num_features.append(next_depth)
            prev_depth = next_depth
            next_depth //= 2

        self.pre_upsamples = nn.ModuleList(pre_upsamples)
        self.pre_sides = nn.ModuleList(pre_sides)
        self.post_upsamples = nn.ModuleList(post_upsamples)
        self.final = nn.Sequential(
            conv_block(prev_depth, prev_depth, 3, padding=1),
        )

        self.num_features = num_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # From deepest to most shallow
        x = x[::-1]
        feats = [x[0]]

        y = x[0]
        for i in range(len(x) - 1):
            y = self.pre_upsamples[i](y)
            y = self.interpolate(y)
            side = self.pre_sides[i](x[i + 1])
            y = torch.cat((y, side), 1)
            y = self.post_upsamples[i](y)
            feats.append(y)

        y = self.final(y)

        feats[-1] = y

        return tuple(feats)

    def interpolate(self, x: torch.Tensor):
        if x.dtype != torch.bfloat16:
            return F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            # TODO(mranzinger): Currently F.interpolate doesn't support bfloat16
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[3], 1)
            x = x.repeat(1, 1, 1, 2, 1, 2)
            x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[4] * 2)
            return x


class output(nn.Module):
    def __init__(self, num_features, downsample, coordinate_mode, scope=512):
        super().__init__()

        self._prior_offsets = dict()

        self.coordinate_mode = coordinate_mode

        self.downsample = downsample

        num_features = num_features[-1]

        self.scope = scope

        self.slices = [("confidence", slice(0, 1))]
        if self.do_quads():
            end = self.slices[-1][1].stop
            self.slices.append(("quads", slice(end, end + 8)))
        if self.do_rbox():
            end = self.slices[-1][1].stop
            self.slices.append(("rbox_coord", slice(end, end + 4)))
            self.slices.append(("rbox_rot", slice(end + 4, end + 5)))

        self.preds = nn.Sequential(
            ASPP(num_features, num_features),
            ASPP(num_features, num_features),
            conv_block(num_features, num_features, 3, padding=1),
            nn.Conv2d(num_features, self.slices[-1][1].stop, 1, bias=False),
        )

        self.slices = {k: v for k, v in self.slices}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_prior_offsets(self, y):
        output_size = (y.shape[-2], y.shape[-1])

        if output_size in self._prior_offsets:
            return self._prior_offsets[output_size].to(y)

        # HWC
        prior_offsets = get_prior_offsets(output_size, self.downsample)
        # CHW
        prior_offsets = prior_offsets.permute(2, 0, 1).contiguous()
        # (4C)HW
        prior_offsets = prior_offsets.repeat(4, 1, 1)
        # BCHW
        prior_offsets = prior_offsets.unsqueeze(0)

        prior_offsets = prior_offsets.to(y)

        self._prior_offsets[output_size] = prior_offsets

        return prior_offsets

    def adjust_offsets(self, offsets):
        return offsets + self.get_prior_offsets(offsets)

    def do_quads(self):
        return self.coordinate_mode in ("QUAD", "BOTH")

    def do_rbox(self):
        return self.coordinate_mode in ("RBOX", "BOTH")

    def forward(
        self, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[torch.Tensor]]:
        x = feats[-1]

        preds = self.preds(x)

        conf = preds[:, self.slices["confidence"]].squeeze(1).contiguous()

        if self.do_quads():
            offsets = preds[:, self.slices["quads"]]
            offsets = torch.tanh(offsets) * self.scope
            offsets = self.adjust_offsets(offsets)
            offsets = offsets.permute(0, 2, 3, 1).contiguous()
            offsets = offsets.reshape(*offsets.shape[:-1], 4, 2)
        else:
            offsets = None

        if self.do_rbox():
            rboxes = preds[:, self.slices["rbox_coord"]]
            rboxes = F.relu(rboxes, inplace=True) * self.scope

            rot = preds[:, self.slices["rbox_rot"]]
            rot = F.hardtanh(rot, min_val=-1, max_val=1) * math.pi

            rboxes = torch.cat((rboxes, rot), dim=1)
            rboxes = rboxes.permute(0, 2, 3, 1).contiguous()
        else:
            rboxes = None

        return conf, offsets, rboxes, x


class FOTSDetector(nn.Module):
    def __init__(
        self, verbose=True, coordinate_mode: str = "RBOX", backbone: str = "regnet_y_8gf", 
        scope: int = 512, **kwargs
    ):
        super().__init__()

        self.extractor = extractor(backbone, **kwargs)
        self.merge = merge(self.extractor.depths)
        self.num_features = self.merge.num_features
        self.output = output(self.num_features, self.extractor.downsample, coordinate_mode, scope=scope)
        self.verbose = verbose
        self.inference_mode = False
        self.scope = scope  # Store for reference

        self.downsample = self.extractor.downsample

        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).reshape(1, -1, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).reshape(1, -1, 1, 1),
        )

    def set_current_and_total_steps(self, current_step, total_steps):
        self.extractor.set_current_and_total_steps(current_step, total_steps)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        x = (x - self.input_mean) / self.input_std
        feats = self.extractor(x)

        mg = self.merge(feats)

        main_op = self.output(mg)

        return main_op
