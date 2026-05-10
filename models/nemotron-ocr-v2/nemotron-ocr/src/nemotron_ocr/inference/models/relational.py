# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import torch
import torch.nn as nn

from nemotron_ocr.inference.models import blocks
from nemotron_ocr.inference.models.utils import options

from nemotron_ocr_cpp import (
    quad_rectify_calc_quad_width,
    ragged_quad_all_2_all_distance_v2,
)


logger = logging.getLogger(__name__)


NULL_CONNECTION_WEIGHT = -math.inf
DEFAULT_CHUNK = 128


def _pad_flat_to_batched(flat: torch.Tensor, region_counts: torch.Tensor, k_max: int, pad_value: float = 0.0):
    """Reshape flat [N_total, ...] tensors into padded [B, k_max, ...] batches."""
    batch_size = int(region_counts.shape[0])
    rest = flat.shape[1:]
    out = flat.new_full((batch_size, k_max, *rest), pad_value)
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=flat.device)
    offsets[1:] = torch.cumsum(region_counts.to(device=flat.device, dtype=torch.long), dim=0)
    for i in range(batch_size):
        region_count = int(region_counts[i].item())
        if region_count > 0:
            out[i, :region_count] = flat[offsets[i] : offsets[i] + region_count]
    return out


def _batched_directions(quads: torch.Tensor, to_centers: torch.Tensor):
    """Compute cosine direction features for batched [B, K, Z, ...] tensors."""
    quads = quads.detach()
    to_centers = to_centers.detach()

    pt0 = (quads[:, :, 0] + quads[:, :, 3]) / 2
    pt1 = (quads[:, :, 1] + quads[:, :, 2]) / 2

    direction = pt1 - pt0
    direction = direction / direction.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = direction.unsqueeze(2).expand_as(to_centers)

    centers = (pt0 + pt1) / 2
    vec_other = to_centers - centers.unsqueeze(2)
    dir_other = vec_other / vec_other.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)

    return (direction * dir_other).sum(dim=-1)


class GlobalRelationalModel(nn.Module):
    def __init__(
        self,
        num_input_channels,
        recog_feature_depth,
        k=32,
        dropout=0.1,
        num_layers=4,
        chunk_size=DEFAULT_CHUNK,
    ):
        super().__init__()

        num_input_channels = num_input_channels[-1]
        self.pos_channels = 14  # 64
        self.current_step = 0
        self.total_steps = 1
        self.k = k
        self.chunk_size = chunk_size
        self.quad_rectify_grid_size = (2, 3)
        self.quad_downscale = 1024.0
        self.inference_mode = False

        self.grid_size = [2, 3]
        self.isotropic = False

        self.initial_depth = 128

        self.rect_proj = blocks.conv2d_block(num_input_channels, num_input_channels, 1)

        self.recog_tx = nn.Linear(recog_feature_depth, num_input_channels)

        # Reserve 2 channels for the joint distance and angle embeddings
        initial_depth = self.initial_depth - 1 - self.pos_channels
        cb_input = 2 * num_input_channels  # + self.pos_channels
        self.combined_proj = nn.Sequential(
            nn.Linear(cb_input, cb_input),
            nn.BatchNorm1d(cb_input),
            nn.ReLU(),
            nn.Linear(cb_input, initial_depth),
            nn.BatchNorm1d(initial_depth),
            nn.ReLU(),
        )

        dim = 2 * self.initial_depth

        self.encoder = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    dim, 8, 2 * dim, batch_first=True, dropout=dropout, norm_first=True
                ),
                num_layers=num_layers,
            ),
            nn.Linear(dim, 3),
        )

    def _chunked_forward(self, fn, x, *extra, pad_extra_ones=False):
        """Run *fn* in fixed ``chunk_size`` batches along dim-0.

        Pads the last chunk so every call sees the same tensor shape,
        preventing cuDNN autotuning on varying batch sizes.
        ``extra`` tensors are sliced/padded in sync with ``x``.
        """
        n = x.shape[0]
        cs = max(1, self.chunk_size)
        if n == 0:
            return fn(x, *extra)
        parts = []
        for start in range(0, n, cs):
            end = min(start + cs, n)
            real_n = end - start
            xc = x[start:end]
            ec = [e[start:end] for e in extra]
            if real_n < cs:
                xc = torch.cat((xc, xc[:1].expand(cs - real_n, *[-1] * (xc.ndim - 1))), dim=0)
                for i, e in enumerate(ec):
                    if pad_extra_ones and e.dtype == torch.bool:
                        ec[i] = torch.cat((e, torch.ones(cs - real_n, *e.shape[1:], dtype=torch.bool, device=e.device)), dim=0)
                    else:
                        ec[i] = torch.cat((e, e[:1].expand(cs - real_n, *[-1] * (e.ndim - 1))), dim=0)
            parts.append(fn(xc, *ec)[:real_n])
        return torch.cat(parts, dim=0)

    def get_target_rects(
        self,
        quads: torch.Tensor,
        curr_rects: torch.Tensor,
        curr_centers: torch.Tensor,
        gt_relations: torch.Tensor,
    ):
        to_rects = curr_rects.unsqueeze(0).expand(curr_rects.shape[0], -1, -1)
        to_centers = curr_centers.unsqueeze(0).expand(quads.shape[0], -1, -1)

        k = max(0, min(to_rects.shape[1] - 1, self.k - 1))

        if k == 0:
            to_rects = torch.zeros(
                curr_rects.shape[0], 1, curr_rects.shape[1] + 2, **options(curr_rects)
            )
            closest_other_idxs = torch.zeros(
                curr_rects.shape[0], 1, dtype=torch.int64, device=curr_rects.device
            )
        else:
            all_dists = get_cdist(quads, curr_centers)

            closest_other_idxs = torch.topk(
                all_dists, k=k, dim=1, largest=False, sorted=False
            ).indices

            # K,K-1,D
            to_rects = torch.gather(
                to_rects,
                dim=1,
                index=closest_other_idxs.unsqueeze(2).expand(-1, -1, curr_rects.shape[1]),
            )
            # K,K-1
            all_dists = torch.gather(all_dists, dim=1, index=closest_other_idxs)
            # K,K-1,2
            to_centers = torch.gather(
                to_centers, dim=1, index=closest_other_idxs.unsqueeze(2).expand(-1, -1, 2)
            )

            # Add the null column to rects
            to_rects = torch.cat(
                (
                    torch.zeros(to_rects.shape[0], 1, to_rects.shape[2], **options(to_rects)),
                    to_rects,
                ),
                dim=1,
            )

            # Add the null column to the dists
            all_dists = torch.cat(
                (
                    torch.full((all_dists.shape[0], 1), -1, **options(all_dists)),
                    all_dists,
                ),
                dim=1,
            )

            directions = get_directions(quads, to_centers)
            directions = torch.cat(
                (
                    torch.full((directions.shape[0], 1), -2, **options(directions)),
                    directions,
                ),
                dim=1,
            )

            # Add the pairwise geometric encodings
            to_rects = torch.cat(
                (
                    to_rects,
                    all_dists.unsqueeze(2),
                    directions.unsqueeze(2),
                ),
                dim=2,
            )

            # Add the null column
            closest_other_idxs = torch.cat(
                [
                    torch.zeros(closest_other_idxs.shape[0], 1, **options(closest_other_idxs)),
                    closest_other_idxs + 1,
                ],
                dim=1,
            )

        return to_rects, closest_other_idxs

    def prohibit_self_connection(self, dots: torch.Tensor, closest_other_idxs: torch.Tensor = None):
        dots = dots.float()

        if closest_other_idxs is None:
            neg_inf = torch.full((dots.shape[-2],), NULL_CONNECTION_WEIGHT, **options(dots)).diag()

            neg_inf = torch.cat(
                (torch.zeros(neg_inf.shape[0], 1, **options(neg_inf)), neg_inf), dim=1
            )

            if dots.ndim == 3:
                neg_inf.unsqueeze_(0)

            dots = dots + neg_inf
        else:
            neg_inf = torch.full(
                (*dots.shape[:-2], dots.shape[-2], dots.shape[-2] + 1),
                NULL_CONNECTION_WEIGHT,
                **options(dots),
            )

            if dots.ndim == 3:
                closest_other_idxs = closest_other_idxs.unsqueeze(0).expand_as(dots)
            elif dots.ndim == 4:
                closest_other_idxs = closest_other_idxs.unsqueeze(1).expand_as(dots)

            neg_inf = torch.scatter(neg_inf, dim=-1, index=closest_other_idxs, src=dots)
            dots = neg_inf

        return dots

    def get_input_encoding(
        self,
        rectified_quads: torch.Tensor,
        original_quads: torch.Tensor,
        region_counts: torch.Tensor,
        recog_features: torch.Tensor,
    ):
        cs_rg = torch.cumsum(region_counts, 0)
        cs_rg = torch.cat([torch.zeros(1, dtype=cs_rg.dtype, device=cs_rg.device), cs_rg])
        ex_offsets = cs_rg

        g_height, g_width = self.quad_rectify_grid_size
        if self.isotropic:
            # Figure out the number of valid positions for each quad, which we can use to compute the mean
            quad_widths = quad_rectify_calc_quad_width(
                original_quads, rectified_quads.shape[-2], 1, rectified_quads.shape[-1]
            )
        else:
            quad_widths = torch.full((original_quads.shape[0],), g_width, **options(original_quads))
        num_valid_pos = (quad_widths * g_height).clamp_min(1)

        # Ensure that these values aren't very large
        original_quads = original_quads / self.quad_downscale
        mid_pts = original_quads.detach().mean(dim=1, dtype=torch.float32)

        def _input_enc_nn(rq, rf, nvp):
            rq = self.rect_proj(rq)
            avg = rq.flatten(2).sum(dim=2, dtype=torch.float32) / nvp.unsqueeze(1)
            rec = self.recog_tx(rf.detach()).mean(dim=1, dtype=torch.float32)
            return self.combined_proj(torch.cat((avg, rec), dim=1))

        semantic_encoding = self._chunked_forward(
            _input_enc_nn, rectified_quads, recog_features, num_valid_pos,
        )

        h1 = original_quads[:, 3] - original_quads[:, 0]
        h2 = original_quads[:, 2] - original_quads[:, 1]

        mp1 = original_quads[:, 0] + (h1 / 2)
        mp2 = original_quads[:, 1] + (h2 / 2)

        d1 = mp2 - mp1

        wdth = d1.norm(dim=1, keepdim=True)

        d1 = d1 / wdth.clamp_min(1e-6)

        hts = ((h1 + h2) / 2).norm(dim=1, keepdim=True)

        d2 = torch.stack([-d1[:, 1], d1[:, 0]], dim=-1)

        # Prevent overfitting to specific quad positions by translating all positions
        # by some random offset (thus preserving inter-quad relationships, but not absolute positions)
        if self.training:
            rand_quad_offset = torch.rand(1, 1, 2, **options(original_quads)) * 4 - 2

            quads_enc = original_quads + rand_quad_offset
        else:
            quads_enc = original_quads

        # The last 5 tensors represent the geometric encoding for each word

        full_encoding = torch.cat(
            (semantic_encoding, quads_enc.flatten(1), d1, d2, wdth, hts), dim=1
        )

        return full_encoding, ex_offsets, region_counts, mid_pts

    def forward(
        self,
        rectified_quads: torch.Tensor,
        original_quads: torch.Tensor,
        region_counts: torch.Tensor,
        recog_features: torch.Tensor,
    ):
        rectified_quads = rectified_quads.float()
        recog_features = recog_features.float()

        assert torch.all(torch.isfinite(rectified_quads))
        assert torch.all(torch.isfinite(recog_features))

        proj_rects, _, region_counts, mid_pts = self.get_input_encoding(
            rectified_quads, original_quads, region_counts, recog_features
        )

        quads = original_quads / self.quad_downscale

        if not self.inference_mode:
            assert torch.all(torch.isfinite(proj_rects)), "Not all proj_rects were finite!"

        counts_list = region_counts.tolist() if region_counts.dim() > 0 else [int(region_counts.item())]
        batch_size = len(counts_list)
        device = proj_rects.device
        dtype = proj_rects.dtype
        feat_dim = proj_rects.shape[1]
        z = self.k - 1
        seq_len = z + 1

        if max(counts_list, default=0) == 0:
            return {
                "words": [torch.empty(0, 1, dtype=dtype, device=device) for _ in range(batch_size)],
                "lines": [torch.empty(0, 1, dtype=dtype, device=device) for _ in range(batch_size)],
                "line_log_var_unc": [
                    torch.empty(0, 1, dtype=dtype, device=device) for _ in range(batch_size)
                ],
            }

        # Per-image cdist + topk, then concatenate into flat [N_total, seq_len, ...]
        offsets = [0]
        for c in counts_list:
            offsets.append(offsets[-1] + c)
        n_total = offsets[-1]

        enc_input_flat = torch.zeros(n_total, seq_len, 2 * feat_dim + 2, dtype=dtype, device=device)
        mask_flat = torch.ones(n_total, seq_len, dtype=torch.bool, device=device)
        closest_flat = torch.zeros(n_total, seq_len, dtype=torch.long, device=device)

        for i, n_i in enumerate(counts_list):
            if n_i == 0:
                continue
            s, e = offsets[i], offsets[i + 1]
            rects_i = proj_rects[s:e]
            centers_i = mid_pts[s:e]
            quads_i = quads[s:e]
            z_i = min(n_i - 1, z)

            from_r = rects_i.unsqueeze(1).expand(-1, seq_len, -1)
            enc_input_flat[s:e, 0, :feat_dim] = rects_i
            enc_input_flat[s:e, 0, 2 * feat_dim] = -1
            enc_input_flat[s:e, 0, 2 * feat_dim + 1] = -2
            mask_flat[s:e, 0] = False

            if z_i > 0:
                dists_i = get_cdist(quads_i, centers_i)
                topk_d, topk_idx = torch.topk(dists_i, k=z_i, dim=1, largest=False, sorted=False)
                nb_r = torch.gather(rects_i.unsqueeze(0).expand(n_i, -1, -1), 1, topk_idx.unsqueeze(2).expand(-1, -1, feat_dim))
                nb_c = torch.gather(centers_i.unsqueeze(0).expand(n_i, -1, -1), 1, topk_idx.unsqueeze(2).expand(-1, -1, 2))
                dirs_i = get_directions(quads_i, nb_c)

                enc_input_flat[s:e, 1:z_i + 1, :feat_dim] = from_r[:, 1:z_i + 1]
                enc_input_flat[s:e, 1:z_i + 1, feat_dim:2 * feat_dim] = nb_r
                enc_input_flat[s:e, 1:z_i + 1, 2 * feat_dim] = topk_d
                enc_input_flat[s:e, 1:z_i + 1, 2 * feat_dim + 1] = dirs_i
                mask_flat[s:e, 1:z_i + 1] = False
                closest_flat[s:e, 1:z_i + 1] = topk_idx + 1

        # Chunked encoder on flat regions — always sees [chunk_size, seq_len, dim]
        def _run_encoder(enc, mask):
            out = self.encoder[0](enc, src_key_padding_mask=mask)
            return self.encoder[1](out)

        dots_flat = self._chunked_forward(_run_encoder, enc_input_flat, mask_flat, pad_extra_ones=True)

        # Per-image: scatter encoder output into full relation matrices
        all_dots = dict(words=[], lines=[], line_log_var_unc=[])
        for i, n_i in enumerate(counts_list):
            if n_i == 0:
                all_dots["words"].append(torch.empty(0, 1, dtype=torch.float32, device=device))
                all_dots["lines"].append(torch.empty(0, 1, dtype=torch.float32, device=device))
                all_dots["line_log_var_unc"].append(torch.empty(0, 1, dtype=torch.float32, device=device))
                continue
            s, e = offsets[i], offsets[i + 1]
            dots_i = dots_flat[s:e].unsqueeze(0).permute(0, 3, 1, 2)
            cidx_i = closest_flat[s:e].unsqueeze(0)
            dots_i = self.prohibit_self_connection(dots_i, cidx_i)
            all_dots["words"].append(dots_i[0, 0, :n_i, :n_i + 1])
            all_dots["lines"].append(dots_i[0, 1, :n_i, :n_i + 1])
            all_dots["line_log_var_unc"].append(dots_i[0, 2, :n_i, :n_i + 1])

        return {
            "words": all_dots["words"],
            "lines": all_dots["lines"],
            "line_log_var_unc": all_dots["line_log_var_unc"],
        }


def get_cdist(
    quads: torch.Tensor, centers: torch.Tensor, x_factor: float = 1.0, y_factor: float = 1.0
):
    region_counts = torch.tensor([quads.shape[0]], dtype=torch.int64, device=quads.device)

    ret = ragged_quad_all_2_all_distance_v2(
        quads.unsqueeze(0), region_counts, x_factor, y_factor, allow_self_distance=False
    )[0]

    return ret


def get_cdist_batched(
    quads: torch.Tensor, region_counts: torch.Tensor, x_factor: float = 1.0, y_factor: float = 1.0
):
    return ragged_quad_all_2_all_distance_v2(
        quads.contiguous(),
        region_counts.to(device=quads.device, dtype=torch.int64),
        x_factor,
        y_factor,
        allow_self_distance=False,
    )


def get_directions(quads: torch.Tensor, to_centers: torch.Tensor):
    quads = quads.detach()
    to_centers = to_centers.detach()

    # quads: N,4,2
    # to_centers: N,K,2

    pt0 = (quads[:, 0] + quads[:, 3]) / 2
    pt1 = (quads[:, 1] + quads[:, 2]) / 2

    direction = pt1 - pt0
    direction /= direction.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
    direction = direction.unsqueeze(1).expand(-1, to_centers.shape[1], -1)

    centers = (pt0 + pt1) / 2

    vec_other = to_centers - centers.unsqueeze(1)
    dir_other = vec_other / vec_other.norm(p=2, dim=2, keepdim=True).clamp_min(1e-6)

    cos_angle = torch.einsum("ftv,ftv->ft", direction, dir_other)

    return cos_angle
