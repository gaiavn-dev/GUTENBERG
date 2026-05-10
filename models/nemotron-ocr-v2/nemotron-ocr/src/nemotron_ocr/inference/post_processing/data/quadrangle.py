# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quadrangle class."""

import logging
from typing import Union

from shapely.geometry import Polygon
import numpy
import torch

from nemotron_ocr_cpp import calc_poly_min_rrect, get_poly_bounds_quad

logger = logging.getLogger(__name__)


def apply_single_stm(vertices, stm):
    """
    Applies the single homogeneous transformation matrix.

    Args:
        vertices (torch tensor): Array of 2D vertices that form the polyline.
        stm (torch.tensor): 3x3 homogeneous matrix.
    """
    homogenous_vertices = torch.cat((vertices, torch.ones(vertices.shape[0], 1)), dim=1)
    transformed = torch.matmul(homogenous_vertices, stm)
    norm_factor = 1.0 / transformed[:, 2:]
    # Handle divide by zero case.
    norm_factor[transformed[:, 2:] == 0] = 0
    return transformed[:, :2].contiguous() * norm_factor


class AABB:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def area(self):
        return self.width * self.height

    def contains(self, x, y):
        return (
            x >= self.x and x < (self.x + self.width) and y >= self.y and y < (self.y + self.height)
        )

    def to_quad(self):
        vertices = [
            self.x,
            self.y,
            self.x + self.width,
            self.y,
            self.x + self.width,
            self.y + self.height,
            self.x,
            self.y + self.height,
        ]
        return Quadrangle(torch.tensor(vertices, dtype=torch.float32).reshape(4, 2))

    def __str__(self):
        return (
            f"[x: {self.x:0.02f}, y: {self.y:0.02f}, w: {self.width:0.02f}, h: {self.height:0.02f}]"
        )


class Quadrangle:
    def __init__(self, vertices):
        self.vertices = torch.as_tensor(vertices)

        if self.vertices.shape[-1] != 2:
            raise ValueError("The vertices must be 2-dimensional!")

        self._reset_cache()

    @staticmethod
    def from_size(height, width):
        vertices = torch.tensor(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=torch.float32
        )

        return Quadrangle(vertices)

    def _reset_cache(self):
        self._bounds = None
        self.__poly = None
        self._min_rrect = None

    def clone(self):
        return Quadrangle(vertices=self.vertices.clone())

    def apply_stm(self, stm, *args, **kwargs):
        """
        Applies the homogeneous transformation matrix.

        Args:
            stm (torch.tensor): 3x3 homogeneous matrix.
        """
        self.vertices = apply_single_stm(self.vertices, stm)
        self._reset_cache()

    def translate(self, delta_vector):
        """
        Translates all of the points by the given 2d delta.

        Args:
            delta_vector (torch.tensor): 2d translation vector.
        """
        self.vertices += delta_vector
        self._reset_cache()

    def scale(self, scale_vector, **kwargs):
        """
        Scales the points by the given 2d size vector.

        Args:
            scale_vector (torch.tensor): 2d scale vector.
                                         E.g. [2.0, 0.5] scales x by 2 and y by 0.5.
        """
        self.vertices *= scale_vector
        self._reset_cache()

    def rotate(self, rot_mat):
        self.vertices = self.vertices @ rot_mat.t()
        self._reset_cache()

    def mark_dirty(self):
        self._reset_cache()

    def shortest_edge_length(self) -> float:
        return min(self.get_magnitudes())

    @property
    def valid(self):
        bds = self.bounds
        width = (bds[1, 0] - bds[0, 0]).item()
        height = (bds[3, 1] - bds[0, 1]).item()
        return width > 0 and height > 0

    @property
    def bounds_quad(self) -> torch.Tensor:
        if self._bounds is None:
            self._bounds = get_poly_bounds_quad(self.vertices)
        return self._bounds

    @property
    def _poly(self) -> Polygon:
        if self.__poly is None:
            self.__poly = Polygon(self.vertices.numpy())
        return self.__poly

    @property
    def min_rrect(self) -> torch.Tensor:
        """Returns a rotated rect set of vertices"""
        if self._min_rrect is None:
            self._min_rrect = calc_poly_min_rrect(self.vertices)
        return self._min_rrect

    @property
    def area(self):
        return self._poly.area

    def get_intersection(self, other_quad):
        return self._poly.intersection(other_quad._poly)

    def get_union(self, other_quad):
        return self._poly.union(other_quad._poly)

    def orient(self):
        vertices = self.vertices.numpy()

        # print('--------------')
        # print('Input Vertices:\n{}'.format(vertices))

        if not is_clockwise(vertices):
            # print('Counter Clockwise')
            vertices = numpy.flip(vertices, 0)
        # else:
        #     print('Clockwise')

        # Super lazy, but top-left will be considered quite literally
        start_idx = numpy.argmin(vertices.sum(axis=1))

        # print('Start Idx: {}'.format(start_idx))

        out_verts = numpy.empty_like(vertices)
        for i in range(4):
            d_i = (start_idx + i) % 4
            out_verts[i] = vertices[d_i]

        # print('Out Vertices:\n{}\n'.format(out_verts))

        self.vertices = torch.from_numpy(out_verts)

    def get_magnitudes(self):
        return get_magnitudes(self.vertices.numpy())

    def apply(self, fn):
        fn(self)

    def validate(self):
        return torch.all(torch.isfinite(self.vertices))


def is_clockwise(vertices) -> bool:
    v = 0
    for i in range(4):
        d_i = (i + 1) % 4
        v += (vertices[d_i, 0] - vertices[i, 0]) * (vertices[d_i, 1] + vertices[i, 1])

    return v < 0


mag_inds_b = [(i + 1) % 4 for i in range(4)]


def get_magnitudes(vertices: Union[torch.Tensor, numpy.ndarray]):
    if isinstance(vertices, numpy.ndarray):
        dkwd = {"axis": -1}
        sqrt = numpy.sqrt
    else:
        dkwd = {"dim": -1}
        sqrt = torch.sqrt

    b = vertices[..., mag_inds_b, :]
    a = vertices

    d_v = (b - a) ** 2
    d_v = d_v.sum(**dkwd)
    d_v = sqrt(d_v)

    return d_v


def box_2_quad(bds: torch.Tensor):
    tl = bds[..., 0, :]
    br = bds[..., 1, :]

    p1 = tl
    p2 = torch.stack((br[..., 0], tl[..., 1]), dim=-1)
    p3 = br
    p4 = torch.stack((tl[..., 0], br[..., 1]), dim=-1)

    ret = torch.stack((p1, p2, p3, p4), dim=-2)

    return ret


def get_quad_height(vertices: Union[torch.Tensor, numpy.ndarray]):
    mags = get_magnitudes(vertices)

    return (mags[..., 1] + mags[..., 3]) / 2
