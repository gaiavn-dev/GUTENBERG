# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for iterable data containers (TextRegion, Example, Batch)."""


class DataContainer(object):
    def __iter__(self):
        raise NotImplementedError("Subclasses must implement this!")
