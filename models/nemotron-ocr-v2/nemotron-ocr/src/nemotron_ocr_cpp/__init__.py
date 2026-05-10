# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

# Ensure torch is imported so its shared libs (e.g., libc10.so) are available
try:
    import torch  # noqa: F401
except Exception:  # torch may be cpu-only/cuda-only, still attempt import of extension
    pass

from ._nemotron_ocr_cpp import *  # noqa: F403