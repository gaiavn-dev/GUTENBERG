#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import platform
import sys
import shutil
from glob import glob

from pathlib import Path

def _parse_arch_list(arch_list: str) -> list[str]:
    """Parse TORCH_CUDA_ARCH_LIST-formatted string into nvcc -gencode flags.

    Accepts tokens like "7.5", "8.6", "8.6+PTX", "86", "90+PTX" separated by
    spaces, semicolons, or commas.
    """
    tokens = arch_list.replace(";", " ").replace(",", " ").split()
    flags: list[str] = []
    for token in tokens:
        with_ptx = token.endswith("+PTX")
        clean = token[:-4] if with_ptx else token
        clean = clean.replace(".", "")
        if not clean.isdigit():
            continue
        mm = clean
        flags.append(f"-gencode=arch=compute_{mm},code=sm_{mm}")
        if with_ptx:
            flags.append(f"-gencode=arch=compute_{mm},code=compute_{mm}")
    return flags


def _detect_arch_list() -> list[str]:
    """Best-effort detection of a single architecture if a GPU is visible.

    Falls back to a conservative list (sm_75 and sm_86) when detection isn't possible
    (common during Docker image build).
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            mm = f"{major}{minor}"
            return [f"-gencode=arch=compute_{mm},code=sm_{mm}"]
    except Exception:
        pass

    # Fallback: reasonably modern GPUs supported by CUDA 12.x toolchains
    return [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_86,code=sm_86",
    ]


# Decide architectures in this order of precedence:
# 1) Respect TORCH_CUDA_ARCH_LIST if provided
# 2) Detect from the visible GPU at runtime
# 3) Fallback to a safe default list
arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
cuda_architectures = _parse_arch_list(arch_env) if arch_env else _detect_arch_list()

includes = [
    "-I/usr/local/cuda/include",
    f"-I{os.getcwd()}/cpp/trove",
]

libs = []

common_args = ["-std=c++17"]
gcc_args = ["-fopenmp", "-DOMP_NESTED=true"]
cuda_args = []

_target_arch = os.environ.get("ARCH", "").lower() or platform.machine().lower()
if _target_arch in ("x86_64", "amd64"):
    gcc_args += ["-mavx2"]

files = sorted(glob("cpp/**/*.cu", recursive=True) + glob("cpp/**/*.cpp", recursive=True))
files = [f for f in files if "trove/tests" not in f]

# debug=True
debug = False
if debug:
    common_args += ["-g", "-O0"]
    cuda_args += ["-G"]
else:
    common_args += ["-O3", "-DNDEBUG"]

compile_args = {
    "cxx": common_args + gcc_args + includes,
    "nvcc": common_args + cuda_args + includes + cuda_architectures,
}


def build() -> None:
    if os.environ.get("BUILD_CPP_EXTENSION", "1") != "1":
        print("Environment variable BUILD_CPP_EXTENSION=1 not set. Skipping build.")
        sys.exit(0)

    from setuptools import Distribution
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    ext_modules = [
        CUDAExtension(
            "_nemotron_ocr_cpp",
            files,
            extra_compile_args=compile_args,
            libraries=libs,
        )
    ]

    distribution = Distribution({"name": "nemotron_ocr_cpp", "ext_modules": ext_modules})

    build_ext = BuildExtension.with_options(parallel=False)
    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        output = Path(output)
        relative_extension = Path("src/nemotron_ocr_cpp") / output.relative_to(cmd.build_lib)

        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == "__main__":
    build()
