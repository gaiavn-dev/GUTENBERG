# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _extension_up_to_date(project_root: Path) -> bool:
    """Return True if a built .so exists and is newer than all sources.

    Respects the following directories:
      - src/nemotron_ocr_cpp (Python shim and built .so location)
      - cpp/ (C++/CUDA sources)
      - scripts/ (build script)
    """
    extension_dir = project_root / "src" / "nemotron_ocr_cpp"
    candidates = list(extension_dir.glob("_nemotron_ocr_cpp*.so"))
    if not candidates:
        return False

    newest_so_mtime = max(p.stat().st_mtime for p in candidates)

    newest_src_mtime = 0.0
    for directory in (project_root / "cpp", project_root / "scripts", extension_dir):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".cu", ".cpp", ".cuh", ".h", ".py"}:
                mtime = path.stat().st_mtime
                if mtime > newest_src_mtime:
                    newest_src_mtime = mtime

    return newest_so_mtime >= newest_src_mtime


def _get_platform_tag() -> str:
    """Return a PEP 425 platform tag for the current system."""
    import sysconfig
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return platform


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict) -> None:
        project_root = Path(__file__).parent
        script_path = project_root / "scripts" / "build-extension.py"

        env = os.environ.copy()
        env.setdefault("BUILD_CPP_EXTENSION", "1")

        force_rebuild = env.get("BUILD_CPP_FORCE", "0") == "1"
        build_enabled = env.get("BUILD_CPP_EXTENSION", "1") == "1"

        if not build_enabled:
            return

        if not force_rebuild and _extension_up_to_date(project_root):
            pass  # skip rebuild, but still set the tag below
        else:
            subprocess.run(
                [
                    os.fspath(sys.executable),
                    os.fspath(script_path),
                ],
                cwd=os.fspath(project_root),
                env=env,
                check=True,
            )

        # Tag the wheel as platform-specific so the .so is usable
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        abi_tag = python_tag
        platform_tag = _get_platform_tag()
        build_data["tag"] = f"{python_tag}-{abi_tag}-{platform_tag}"
        build_data["pure_python"] = False


