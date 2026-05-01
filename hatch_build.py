"""Hatchling build hook: tag wheels as platform-specific when GPU `.so`
artifacts are bundled, so pip won't install a `py3-none-any` wheel onto a
machine whose Python ABI / OS / arch can't actually load the extension.

If no `.so` is found we leave the wheel pure (so a CPU-only sdist build
still produces a usable cross-platform wheel)."""

from __future__ import annotations

import glob
import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        gpu_so_globs = [
            "src/xtrace_sdk/x_vec/crypto/paillier_gpu_ext/*.so",
            "src/xtrace_sdk/x_vec/crypto/paillier_lookup_gpu_ext/*.so",
        ]
        has_binary = any(
            glob.glob(os.path.join(self.root, pattern)) for pattern in gpu_so_globs
        )
        if has_binary:
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
