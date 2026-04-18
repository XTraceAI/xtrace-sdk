import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


_CRYPTO_DIR = Path(__file__).resolve().parent
_EXTENSION_GLOBS = ("*.so", "*.pyd", "*.dylib")


@lru_cache(maxsize=None)
def load_gpu_extension(module_name: str, relative_dir: str) -> ModuleType:
    """Load a pybind extension from one of the in-tree GPU build directories."""
    search_dir = _CRYPTO_DIR / relative_dir
    candidates: list[Path] = []
    for pattern in _EXTENSION_GLOBS:
        candidates.extend(sorted(search_dir.glob(f"{module_name}{pattern[1:]}")))
        candidates.extend(sorted(search_dir.glob(f"{module_name}*{pattern[1:]}")))

    if not candidates:
        raise ImportError(
            f"Unable to locate GPU extension {module_name!r} under {search_dir}. "
            "To build it, run `./build_gpu_binaries.sh` from the repo root "
            "(requires Docker). Runtime requires NVIDIA driver >= 550 (CUDA 12.x)."
        )

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {module_path}.")

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module
