"""Shared device-resolution helper for clients with optional GPU backends.

Used by ``paillier_client`` and ``paillier_lookup_client``, and intended to be
reused by future homomorphic clients that ship a CUDA extension.
"""

import logging
import os
import shutil
from collections.abc import Callable
from typing import Any, Literal

DeviceMode = Literal["auto", "cpu", "gpu"]

_log = logging.getLogger(__name__)


def _gpu_expected() -> bool:
    return (
        os.path.exists("/dev/nvidia0")
        or shutil.which("nvidia-smi") is not None
        or "CUDA_VISIBLE_DEVICES" in os.environ
    )


def resolve_device(
    requested: DeviceMode,
    gpu_loader: Callable[[], type[Any]],
) -> tuple[str, type[Any] | None]:
    """Resolve the compute backend for a homomorphic client.

    - ``"cpu"``: never probes; returns CPU.
    - ``"gpu"``: loads and probes the extension; raises if it fails.
    - ``"auto"``: probes; on failure falls back to CPU. If a GPU is
      detected on the host (``/dev/nvidia0``, ``nvidia-smi``,
      ``CUDA_VISIBLE_DEVICES``) but the probe fails, logs a warning so
      misconfigured GPU hosts don't silently run on CPU.

    :param requested: User-supplied device mode.
    :param gpu_loader: Callable returning the GPU client class. Typically
        wraps an import of the extension module so that import errors
        propagate as load failures.
    :return: ``(device, gpu_cls_or_None)``. ``gpu_cls`` is ``None`` for CPU.
    """
    if requested == "cpu":
        return "cpu", None

    try:
        gpu_cls = gpu_loader()
        gpu_cls(skip_key_gen=True)
    except Exception as exc:
        if requested == "gpu":
            raise
        if _gpu_expected():
            _log.warning(
                "GPU detected on host but extension failed to load (%s); falling back to CPU.",
                exc,
            )
        return "cpu", None

    return "gpu", gpu_cls
