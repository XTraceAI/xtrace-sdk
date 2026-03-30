"""Lightweight benchmark context manager.

Enabled by setting the ``XTRACE_BENCHMARK`` environment variable to ``1`` or ``true``.
When disabled, ``bench()`` is a zero-cost no-op.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from time import perf_counter
from typing import Generator

_log = logging.getLogger("xtrace_sdk.bench")
_ENABLED = os.environ.get("XTRACE_BENCHMARK", "").lower() in ("1", "true")


@contextmanager
def bench(label: str) -> Generator[None, None, None]:
    """Time a block and log the result when ``XTRACE_BENCHMARK`` is enabled.

    Usage::

        with bench("query_encrypt"):
            encrypted = client.encrypt_vec_one(vec)
    """
    if not _ENABLED:
        yield
        return
    t0 = perf_counter()
    yield
    _log.info("%s: %.1fms", label, (perf_counter() - t0) * 1000)
