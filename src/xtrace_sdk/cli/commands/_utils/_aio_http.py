"""Thin synchronous wrapper around aiohttp for CLI commands.

Replaces ``requests`` so that the core SDK only depends on ``aiohttp``.
Each function blocks via ``asyncio.run()`` — safe in sync typer callbacks.
"""
from __future__ import annotations

import asyncio
import json as _json_mod
import ssl
from typing import Any

import aiohttp
import certifi


class _Response:
    """Minimal requests.Response-compatible object."""

    def __init__(self, status: int, text: str, data: Any) -> None:
        self.status_code = status
        self.ok = status < 400
        self.text = text
        self._data = data

    def json(self) -> Any:
        return self._data


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: Any = None,
    timeout: int = 20,
) -> _Response:
    async def _run() -> _Response:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with aiohttp.ClientSession() as session, session.request(
            method,
            url,
            headers=headers,
            json=json,
            timeout=aiohttp.ClientTimeout(total=timeout),
            ssl=ssl_ctx,
        ) as resp:
            text = await resp.text()
            try:
                data = _json_mod.loads(text)
            except Exception:
                data = None
            return _Response(resp.status, text, data)

    return asyncio.run(_run())


def get(url: str, *, headers: dict[str, str], timeout: int = 20) -> _Response:
    return _request("GET", url, headers=headers, timeout=timeout)


def post(url: str, *, headers: dict[str, str], json: Any = None, timeout: int = 20) -> _Response:
    return _request("POST", url, headers=headers, json=json, timeout=timeout)


def put(url: str, *, headers: dict[str, str], json: Any = None, timeout: int = 20) -> _Response:
    return _request("PUT", url, headers=headers, json=json, timeout=timeout)


def delete(url: str, *, headers: dict[str, str], timeout: int = 20) -> _Response:
    return _request("DELETE", url, headers=headers, timeout=timeout)
