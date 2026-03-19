from __future__ import annotations
import asyncio
from typing import Any, Optional


def resolve_api_key_hash(
    *,
    admin_key: str,
    org_id: str,
    api_key_to_match: str,
    integration: Any = None,
) -> str | None:
    """
    Call /keys using ADMIN credentials, then find the entry whose `firstN`
    matches the first 5 chars of the API key (no xtr_ prefix).
    Returns the key's `hash`, or None if not found.

    If *integration* is provided it is used (reusing its session); otherwise
    a fresh aiohttp session is created via the _aio_http fallback.
    """
    try:
        if integration is not None:
            keys = asyncio.run(integration.list_api_keys())
        else:
            from xtrace_sdk.cli.commands._utils._aio_http import get
            headers = {
                "content-type": "application/json",
                "xtrace-admin-key": admin_key,
                "xtrace-org-id": org_id,
            }
            r = get("https://api.production.xtrace.ai/api/v1/keys", headers=headers)
            if not r.ok:
                return None
            keys = r.json() or []

        if not keys:
            return None
        if len(keys) == 1:
            return keys[0].get("hash")

        first5 = (api_key_to_match or "")[:5]
        first5_lower = first5.lower()

        for k in keys:
            fn = (k.get("firstN") or "")
            if fn == first5 or fn.lower() == first5_lower:
                return k.get("hash")
    except Exception:
        pass
    return None
