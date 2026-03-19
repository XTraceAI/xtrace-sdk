from __future__ import annotations
import json
import re
import ast
from typing import Optional, Tuple, Any, Dict

_JSON_BLOCK = re.compile(r"(\{.*\})", re.DOTALL)
_CODE_RE = re.compile(r"Error code:\s*(\d+)", re.IGNORECASE)

def _try_json_or_literal(txt: str | bytes | None) -> Dict[str, Any] | None:
    if not txt:
        return None
    if isinstance(txt, bytes):
        try:
            txt = txt.decode("utf-8", "ignore")
        except Exception:
            return None
    # first try proper JSON
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # then try Python literal dict
    try:
        obj = ast.literal_eval(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _extract_from_detail(detail: Dict[str, Any]) -> Tuple[str| None, Any| None]:
    status = detail.get("statusCode")
    body = detail.get("body")

    if isinstance(body, (str, bytes)):
        inner = _try_json_or_literal(body)
        if isinstance(inner, dict):
            if "response" in inner:
                return str(inner.get("response")), int(status) if isinstance(status, int) else None
            err = inner.get("error")
            if isinstance(err, dict) and "message" in err:
                return str(err.get("message")), int(status) if isinstance(status, int) else None
            if "message" in inner:
                return str(inner.get("message")), int(status) if isinstance(status, int) else None
        # body might be plain string
        return (body.decode() if isinstance(body, bytes) else str(body)), int(status) if isinstance(status, int) else None

    # sometimes message/response directly in detail
    if "response" in detail:
        return str(detail.get("response")), int(status) if isinstance(status, int) else None
    if "message" in detail:
        return str(detail.get("message")), int(status) if isinstance(status, int) else None

    return None, int(status) if isinstance(status, int) else None

def _extract_from_top(obj: Dict[str, Any]) -> Tuple[str| None, int| None]:
    err = obj.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("error") or err.get("description")
        code = err.get("code")
        return (str(msg) if msg else None, int(code) if isinstance(code, int) else None)
    # generic top-level
    msg = obj.get("response") or obj.get("message")
    return (str(msg) if msg else None, None)

def extract_server_error(exc: Exception) -> tuple[str | None, int | None]:
    """
    Extract a concise message and HTTP-like status code from various
    SDK/HTTP exception shapes. Returns (message, status_code) or (None, None).
    """
    # common attributes first
    for attr in ("detail",):
        try:
            val = getattr(exc, attr)
            if isinstance(val, dict):
                msg, code = _extract_from_detail(val)
                if msg:
                    return msg, code
        except Exception:
            pass

    # sometimes there is a response object
    for attr in ("response",):
        try:
            resp = getattr(exc, attr)
            status = getattr(resp, "status_code", None)
            text = None
            if hasattr(resp, "text"):
                text = resp.text
            elif hasattr(resp, "content"):
                text = resp.content
            obj = _try_json_or_literal(text)
            if isinstance(obj, dict):
                msg, code = _extract_from_top(obj)
                if msg:
                    return msg, int(status) if isinstance(status, int) else code
        except Exception:
            pass

    # parse exception string
    s = str(exc)
    code = None
    m = _CODE_RE.search(s)
    if m:
        try:
            code = int(m.group(1))
        except Exception:
            pass

    # direct parse of the whole string
    obj = _try_json_or_literal(s)
    if isinstance(obj, dict):
        # detail-based shape
        detail = obj.get("detail")
        if isinstance(detail, dict):
            msg, dcode = _extract_from_detail(detail)
            if msg:
                return msg, dcode if dcode is not None else code
        # top-level error/message
        msg, tcode = _extract_from_top(obj)
        if msg:
            return msg, tcode if tcode is not None else code

    # heuristic: extract last JSON-like block
    blocks = _JSON_BLOCK.findall(s)
    for candidate in reversed(blocks):
        obj = _try_json_or_literal(candidate)
        if not isinstance(obj, dict):
            continue
        detail = obj.get("detail")
        if isinstance(detail, dict):
            msg, dcode = _extract_from_detail(detail)
            if msg:
                return msg, dcode if dcode is not None else code
        msg, tcode = _extract_from_top(obj)
        if msg:
            return msg, tcode if tcode is not None else code

    return None, code
