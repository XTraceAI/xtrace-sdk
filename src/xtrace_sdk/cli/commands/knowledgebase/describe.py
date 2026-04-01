from __future__ import annotations
import os
import json
import typer
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from contextlib import contextmanager
from datetime import datetime, timezone, UTC
from .._utils._prompt import secret_prompt
from .._utils._getHash import resolve_api_key_hash
import asyncio
from .._utils._admin_key import get_admin_key, resolve_api_key_override
from ...state import get_admin_key_cached, get_integration

from typing import Annotated
from typing import Iterator


app = typer.Typer()
console = Console()

@contextmanager
def safe_status(message: str) -> Iterator[None]:
    """Rich status spinner that also works on older Rich versions."""
    try:
        with console.status(message, spinner="dots"):
            yield
    except TypeError:
        console.print(f"[dim]{message}[/]")
        yield

@contextmanager
def _maybe_status(message: str, enable: bool) -> Iterator[None]:
    """Disable status output when enable=False (e.g., --json)."""
    if enable:
        with safe_status(message):
            yield
    else:
        yield

def _require_env(names: list[str]) -> dict[str, str]:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        console.print(
            "[red]Missing required environment variables:[/] "
            + ", ".join(missing)
            + "\n[dim]Tip: run `xtrace init` first to create .env[/]"
        )
        raise typer.Exit(2)
    return {n: os.environ[n] for n in names}

def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def _pretty_ts(ts: str | None) -> str:
    dt = _parse_ts(ts)
    if not dt:
        return ""
    dt = dt.astimezone(UTC)
    return dt.strftime("%b %d, %Y %H:%M UTC")

_PERM_LABEL = {1: "read", 3: "write", 7: "delete"}

@app.command("describe", help="Describe one or more knowledge bases by ID.")
def describe_kb(
        kb_ids:  Annotated[
            List[str],
            typer.Argument(
                ...,
                metavar="KB_ID...",
                help="One or more knowledge base IDs"
            ),
        ],
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON objects (one per KB)"),
    api_key_override: str | None = typer.Option(
        None, "-a", "--api_key",
        help="Use this API key instead of XTRACE_API_KEY from .env (for permission display)"
    ),
)-> None:
    # load environment
    with _maybe_status("Loading environment…", enable=not json_out):
        import dotenv  # lazy
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
        required = ["XTRACE_ORG_ID"]
        if api_key_override is None:
            required.append("XTRACE_API_KEY")
        env = _require_env(required)

    org_id = env["XTRACE_ORG_ID"]
    api_key = resolve_api_key_override(api_key_override, env, console=console)
    api_url = (os.getenv("XTRACE_API_URL") or "https://api.production.xtrace.ai").rstrip("/")

    # admin key (env first; else prompt; prompt goes to STDERR in --json)
    admin_key = get_admin_key_cached(json_out=json_out, human_prompt_fn=secret_prompt)

    # use shared integration if available, else create one for this command
    _shared = get_integration()
    if _shared is not None:
        if not _shared.admin_key and admin_key:
            _shared.admin_key = admin_key
        integration = _shared
        _owned = False
    else:
        from xtrace_sdk.integrations.xtrace import XTraceIntegration
        integration = XTraceIntegration(org_id=org_id, api_key=api_key or "", api_url=api_url, admin_key=admin_key)
        _owned = True

    try:
        # resolve permissions for specified key
        perms_by_kb: Dict[str, int] = {}
        with _maybe_status("Resolving API key permissions…", enable=not json_out):
            api_key_hash = resolve_api_key_hash(
                admin_key=admin_key, org_id=org_id, api_key_to_match=api_key or "",
                integration=integration,
            )
            if api_key_hash:
                try:
                    perms = asyncio.run(integration.get_key_permissions(api_key_hash)) or []
                    perms_by_kb = {p.get("knowledgeBaseId"): int(p.get("permission", 0)) for p in perms}
                except Exception as e:
                    if not json_out:
                        console.print(f"[yellow]Warning:[/] error fetching permissions: {e}")
            else:
                if not json_out:
                    console.print("[yellow]Warning:[/] Could not resolve API key hash; permissions will show as NONE.")

        # fetch each KB (preserve input order)
        results: List[Dict[str, Any]] = []
        for kb_id in kb_ids:
            with _maybe_status(f"Fetching KB {kb_id}…", enable=not json_out):
                try:
                    data = asyncio.run(integration.get_kb(kb_id))
                    results.append({"id": kb_id, "status": "ok", "data": data or {}})
                except Exception as e:
                    msg = str(e)
                    code = None
                    try:
                        import aiohttp
                        if isinstance(e, aiohttp.ClientResponseError):
                            code = e.status
                            if code == 404:
                                results.append({"id": kb_id, "status": "not_found", "code": 404, "message": msg[:180]})
                                continue
                    except Exception:
                        pass
                    results.append({"id": kb_id, "status": "error", "code": code, "message": msg[:180]})
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass

    # JSON mode
    if json_out:
        payload: List[Dict[str, Any]] = []
        for r in results:
            if r["status"] == "ok":
                payload.append(r["data"])  # raw KB object
            else:
                payload.append({k: v for k, v in r.items() if k != "data"})  # compact error object
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return

    # Human table
    ok_rows = [r for r in results if r["status"] == "ok"]
    if not ok_rows:
        for r in results:
            if r["status"] == "not_found":
                console.print(f"[yellow]Not found:[/] {r['id']}")
            elif r["status"] == "error":
                console.print(f"[red]Error:[/] {r['id']} — {r.get('message','')}")
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("ID", style="blue", no_wrap=True)
    table.add_column("Name")
    table.add_column("Description", style="dim")
    table.add_column("# of Vectors")
    table.add_column("Permission")
    table.add_column("Updated", style="dim")
    table.add_column("Created", style="dim")

    for r in ok_rows:
        kb = r["data"]
        kb_id = str(kb.get("id", ""))
        perm_num = perms_by_kb.get(kb_id)
        perm_label = _PERM_LABEL.get(perm_num or -1, "NONE")
        perm_cell = f"[red]{perm_label}[/]" if perm_label == "NONE" else perm_label

        num_vecs = kb.get("numberOfVectors")
        try:
            num_vecs_str = f"{int(num_vecs):,}"
        except Exception:
            num_vecs_str = str(num_vecs or "")

        table.add_row(
            kb_id,
            str(kb.get("name", "")),
            str(kb.get("description", "")) if kb.get("description") else "",
            num_vecs_str,
            perm_cell,
            _pretty_ts(kb.get("updatedAt")),
            _pretty_ts(kb.get("createdAt")),
        )

    console.print(table)

    # show misses/errors under the table
    misses = [r for r in results if r["status"] != "ok"]
    if misses:
        for r in misses:
            if r["status"] == "not_found":
                console.print(f"[yellow]Not found:[/] {r['id']}")
            else:
                console.print(f"[red]Error:[/] {r['id']} — {r.get('message','')}")

    #legend
    if api_key_override:
        key_preview = (api_key or "")[:5] + "…"
        console.print(f"[dim]** Table displays permissions for API key {key_preview} **[/]")
    else:
        console.print(
            "[dim]** Table displays permissions for API key saved via[/] [bold]xtrace init[/] [dim]**[/]"
        )
