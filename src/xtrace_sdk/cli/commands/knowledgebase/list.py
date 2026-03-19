from __future__ import annotations
import os
import json
import typer
import asyncio
from typing import Any, Dict, List, Optional
from typing import Iterator
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from contextlib import contextmanager
from datetime import datetime, timezone, UTC
from .._utils._prompt import secret_prompt
from .._utils._getHash import resolve_api_key_hash
from ...state import get_admin_key_cached, get_integration

app = typer.Typer()
console = Console()

@contextmanager
def safe_status(message: str) -> Iterator[None]:
    try:
        with console.status(message, spinner="dots"):
            yield
    except TypeError:
        console.print(f"[dim]{message}[/]")
        yield

@contextmanager
def _maybe_status(message: str, enable: bool) -> Iterator[None]:
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

_PERM_LABEL_TXT = {1: "read", 3: "write", 7: "delete"}

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

@app.command("list-kbs", help="List knowledge bases available to your API key.")
def list_kbs(
    all: bool = typer.Option(False, "--all", help="Include KBs with no explicit permission (show 'NONE')"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON (plus numeric permissionLabel)"),
    api_key_override: str | None = typer.Option(
        None, "-a", "--api_key",
        help="Use this API key instead of XTRACE_API_KEY from .env"
    ),
)-> None:
    # load env
    with _maybe_status("Loading environment…", enable=not json_out):
        import dotenv  # lazy
        dotenv.load_dotenv()
        # org is always required; API key only when no override
        required = ["XTRACE_ORG_ID"]
        if not api_key_override:
            required.append("XTRACE_API_KEY")
        env = _require_env(required)

    org_id = env["XTRACE_ORG_ID"]
    api_key = api_key_override or env.get("XTRACE_API_KEY")
    api_url = (os.getenv("XTRACE_API_URL") or "https://api.production.xtrace.ai").rstrip("/")

    admin_key = get_admin_key_cached(json_out=json_out, human_prompt_fn=secret_prompt)

    # use shared integration if available, else create one for this command
    _shared = get_integration()
    if _shared is not None:
        integration = _shared
        _owned = False
    else:
        from xtrace_sdk.integrations.xtrace import XTraceIntegration
        integration = XTraceIntegration(org_id=org_id, api_key=api_key or "", api_url=api_url, admin_key=admin_key)
        _owned = True

    try:
        # fetch all kbs
        with _maybe_status("Fetching knowledge bases…", enable=not json_out):
            try:
                kbs: List[Dict[str, Any]] = asyncio.run(integration.list_kbs()) or []
            except Exception as e:
                if json_out:
                    typer.echo(json.dumps({"error": {"message": f"Network error (KBs): {e}", "status": None}}, ensure_ascii=False))
                else:
                    console.print(f"[red]Network error (KBs):[/] {e}")
                raise typer.Exit(1)

        # resolve permissions for specified key
        perms_by_kb: dict[str, int] = {}
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
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass

    # sort: KBs WITH permission first (by updsated desc), then WITHOUT (by updated desc)
    def sort_key(kb: Dict[str, Any])-> tuple[int, float]:
        kb_id = kb.get("id") or ""
        has_perm = 1 if perms_by_kb.get(kb_id) else 0
        dt = _parse_ts(kb.get("updatedAt"))
        ts = dt.timestamp() if dt else 0.0
        return (-has_perm, -ts)

    kbs.sort(key=sort_key)

    # filter for table view: only those with perms, unless --all
    display_kbs = kbs if all else [kb for kb in kbs if (kb.get("id") or "") in perms_by_kb]

    if json_out:
        # raw output , only modification add numeric permissionLabel (1/3/7) when available
        out: List[Dict[str, Any]] = []
        for kb in kbs:
            kb_out = dict(kb)
            perm_num = perms_by_kb.get(kb.get("id") or "")
            if perm_num in (1, 3, 7):
                kb_out["permissionLabel"] = perm_num
            out.append(kb_out)
        typer.echo(json.dumps(out, ensure_ascii=False))
        return

    if not display_kbs:
        msg = "No knowledge bases found." if all else "No knowledge bases with permissions for this API key."
        console.print(f"[yellow]{msg}[/]")
        return

    # human table
    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("ID", style="blue", no_wrap=True)
    table.add_column("Name")
    table.add_column("Description", style="dim")
    table.add_column("# of Vectors")
    table.add_column("Permission")
    table.add_column("Updated", style="dim")
    table.add_column("Created", style="dim")

    for kb in display_kbs:
        kb_id = str(kb.get("id", ""))
        perm_num = perms_by_kb.get(kb_id)
        perm_txt = _PERM_LABEL_TXT.get(perm_num or -1, "NONE")
        perm_cell = f"[red]{perm_txt}[/]" if perm_txt == "NONE" else perm_txt

        table.add_row(
            kb_id,
            str(kb.get("name", "")),
            str(kb.get("description", "")) if kb.get("description") else "",
            str(kb.get("numberOfVectors", "")),
            perm_cell,
            _pretty_ts(kb.get("updatedAt")),
            _pretty_ts(kb.get("createdAt")),
        )

    console.print(f"[dim]{len(display_kbs)} knowledge base(s).[/]")
    console.print(table)

    if api_key_override: 
        key_preview = (api_key or "")[:5] + "…"
        console.print(
            f"[dim]** Table displays permissions for API key {key_preview} **[/]"
        )
    else: 
        console.print(
            "[dim]** Table displays permissions for API key saved via[/] [bold]xtrace init[/] [dim]**[/]"
        )
