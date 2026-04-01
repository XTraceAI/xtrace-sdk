from __future__ import annotations
import os
import json
import typer
from typing import Dict, Any
from rich.console import Console
from rich.rule import Rule
from contextlib import contextmanager
from datetime import datetime, timezone, UTC
import asyncio
from .._utils._prompt import secret_prompt
from .._utils._getHash import resolve_api_key_hash
from .._utils._errors import extract_server_error
from ...state import get_admin_key_cached, get_integration
from .._utils._admin_key import resolve_api_key_override
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

def _pretty_ts(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
        return dt.strftime("%b %d, %Y %H:%M UTC")
    except Exception:
        return ts or ""

_PERM_NAME_TO_NUM = {"read": 1, "write": 3, "delete": 7, "none": None}
_PERM_COLOR = {"read": "cyan", "write": "green", "delete": "magenta", "none": "red"}

@app.command("create", help="Create a new knowledge base.")
def create_kb(
    name: str = typer.Argument(..., help="Name of the knowledge base"),
    description: str = typer.Option("", "--description", "-d", help="Optional description (wrap in quotes)"),
    permission: str = typer.Option(
        "write", "--permission", "-p",
        help="Permission to grant to the current (or specified) API key for this KB: read | write | delete | none (default: write)",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON from the API"),
    api_key_override: str | None = typer.Option(
        None, "-a", "--api_key",
        help="Grant permission to this API key instead of the one in .env"
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
    target_api_key = resolve_api_key_override(api_key_override, env, console=console)
    api_url = (os.getenv("XTRACE_API_URL") or "https://api.production.xtrace.ai").rstrip("/")

    admin_key = get_admin_key_cached(json_out=json_out, human_prompt_fn=secret_prompt)

    if not json_out:
        console.print(Rule(style="dim"))

    # validate permission option
    perm_name = (permission or "").lower()
    if perm_name not in _PERM_NAME_TO_NUM:
        raise typer.BadParameter("--permission must be one of: read, write, delete, none")
    perm_num = _PERM_NAME_TO_NUM[perm_name]

    # use shared integration if available, else create one for this command
    _shared = get_integration()
    if _shared is not None:
        if not _shared.admin_key and admin_key:
            _shared.admin_key = admin_key
        integration = _shared
        _owned = False
    else:
        from xtrace_sdk.integrations.xtrace import XTraceIntegration
        integration = XTraceIntegration(
            org_id=org_id, api_key=target_api_key or "", api_url=api_url, admin_key=admin_key
        )
        _owned = True

    try:
        # create KB
        with _maybe_status("Creating knowledge base…", enable=not json_out):
            try:
                data = asyncio.run(integration.create_kb(name, description))
            except Exception as e:
                msg, code = extract_server_error(e)
                if json_out:
                    typer.echo(json.dumps({"error": {"message": msg or str(e), "status": code}}, ensure_ascii=False))
                else:
                    console.print(f"[red]Create failed{f' ({code})' if code is not None else ''}:[/] {msg or e}")
                raise typer.Exit(1)

        kb_id = str(data.get("id", ""))
        kb_name = str(data.get("name", name))
        kb_desc = str(data.get("description", description))

        # grant permission for the selected API key (both JSON and human paths)
        perm_granted: str | None = None
        perm_error: str | None = None

        if perm_num is not None:
            with _maybe_status("Resolving API key hash…", enable=not json_out):
                assert target_api_key is not None
                api_key_hash = resolve_api_key_hash(
                    admin_key=admin_key, org_id=org_id, api_key_to_match=target_api_key,
                    integration=integration,
                )

            if not api_key_hash:
                perm_error = "Could not resolve API key hash; permission was not granted."
            else:
                with _maybe_status(f"Granting [bold]{perm_name}[/] permission…", enable=not json_out):
                    try:
                        asyncio.run(integration.set_key_permission(api_key_hash, kb_id, int(perm_num)))
                        perm_granted = perm_name
                    except Exception as e:
                        msg, code = extract_server_error(e)
                        perm_error = f"grant failed{f' ({code})' if code is not None else ''}: {msg or e}"

        if json_out:
            if perm_granted:
                data["permission"] = perm_granted
            elif perm_num is None:
                data["permission"] = "none"
            if perm_error:
                data["permissionError"] = perm_error
            typer.echo(json.dumps(data, ensure_ascii=False))
            return

        # human-readable output
        console.print(f"[green]Successfully created KB [/][bold green]{kb_name}[/] (ID: [blue]{kb_id}[/])")
        if kb_desc:
            console.print(f"[dim]Description[/]: {kb_desc}")

        if perm_num is None:
            console.print("[dim]Permission[/]: [red]none[/] (no grant applied)")
        elif perm_error:
            console.print(f"[yellow]Warning:[/] {perm_error}")
        elif perm_granted:
            color = _PERM_COLOR.get(perm_name, "white")
            key_preview = (target_api_key or "")[:5] + "…"
            console.print(f"[dim]Permission[/]: [bold {color}]{perm_name}[/] (granted to key [cyan]{key_preview}[/])")
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass
