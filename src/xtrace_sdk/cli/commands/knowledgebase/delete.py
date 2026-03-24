from __future__ import annotations
import json
import os
import asyncio
import typer
from typing import List, Dict, Any, Iterator
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from contextlib import contextmanager
from .._utils._prompt import secret_prompt
from ...state import get_integration, get_admin_key_cached
from typing import Annotated

app = typer.Typer()
console = Console()

@contextmanager
def safe_status(message: str)-> Iterator[None]:
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

@app.command("delete-kb", help="Delete one or more knowledge bases by ID.")
def delete_kb(
    kb_ids: Annotated[
        List[str],
        typer.Argument(
            ...,
            metavar="KB_ID...",
            help="One or more knowledge base IDs to delete",
        ),
    ],
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON results"),
)-> None:
    with _maybe_status("Loading environment…", enable=not json_out):
        import dotenv  # lazy
        dotenv.load_dotenv()
        env = _require_env(["XTRACE_ORG_ID"])
    org_id = env["XTRACE_ORG_ID"]
    api_url = (os.getenv("XTRACE_API_URL") or "https://api.production.xtrace.ai").rstrip("/")

    admin_key = get_admin_key_cached(json_out=json_out, human_prompt_fn=secret_prompt)
    if not json_out:
        console.print(Rule(style="dim"))

    # confirmation
    if not json_out:
        ids_preview = ", ".join(kb_ids[:5]) + ("…" if len(kb_ids) > 5 else "")
        if not typer.confirm(
            f"You are about to permanently delete {len(kb_ids)} knowledge base(s): {ids_preview}. Continue?",
            default=False
        ):
            console.print("[yellow]Aborted. No changes made.[/]")
            raise typer.Exit(1)

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
            org_id=org_id, api_key="", api_url=api_url, admin_key=admin_key
        )
        _owned = True

    results: List[Dict[str, Any]] = []

    try:
        for kb_id in kb_ids:
            with _maybe_status(f"Deleting KB {kb_id}…", enable=not json_out):
                try:
                    asyncio.run(integration.delete_kb(kb_id))
                    results.append({"id": kb_id, "status": "deleted"})
                except Exception as e:
                    msg = str(e)
                    code = None
                    try:
                        import aiohttp
                        if isinstance(e, aiohttp.ClientResponseError):
                            code = e.status
                    except Exception:
                        pass
                    if code == 404:
                        results.append({"id": kb_id, "status": "not_found", "code": 404, "message": msg[:180]})
                    else:
                        results.append({"id": kb_id, "status": "error", "code": code, "message": msg[:180]})
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass

    if json_out:
        typer.echo(json.dumps(results, ensure_ascii=False))
        return

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("ID", style="blue", no_wrap=True)
    table.add_column("Result")
    table.add_column("Message", style="dim")

    ok = err = nf = 0
    for r in results:
        status = r["status"]
        if status == "deleted":
            ok += 1
            result_cell = "[green]deleted[/]"
            msg = ""
        elif status == "not_found":
            nf += 1
            result_cell = "[yellow]not found[/]"
            msg = r.get("message", "")
        else:
            err += 1
            result_cell = "[red]error[/]"
            msg = r.get("message", "")
        table.add_row(r["id"], result_cell, msg)

    console.print(table)
    console.print(
        f"[dim]Summary:[/] "
        f"[green]{ok} deleted[/], "
        f"[yellow]{nf} not found[/], "
        f"[red]{err} errors[/]"
    )
