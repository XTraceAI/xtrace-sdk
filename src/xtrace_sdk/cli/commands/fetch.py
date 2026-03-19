from __future__ import annotations
import os
import re
import json
import asyncio
import typer
from typing import List, Dict, Any
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from contextlib import contextmanager
from ._utils._errors import extract_server_error

from xtrace_sdk.cli.state import get_pre, get_exec_context, get_integration

from typing import Annotated
from typing import Any, Generator
from typing import Iterator

app = typer.Typer()
console = Console()

@contextmanager
def safe_status(message: str) -> Iterator[None]:
    """Rich status spinner that works on older Rich versions, too."""
    try:
        with console.status(message, spinner="dots"):
            yield
    except TypeError:
        console.print(f"[dim]{message}[/]")
        yield

@contextmanager
def _maybe_status(message: str, enable: bool) -> Iterator[None]:
    """Disable spinner output when enable=False (e.g., --json)."""
    if enable:
        with safe_status(message):
            yield
    else:
        yield

@contextmanager
def _integration_scope(integration: Any) -> Generator[None, None, None]:
    try:
        yield
    finally:
        try:
            asyncio.run(integration.close())
        except Exception:
            pass

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

_ws_re = re.compile(r"[\n\r\t]+")
_collapse_re = re.compile(r"\s+")

def _clean(text: str) -> str:
    if text is None:
        return ""
    s = _ws_re.sub(" ", text)
    s = _collapse_re.sub(" ", s).strip()
    return s

def _preview_table(text: str, fullChunks: bool) -> str:
    """Human/table: quote, truncate to 50 chars unless fullChunks."""
    s = _clean(text)
    if fullChunks or len(s) <= 50:
        return f"\"{s}\""
    return f"\"{s[:50]}...\""

def _preview_json(text: str, fullChunks: bool) -> str:
    """JSON: raw string (not additionally quoted), truncate to 50 chars unless fullChunks."""
    s = _clean(text)
    if fullChunks or len(s) <= 50:
        return s
    return s[:50] + "..."

@app.command("fetch", help="Fetch specific vectors by ID from a knowledge base.")
def fetch(
    kb_id: Annotated[str, typer.Argument(..., help="Knowledge Base ID")],
    vector_ids: Annotated[
        list[int] | None,
        typer.Argument(..., metavar="VECTOR_ID...", help="One or more vector IDs to fetch")
    ] = None,
    fullChunks: bool = typer.Option(False, "--fullChunks", help="Show full chunk content (no truncation)"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON array of {id, content}"),
    api_key_override: str | None = typer.Option(
        None, "-a", "--api_key", help="Use this API key instead of XTRACE_API_KEY from .env"
    ),
)-> None:
    # mypy: Typer guarantees vector_ids is never None because argument is required
    assert vector_ids is not None
    # load environment
    with _maybe_status("Loading environment…", enable=not json_out):
        import dotenv  # lazy
        dotenv.load_dotenv()
        required = ["XTRACE_ORG_ID", "XTRACE_API_URL", "XTRACE_EXECUTION_CONTEXT_PATH", "XTRACE_PASS_PHRASE"]
        if not api_key_override:
            required.append("XTRACE_API_KEY")
        env = _require_env(required)

    org_id  = env["XTRACE_ORG_ID"]
    api_key = api_key_override or env.get("XTRACE_API_KEY")
    api_url = env["XTRACE_API_URL"].rstrip("/")
    exec_ctx = env["XTRACE_EXECUTION_CONTEXT_PATH"]

    if not json_out:
        ids_preview = ", ".join(str(i) for i in vector_ids[:6]) + ("…" if len(vector_ids) > 6 else "")
        console.print(f"[dim]KB[/]: [bold blue]{kb_id}[/]  •  [dim]IDs[/]: [bold]{ids_preview}[/]")
        console.print(Rule(style="dim"))

    # build retriever
    with _maybe_status("Loading XTrace components…", enable=not json_out):
        # Prefer preloaded deps; fall back to direct imports when missing
        pre = get_pre()
        SimpleRetriever   = pre.SimpleRetriever
        XTraceIntegration = pre.XTraceIntegration
        ExecutionContext  = pre.ExecutionContext

        if SimpleRetriever is None:
            from xtrace_sdk.x_vec.retrievers.retriever import Retriever as _SR
            SimpleRetriever = _SR
        if XTraceIntegration is None:
            from xtrace_sdk.integrations.xtrace import XTraceIntegration as _XI
            XTraceIntegration = _XI
        if ExecutionContext is None:
            from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext as _EC
            ExecutionContext = _EC

        _shared = get_integration()
        if _shared is not None and not api_key_override:
            integration = _shared
            _owned = False
        else:
            integration = XTraceIntegration(org_id=org_id, api_key=api_key, api_url=api_url)
            _owned = True

        try:
            # Reuse in-session execution context if available; otherwise load from disk
            exec_context = get_exec_context(load_if_missing=False)
            if exec_context is None:
                exec_context = ExecutionContext.load_from_disk(env["XTRACE_PASS_PHRASE"], exec_ctx)

            retriever = SimpleRetriever(exec_context, integration)
        except Exception:
            # ensure we don't leak the aiohttp session on init errors
            if _owned:
                try:
                    asyncio.run(integration.close())
                except Exception:
                    pass
            raise

    # fetch & decrypt
    try:
        # fetch & decrypt
        with _maybe_status("Fetching and decrypting chunks…", enable=not json_out):
            try:
                contexts = asyncio.run(retriever.retrieve_and_decrypt(vector_ids, kb_id=kb_id))
            except Exception as e:
                # clean error 
                msg, code = extract_server_error(e)
                if json_out:
                    typer.echo(json.dumps(
                        {"error": {"message": msg or str(e), "status": code, "kb_id": kb_id, "ids": vector_ids}},
                        ensure_ascii=False,
                    ))
                else:
                    status = f" ({code})" if code is not None else ""
                    console.print(f"[red]Fetch failed{status}:[/] {msg or e}")
                raise typer.Exit(1)

        # json output
        if json_out:
            out: List[Dict[str, Any]] = []
            for cid, ctx in zip(vector_ids, contexts, strict=False):
                out.append({
                    "id": cid,
                    "content": _preview_json(str(ctx.get("chunk_content", "") or ""), fullChunks=fullChunks)
                })
            typer.echo(json.dumps(out, ensure_ascii=False))
            return

        # human table
        table = Table(show_header=True, header_style="bold", expand=True, show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Content")

        if len(contexts) != len(vector_ids):
            console.print("[yellow]Note:[/] some requested vector IDs did not return results.")

        for cid, ctx in zip(vector_ids, contexts, strict=False):
            content = _preview_table(str(ctx.get("chunk_content", "") or ""), fullChunks=fullChunks)
            table.add_row(str(cid), content)

        console.print(table)
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass