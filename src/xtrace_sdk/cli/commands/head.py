from __future__ import annotations
import os
import re
import json
import asyncio
import typer
from typing import Generator, Iterator
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from contextlib import contextmanager
from ._utils._prompt import secret_prompt
from ._utils._admin_key import get_admin_key
from ..state import get_admin_key_cached
from ._utils._errors import extract_server_error

from xtrace_sdk.cli.state import get_pre, get_exec_context, get_integration

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

@contextmanager
def _integration_scope(integration: Any)-> Generator[None, None, None]:
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
    s = _clean(text)
    if fullChunks or len(s) <= 100:
        return f"\"{s}\""
    return f"\"{s[:100]}...\""

def _preview_json(text: str, fullChunks: bool) -> str:
    s = _clean(text)
    if fullChunks or len(s) <= 100:
        return s
    return s[:100] + "..."

@app.command("head", help="Preview vectors in a knowledge base.")
def head(
    kb_id: str = typer.Argument(..., help="Knowledge Base ID to preview"),
    all: bool = typer.Option(False, "--all", help="Show all vectors (default shows up to 25)"),
    fullChunks: bool = typer.Option(False, "--fullChunks", help="Show full chunk content (no truncation)"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON array with id and content"),
    api_key_override: str | None = typer.Option(
        None, "-a", "--api_key", help="Use this API key instead of XTRACE_API_KEY from .env"
    ),
)-> None:
    # load env
    with _maybe_status("Loading environment…", enable=not json_out):
        import dotenv
        dotenv.load_dotenv()
        required = ["XTRACE_ORG_ID", "XTRACE_API_URL", "XTRACE_EXECUTION_CONTEXT_PATH", "XTRACE_PASS_PHRASE"]
        if not api_key_override:
            required.append("XTRACE_API_KEY")
        env = _require_env(required)

    org_id = env["XTRACE_ORG_ID"]
    api_key = api_key_override or env.get("XTRACE_API_KEY")
    api_url = env["XTRACE_API_URL"].rstrip("/")
    exec_ctx = env["XTRACE_EXECUTION_CONTEXT_PATH"]

    # admin key from env, else prompt
    admin_key = get_admin_key_cached(json_out=json_out, human_prompt_fn=secret_prompt)

    if not json_out:
        console.print(f"[dim]KB[/]: [bold blue]{kb_id}[/]")
        console.print(Rule(style="dim"))

    # resolve components
    with _maybe_status("Loading XTrace components…", enable=not json_out):
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

    # one integration for both the metadata fetch and the retriever
    _shared = get_integration()
    if _shared is not None and not api_key_override:
        integration = _shared
        _owned = False
    else:
        integration = XTraceIntegration(org_id=org_id, api_key=api_key, api_url=api_url, admin_key=admin_key)
        _owned = True

    try:
        # fetch kb metadata
        with _maybe_status("Fetching knowledge base metadata…", enable=not json_out):
            try:
                kb: Dict[str, Any] = asyncio.run(integration.get_kb(kb_id)) or {}
            except Exception as e:
                msg, code = extract_server_error(e)
                if json_out:
                    typer.echo(json.dumps({"error": {"message": msg or f"Network error: {e}", "status": code}}, ensure_ascii=False))
                else:
                    console.print(f"[red]Network error[/]: {msg or e}")
                raise typer.Exit(1)

        try:
            total = int(kb.get("numberOfVectors") or 0)
        except Exception:
            total = 0

        if total == 0:
            if json_out:
                typer.echo(json.dumps({"kb_id": kb_id, "vectors": []}, ensure_ascii=False))
            else:
                console.print("[yellow]This knowledge base has no vectors.[/]")
            return

        limit = total if all else min(25, total)
        if not json_out:
            console.print(f"[dim]Vectors[/]: showing [bold]{limit}[/] of [bold]{total}[/]")

        # build retriever
        exec_context = get_exec_context(load_if_missing=False)
        if exec_context is None:
            assert ExecutionContext is not None
            exec_context = ExecutionContext.load_from_disk(env["XTRACE_PASS_PHRASE"], exec_ctx)
        assert SimpleRetriever is not None
        retriever = SimpleRetriever(exec_context, integration)

        # fetch and decrypt
        chunk_ids = list(range(1, limit + 1))
        with _maybe_status("Fetching and decrypting chunks…", enable=not json_out):
            try:
                contexts = asyncio.run(retriever.retrieve_and_decrypt(chunk_ids, kb_id=kb_id))
            except Exception as e:
                msg, code = extract_server_error(e)
                if json_out:
                    typer.echo(json.dumps(
                        {"error": {"message": msg or str(e), "status": code, "kb_id": kb_id}},
                        ensure_ascii=False,
                    ))
                else:
                    console.print(f"[red]Failed to fetch chunks{f' ({code})' if code is not None else ''}:[/] {msg or e}")
                raise typer.Exit(1)

        if json_out:
            out = []
            for cid, ctx in zip(chunk_ids, contexts, strict=False):
                content = _preview_json(str(ctx.get("chunk_content", "") or ""), fullChunks=fullChunks)
                out.append({"id": cid, "content": content})
            typer.echo(json.dumps(out, ensure_ascii=False))
            return

        # human table
        table = Table(show_header=True, header_style="bold", expand=True, show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Content")

        for cid, ctx in zip(chunk_ids, contexts, strict=False):
            content = _preview_table(str(ctx.get("chunk_content", "") or ""), fullChunks=fullChunks)
            table.add_row(str(cid), content)

        console.print(table)
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass
