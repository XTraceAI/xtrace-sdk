from __future__ import annotations
import os
import asyncio
import warnings
import typer
from typing import Dict, Any
from rich.console import Console
from rich.rule import Rule
from contextlib import contextmanager
from ._utils._errors import extract_server_error
from xtrace_sdk.x_vec.inference.embedding import EmbeddingError
from collections.abc import Generator

from xtrace_sdk.cli.state import get_pre, get_exec_context, get_embed_model, get_integration
from typing import Protocol, Sequence, Any

class _EmbeddingProto(Protocol):
    async def bin_embed(self, text: str) -> Sequence[int]: ...


app = typer.Typer()
console = Console()

@contextmanager
def safe_status(message: str)-> Generator[None, None, None]:
    try:
        with console.status(message, spinner="dots"):
            yield
    except TypeError:
        console.print(f"[dim]{message}[/]")
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

def _default_meta() -> Dict[str, Any]:
    return {"tag1": "cli", "tag2": None, "tag3": None, "tag4": None, "tag5": None}

def _silence_transformers_future_warning()-> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*`encoder_attention_mask` is deprecated.*BertSdpaSelfAttention\.forward.*",
        category=FutureWarning,
    )

@app.command("upsert", help="Upsert a single text chunk into a knowledge base.")
def upsert(
    kb_id: str = typer.Argument(..., help="Knowledge Base ID"),
    text: str  = typer.Argument(..., help='Text to store (wrap in quotes)'),
)->None:
    _silence_transformers_future_warning()

    with safe_status("Loading environment…"):
        import dotenv 
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
        env = _require_env([
            "XTRACE_ORG_ID",
            "XTRACE_API_KEY",
            "XTRACE_API_URL",
            "XTRACE_EXECUTION_CONTEXT_PATH",
            "XTRACE_PASS_PHRASE",
            "XTRACE_EMBEDDING_MODEL_PATH",
        ])

    org_id  = env["XTRACE_ORG_ID"]
    api_key = env["XTRACE_API_KEY"]
    api_url = env["XTRACE_API_URL"].rstrip("/")
    exec_ctx = env["XTRACE_EXECUTION_CONTEXT_PATH"]

    console.print(f"[dim]KB[/]: [bold blue]{kb_id}[/]")
    console.print(Rule(style="dim"))

    collection = [{"chunk_content": text, "meta_data": _default_meta()}]

    with safe_status("Loading XTrace components…"):
        pre = get_pre()
        XTraceIntegration = pre.XTraceIntegration
        DataLoaderBase    = pre.DataLoaderBase
        ExecutionContext  = pre.ExecutionContext
        Embedding         = pre.Embedding
        pkl               = pre.pkl
        Path              = pre.Path

        # standalone fallback for anything missing
        if XTraceIntegration is None:
            from xtrace_sdk.integrations.xtrace import XTraceIntegration as _XI
            XTraceIntegration = _XI
        if DataLoaderBase is None:
            from xtrace_sdk.x_vec.data_loaders.loader import DataLoader as _DLB
            DataLoaderBase = _DLB
        if ExecutionContext is None:
            from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext as _EC
            ExecutionContext = _EC
        if Embedding is None:
            from xtrace_sdk.x_vec.inference.embedding import Embedding as _EM
            Embedding = _EM
        if pkl is None:
            import pickle as _pkl
            pkl = _pkl
        if Path is None:
            from pathlib import Path as _Path
            Path = _Path

        _shared = get_integration()
        integration = _shared or XTraceIntegration(org_id=org_id, api_key=api_key, api_url=api_url)
        _owned = _shared is None

    try:
        # Reuse in-session execution context if available; otherwise fall back to disk
        exec_obj = get_exec_context(load_if_missing=False)
        try:
            if exec_obj is None:
                exec_obj = ExecutionContext.load_from_disk(env["XTRACE_PASS_PHRASE"], exec_ctx)
            dl = DataLoaderBase(execution_context=exec_obj, integration=integration)
        except Exception as e:
            msg, code = extract_server_error(e)
            console.print(f"[red]Failed to initialize loader{f' ({code})' if code is not None else ''}:[/] {msg or e}")
            raise typer.Exit(1)

        with safe_status("Encrypting chunk and building index…"):
            try:
                # Load embedding model and create binary vector(s)
                embed_bytes = Path(env["XTRACE_EMBEDDING_MODEL_PATH"]).read_bytes()
                embedding_model: _EmbeddingProto = pkl.loads(embed_bytes)

                # mypy expects list[list[float]] → convert bin_embed output to floats
                async def _embed_all() -> list[Sequence[int]]:
                    return list(await asyncio.gather(*[
                        embedding_model.bin_embed(str(c["chunk_content"])) for c in collection
                    ]))
                vectors = [[float(x) for x in emb] for emb in asyncio.run(_embed_all())]

                index, db = asyncio.run(dl.load_data_from_memory(
                    collection,  # type: ignore[arg-type]
                    vectors,
                    disable_progress=True
                ))
            except EmbeddingError as e:
                status = f" ({e.status})" if e.status is not None else ""
                detail = f" (chunk was {e.chunk_len:,} chars)" if e.chunk_len else ""
                console.print(f"[red]Embedding failed{status}:[/] {e}{detail}")
                err_lower = str(e).lower()
                if e.chunk_len and "context length" in err_lower:
                    console.print(
                        "[yellow]Hint:[/] the text exceeds your embedding model's context window. "
                        "Try shortening the input."
                    )
                elif "server busy" in err_lower or "pending requests" in err_lower:
                    console.print(
                        "[yellow]Hint:[/] embedding server is overloaded. Try again shortly."
                    )
                raise typer.Exit(1)
            except Exception as e:
                msg, code = extract_server_error(e)
                console.print(
                    f"[red]Encryption failed{f' ({code})' if code is not None else ''}:[/] {msg or e}"
                )
                raise typer.Exit(1)

        with safe_status(f"Uploading encrypted data to KB '{kb_id}'…"):
            try:
                asyncio.run(dl.dump_db(db, index=index, kb_id=kb_id))
            except Exception as e:
                msg, code = extract_server_error(e)
                console.print(f"[red]Upload failed{f' ({code})' if code is not None else ''}:[/] {msg or e}")
                raise typer.Exit(1)

        console.print(f"[bold green]Upserted 1 chunk[/] → KB [bold blue]{kb_id}[/].")
        console.print("[dim]Note: the CLI uses basic chunking. For custom chunking strategies, use the Python SDK.[/]")
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass