from __future__ import annotations
import os
import asyncio
import warnings
import typer
from pathlib import Path
from typing import Dict, Any, Iterable, Iterator
from rich.console import Console
from rich.rule import Rule
from contextlib import contextmanager
from ._utils._errors import extract_server_error
from xtrace_sdk.x_vec.inference.embedding import EmbeddingError

from xtrace_sdk.cli.state import get_pre, get_exec_context, get_embed_model, get_integration

app = typer.Typer()
console = Console()

@contextmanager
def safe_status(message: str) ->Iterator[None]:
    try:
        with console.status(message, spinner="dots"):
            yield
    except TypeError:
        console.print(f"[dim]{message}[/]")
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

def _default_meta() -> Dict[str, Any]:
    return {"tag1": "cli", "tag2": None, "tag3": None, "tag4": None, "tag5": None}

def _to_chunk_collection(docs: Iterable) -> list[Dict[str, Any]]:
    """Convert LocalDiskConnector.load_data_from_file() output into chunks for DataLoaderBase."""
    out = []
    for doc in docs:
        content = getattr(doc, "page_content", doc)
        if content is None:
            continue
        out.append({"chunk_content": str(content), "meta_data": _default_meta()})
    return out

def _silence_transformers_future_warning()-> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*`encoder_attention_mask` is deprecated.*BertSdpaSelfAttention\.forward.*",
        category=FutureWarning,
    )

@app.command("upsert-file", help="Upsert chunks from a single file into a knowledge base.")
def upsert_file(
    file_path: str = typer.Argument(..., help="Path to a single file"),
    kb_id: str    = typer.Argument(..., help="Knowledge Base ID"),
    max_chunk_chars: int | None = typer.Option(
        None, "--max-chunk-chars",
        help="Maximum characters per chunk. Oversized chunks (e.g. large JSON objects) "
             "are split to fit within this limit. Useful when your embedding model has a "
             "small context window.",
    ),
    max_parallel_embeddings: int | None = typer.Option(
        None, "--max-parallel-embeddings",
        help="Maximum number of concurrent embedding requests. "
             "Useful when your embedding provider has concurrency limits (e.g. local Ollama).",
    ),
) -> None:
    _silence_transformers_future_warning()

    # env
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

    # validate file 
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        console.print(f"[red]File not found:[/] {file_path}")
        raise typer.Exit(1)

    console.print(f"[dim]KB[/]: [bold blue]{kb_id}[/]   •  [dim]File[/]: [bold magenta]{p}[/]")
    console.print(Rule(style="dim"))

    integration = None

    # components + file load
    with safe_status("Loading XTrace components and reading file…"):
        # Prefer preloaded deps; fall back to direct imports when missing
        pre = get_pre()
        XTraceIntegration = pre.XTraceIntegration
        DataLoaderBase    = pre.DataLoaderBase
        ExecutionContext  = pre.ExecutionContext
        Embedding         = pre.Embedding
        pkl               = pre.pkl
        PathType          = pre.Path  # avoid shadowing Path import above

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
        if PathType is None:
            from pathlib import Path as _PathT
            PathType = _PathT

        # LocalDiskConnector is lightweight; import on demand
        from xtrace_sdk.cli.commands._utils._disk_loader import LocalDiskConnector

        _shared = get_integration()
        integration = _shared or XTraceIntegration(org_id=org_id, api_key=api_key, api_url=api_url)
        _owned = _shared is None

    try:
        # Load file and build collection
        load_kwargs: dict[str, Any] = {}
        if max_chunk_chars is not None:
            load_kwargs["page_size"] = max_chunk_chars
            load_kwargs["cap_json_elements"] = True
        docs = LocalDiskConnector.load_data_from_file(str(p), **load_kwargs)  # may raise ValueError for unsupported types
        collection = _to_chunk_collection(docs)

        # Reuse in-session execution context if available; otherwise fall back to disk
        exec_obj = get_exec_context(load_if_missing=False)
        if exec_obj is None:
            exec_obj = ExecutionContext.load_from_disk(env["XTRACE_PASS_PHRASE"], exec_ctx)
        dl = DataLoaderBase(execution_context=exec_obj, integration=integration)

        if not collection:
            console.print("[yellow]No chunks produced from file.[/]")
            return

        with safe_status("Encrypting chunks and building index…"):
            try:
                # Reuse embedding model from session; otherwise load from disk
                embedding_model = get_embed_model(load_if_missing=False)
                if embedding_model is None:
                    # Use the preloaded Path/pkl if we captured them above; otherwise import here
                    try:
                        _ = PathType  
                    except NameError:
                        from pathlib import Path as PathType  
                    try:    
                        _ = pkl  
                    except NameError:
                        import pickle as pkl  

                    embed_bytes = PathType(env["XTRACE_EMBEDDING_MODEL_PATH"]).read_bytes()
                    obj = pkl.loads(embed_bytes)
                    embedding_model = obj.get("embedding") if isinstance(obj, dict) else obj

                async def _embed_all() -> list[Any]:
                    if max_parallel_embeddings is not None:
                        sem = asyncio.Semaphore(max_parallel_embeddings)
                        async def _limited(text: str) -> Any:
                            async with sem:
                                return await embedding_model.bin_embed(text)
                        return list(await asyncio.gather(*[
                            _limited(c["chunk_content"]) for c in collection
                        ]))
                    return list(await asyncio.gather(*[
                        embedding_model.bin_embed(c["chunk_content"]) for c in collection
                    ]))
                vectors = asyncio.run(_embed_all())
                index, db = asyncio.run(dl.load_data_from_memory(collection, vectors, disable_progress=True))  # type: ignore[arg-type]
            except EmbeddingError as e:
                status = f" ({e.status})" if e.status is not None else ""
                detail = f" (chunk was {e.chunk_len:,} chars)" if e.chunk_len else ""
                console.print(f"[red]Embedding failed{status}:[/] {e}{detail}")
                err_lower = str(e).lower()
                if e.chunk_len and "context length" in err_lower:
                    console.print(
                        "[yellow]Hint:[/] some chunks exceed your embedding model's context window. "
                        "Re-run with [bold]--max-chunk-chars N[/] to split large chunks "
                        "(e.g. [dim]--max-chunk-chars 3000[/])."
                    )
                elif "server busy" in err_lower or "pending requests" in err_lower:
                    console.print(
                        "[yellow]Hint:[/] too many concurrent embedding requests. "
                        "Re-run with [bold]--max-parallel-embeddings N[/] to limit concurrency "
                        "(e.g. [dim]--max-parallel-embeddings 5[/])."
                    )
                raise typer.Exit(1)
            except Exception as e:
                msg, code = extract_server_error(e)
                console.print(f"[red]Encryption failed{f' ({code})' if code is not None else ''}:[/] {msg or e}")
                raise typer.Exit(1)

        with safe_status(f"Uploading encrypted data to KB '{kb_id}'…"):
            try:
                asyncio.run(dl.dump_db(db, index=index, kb_id=kb_id))
            except Exception as e:
                msg, code = extract_server_error(e)
                console.print(f"[red]Upload failed{f' ({code})' if code is not None else ''}:[/] {msg or e}")
                raise typer.Exit(1)

        console.print(f"[bold green]Upserted {len(collection)} chunk(s)[/] → KB [bold blue]{kb_id}[/].")
        console.print("[dim]Note: the CLI uses basic chunking. For custom chunking strategies, use the Python SDK.[/]")
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass
