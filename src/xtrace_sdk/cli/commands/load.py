from __future__ import annotations
import os
import re
import json
import asyncio
import typer
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Set, Tuple, cast, Iterator
from rich.console import Console
from rich.rule import Rule
from contextlib import contextmanager
import warnings
from ._utils._errors import extract_server_error
from xtrace_sdk.x_vec.inference.embedding import EmbeddingError

from xtrace_sdk.cli.state import get_pre, get_exec_context, get_embed_model, get_integration

app = typer.Typer()
console = Console()

def _silence_transformers_future_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*`encoder_attention_mask` is deprecated.*BertSdpaSelfAttention\.forward.*",
        category=FutureWarning,
    )

@contextmanager
def safe_status(message: str) -> Iterator[None]:
    """Rich status spinner that also works on older Rich versions (no transient kw)."""
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

def _require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        console.print(
            "[red]Missing required environment variables:[/] "
            + ", ".join(missing)
            + "\n[dim]Tip: run `xtrace init` first to create .env[/]"
        )
        raise typer.Exit(2)
    return {n: os.environ[n] for n in names}

# ----- filetype filter helpers -----
_brackets = re.compile(r"^[\[\(\{](.*)[\]\)\}]$")

def _parse_filetypes(s: str | None) -> Set[str] | None:
    """Parse 'txt,json' or '[txt, json]' → {'txt','json'}; returns None if s is falsy."""
    if not s:
        return None
    s = s.strip()
    m = _brackets.match(s)
    if m:
        s = m.group(1)
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    # ensure no leading dots
    return {p[1:] if p.startswith(".") else p for p in parts}

def _default_meta() -> Dict[str, Any]:
    return {"tag1": "cli", "tag2": None, "tag3": None, "tag4": None, "tag5": None}

def _to_chunk_collection(docs: Iterable) -> List[Dict[str, Any]]:
    """
    Convert load_data_from_file() output into chunks expected by DataLoaderBase.
    Each element may be a Document-like object with .page_content or a str.
    """
    out: List[Dict[str, Any]] = []
    for doc in docs:
        content = getattr(doc, "page_content", doc)
        if content is None:
            continue
        out.append({"chunk_content": str(content), "meta_data": _default_meta()})
    return out

def _iter_matching_files(root: Path, types: Set[str] | None) -> Iterable[Path]:
    """Yield absolute file paths under root matching given extensions (no leading dot), case-insensitive."""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = (Path(dirpath) / fn).resolve()
            if types is None:
                yield p
                continue
            ext = p.suffix.lower().lstrip(".")
            if ext in types:
                yield p

def _collect_from_folder(
    folder: str,
    types: Set[str] | None,
    *,
    max_chunk_chars: int | None = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Walk folder, load only matching files via LocalDiskConnector.load_data_from_file,
    convert to chunks with required meta_data. Returns (chunks, matched_file_count).
    """
    from xtrace_sdk.cli.commands._utils._disk_loader import LocalDiskConnector  # lazy

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"Folder not found or not a directory: {folder}")

    load_kwargs: Dict[str, Any] = {}
    if max_chunk_chars is not None:
        load_kwargs["page_size"] = max_chunk_chars
        load_kwargs["cap_json_elements"] = True

    collection: List[Dict[str, Any]] = []
    matched_files = 0

    for path in _iter_matching_files(root, types):
        try:
            docs = LocalDiskConnector.load_data_from_file(str(path), **load_kwargs)
        except ValueError:
            # unsupported type or unreadable file — skip silently since we filtered by ext
            continue
        chunks = _to_chunk_collection(docs)
        if chunks:
            matched_files += 1
            collection.extend(chunks)

    return collection, matched_files

def _json_error(msg: str | None, code: int | None, **extra: Any) -> None:
    payload: Dict[str, Any] = {"error": {"message": msg or "Unknown error", "status": code}}
    payload["error"].update(extra)
    typer.echo(json.dumps(payload, ensure_ascii=False))

@app.command("load", help="Load data into an XTrace Knowledge Base.")
def load(
    folder_path: str = typer.Argument(..., help="Path to the folder containing data"),
    kb_id: str = typer.Argument(..., help="Knowledge Base ID to load data into"),
    filetypes: str | None = typer.Option(
        None, "-f", "--filetypes",
        help="Comma-separated list of file types to include, e.g. 'txt,json' or '[txt, json]'. "
             "Types are matched to file extensions (no dot), case-insensitive."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON summary"),
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
    # load .env
    with _maybe_status("Loading environment…", enable=not json_out):
        import dotenv  # lazy import
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
        env = _require_env(
            "XTRACE_ORG_ID", "XTRACE_API_KEY", "XTRACE_API_URL",
            "XTRACE_EXECUTION_CONTEXT_PATH", "XTRACE_PASS_PHRASE", "XTRACE_EMBEDDING_MODEL_PATH"
        )

    types = _parse_filetypes(filetypes)
    if filetypes is not None and not types:
        if json_out:
            _json_error("No valid file types provided after parsing filter.", None)
        else:
            console.print("[yellow]No valid file types provided after parsing filter.[/]")
        raise typer.Exit(2)

    if not json_out:
        types_msg = f"  •  [dim]Types[/]: [bold]{', '.join(sorted(types))}[/]" if types else ""
        console.print(
            f"[dim]KB[/]: [bold blue]{kb_id}[/]  •  [dim]Folder[/]: [bold white]{folder_path}[/]{types_msg}"
        )
        console.print(Rule(style="dim"))

    with _maybe_status("Loading XTrace components…", enable=not json_out):
        _silence_transformers_future_warning()

        pre = get_pre()
        XTraceIntegration = pre.XTraceIntegration
        DataLoaderBase    = pre.DataLoaderBase
        ExecutionContext  = pre.ExecutionContext
        Embedding         = pre.Embedding
        pkl               = pre.pkl
        Path              = getattr(pre, "Path", None)

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
    integration = _shared or XTraceIntegration(
        org_id=env["XTRACE_ORG_ID"],
        api_key=env["XTRACE_API_KEY"],
        api_url=env["XTRACE_API_URL"],
    )
    _owned = _shared is None
    try:
        # scan and chunk local files (filtered)
        with _maybe_status("Scanning files and preparing chunks…", enable=not json_out):
            collection, matched_files = _collect_from_folder(folder_path, types, max_chunk_chars=max_chunk_chars)

        if matched_files == 0:
            if json_out:
                _json_error("No files matched" + (f" the provided types ({', '.join(sorted(types))})" if types else ""), None, kb_id=kb_id)
            elif types:
                console.print(f"[yellow]No files matched the provided types ({', '.join(sorted(types))}).[/]")
            else:
                from xtrace_sdk.cli.commands._utils._disk_loader import _SUPPORTED
                supported = ", ".join(sorted(_SUPPORTED))
                console.print(f"[yellow]No readable files found. Supported types: {supported}[/]")
            console.print("[dim]Hint: for advanced ingestion workflows, use the Python SDK directly.[/]")
            raise typer.Exit(1)

        n_chunks = len(collection)
        if not json_out:
            console.print(f"[dim]Matched files[/]: [bold]{matched_files}[/]")
            console.print(f"[dim]Prepared[/]: [bold cyan]{n_chunks} chunk(s)[/]")

        exec_context = get_exec_context(load_if_missing=False)
        if exec_context is None:
            exec_context = ExecutionContext.load_from_disk(env["XTRACE_PASS_PHRASE"], env["XTRACE_EXECUTION_CONTEXT_PATH"])
        dl = DataLoaderBase(execution_context=exec_context, integration=integration)

        embedding_model = get_embed_model(load_if_missing=False)
        if embedding_model is None:
            embed_bytes = Path(env["XTRACE_EMBEDDING_MODEL_PATH"]).read_bytes()
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
        with _maybe_status("Embedding chunks…", enable=not json_out):
            try:
                vectors = asyncio.run(_embed_all())
            except EmbeddingError as e:
                status = f" ({e.status})" if e.status is not None else ""
                detail = f" (chunk was {e.chunk_len:,} chars)" if e.chunk_len else ""
                if json_out:
                    _json_error(str(e), e.status, kb_id=kb_id)
                else:
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
                if json_out:
                    _json_error(msg or str(e), code, kb_id=kb_id)
                elif msg:
                    status = f" ({code})" if code is not None else ""
                    console.print(f"[red]Embedding failed{status}:[/] {msg}")
                else:
                    console.print(f"[red]Embedding failed:[/] {e}")
                raise typer.Exit(1)

        index, db = asyncio.run(dl.load_data_from_memory(collection, vectors))  # type: ignore[arg-type]
        if not json_out:
            console.print(Rule(style="dim"))

        # upload
        with _maybe_status(f"Uploading encrypted data to KB '{kb_id}'…", enable=not json_out):
            try:
                asyncio.run(dl.dump_db(db, index=index, kb_id=kb_id))
            except Exception as e:
                msg, code = extract_server_error(e)
                if json_out:
                    _json_error(msg or str(e), code, kb_id=kb_id)
                elif msg:
                    status = f" ({code})" if code is not None else ""
                    console.print(f"[red]Upload failed{status}:[/] {msg}")
                else:
                    console.print(f"[red]Upload failed:[/] {e}")
                raise typer.Exit(1)

        if json_out:
            typer.echo(json.dumps({
                "kb_id": kb_id,
                "matched_files": matched_files,
                "chunks_loaded": n_chunks,
                "status": "success",
            }, ensure_ascii=False))
        else:
            console.print(f"[bold green]Loaded {n_chunks} chunk(s)[/] → KB [bold blue]{kb_id}[/]")
            console.print("[dim]Note: the CLI uses basic chunking. For custom chunking strategies, use the Python SDK.[/]")
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass
