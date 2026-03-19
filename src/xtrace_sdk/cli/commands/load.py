from __future__ import annotations
import os
import re
import asyncio
import typer
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Set, Tuple, cast, Iterator
from rich.console import Console
from rich.rule import Rule
from contextlib import contextmanager
import warnings
from ._utils._errors import extract_server_error

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

def _empty_meta() -> Dict[str, Any]:
    return {"tag1": None, "tag2": None, "tag3": None, "tag4": None, "tag5": None}

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
        out.append({"chunk_content": str(content), "meta_data": _empty_meta()})
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

def _collect_from_folder(folder: str, types: Set[str] | None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Walk folder, load only matching files via LocalDiskConnector.load_data_from_file,
    convert to chunks with required meta_data. Returns (chunks, matched_file_count).
    """
    from xtrace_sdk.cli.commands._utils._disk_loader import LocalDiskConnector  # lazy

    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"Folder not found or not a directory: {folder}")

    collection: List[Dict[str, Any]] = []
    matched_files = 0

    for path in _iter_matching_files(root, types):
        try:
            docs = LocalDiskConnector.load_data_from_file(str(path))
        except ValueError:
            # unsupported type or unreadable file — skip silently since we filtered by ext
            continue
        chunks = _to_chunk_collection(docs)
        if chunks:
            matched_files += 1
            collection.extend(chunks)

    return collection, matched_files

@app.command("load", help="Load data into an XTrace Knowledge Base.")
def load(
    folder_path: str = typer.Argument(..., help="Path to the folder containing data"),
    kb_id: str = typer.Argument(..., help="Knowledge Base ID to load data into"),
    filetypes: str | None = typer.Option(
        None, "-f", "--filetypes",
        help="Comma-separated list of file types to include, e.g. 'txt,json' or '[txt, json]'. "
             "Types are matched to file extensions (no dot), case-insensitive."
    ),
) -> None:
    # load .env
    with safe_status("Loading environment…"):
        import dotenv  # lazy import
        dotenv.load_dotenv()
        env = _require_env(
            "XTRACE_ORG_ID", "XTRACE_API_KEY", "XTRACE_API_URL",
            "XTRACE_EXECUTION_CONTEXT_PATH", "XTRACE_PASS_PHRASE", "XTRACE_EMBEDDING_MODEL_PATH"
        )

    types = _parse_filetypes(filetypes)
    if types is not None and not types:
        console.print("[yellow]No valid file types provided after parsing filter.[/]")
        raise typer.Exit(2)

    types_msg = f"  •  [dim]Types[/]: [bold]{', '.join(sorted(types))}[/]" if types else ""
    console.print(
        f"[dim]KB[/]: [bold blue]{kb_id}[/]  •  [dim]Folder[/]: [bold white]{folder_path}[/]{types_msg}"
    )
    console.print(Rule(style="dim"))

    with safe_status("Loading XTrace components…"):
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
        with safe_status("Scanning files and preparing chunks…"):
            collection, matched_files = _collect_from_folder(folder_path, types)

        if matched_files == 0:
            if types:
                console.print(f"[yellow]No files matched the provided types ({', '.join(sorted(types))}).[/]")
            else:
                console.print("[yellow]No readable files found.[/]")
            raise typer.Exit(1)

        console.print(f"[dim]Matched files[/]: [bold]{matched_files}[/]")
        try:
            n_chunks = len(collection)
            console.print(f"[dim]Prepared[/]: [bold cyan]{n_chunks} chunk(s)[/]")
        except Exception:
            n_chunks = None

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
            return list(await asyncio.gather(*[
                embedding_model.bin_embed(c["chunk_content"]) for c in collection
            ]))
        vectors = asyncio.run(_embed_all())

        index, db = dl.load_data_from_memory(collection, vectors)  # type: ignore[arg-type]
        console.print(Rule(style="dim"))

        # upload
        with safe_status(f"Uploading encrypted data to KB '{kb_id}'…"):
            try:
                asyncio.run(dl.dump_db(db, index=index, kb_id=kb_id))
            except Exception as e:
                msg, code = extract_server_error(e)
                if msg:
                    status = f" ({code})" if code is not None else ""
                    console.print(f"[red]Upload failed{status}:[/] {msg}")
                else:
                    console.print(f"[red]Upload failed:[/] {e}")
                raise typer.Exit(1)

        if n_chunks is not None:
            console.print(f"[bold green]Loaded {n_chunks} chunk(s)[/] → KB [bold blue]{kb_id}[/]")
        else:
            console.print(f"Loaded data → KB [bold]{kb_id}[/]")
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass
