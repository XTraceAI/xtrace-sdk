from __future__ import annotations
import os
import re
import io
import sys
import json
import asyncio
import warnings
import typer
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
from ._utils._errors import extract_server_error
from typing import cast
from typing import NoReturn
from typing import Any, Iterator
from typing import List

from xtrace_sdk.cli.state import get_pre, get_exec_context, get_embed_model, get_integration
from ._utils._admin_key import resolve_api_key_override

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

def _silence_transformers_future_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*`encoder_attention_mask` is deprecated.*BertSdpaSelfAttention\.forward.*",
        category=FutureWarning,
    )

_ALLOWED_INFERENCE = {"ollama", "openai", "redpill", "claude"}

_ws_re = re.compile(r"[\n\r\t]+")
_collapse_re = re.compile(r"\s+")

def _clean_full(text: str) -> str:
    """Replace newlines/tabs with spaces and collapse whitespace; no truncation."""
    if text is None:
        return ""
    s = _ws_re.sub(" ", text)
    s = _collapse_re.sub(" ", s).strip()
    return s

class _SafeCapture(io.StringIO):
    # Pretend to be a non-TTY text stream with an encoding; avoids AttributeErrors in libs
    def isatty(self)  -> bool: return False
    def __init__(self) -> None:
        super().__init__()
        self.encoding = "utf-8"
    def fileno(self) -> NoReturn:  # some libs probe for fileno; raise OSError like StringIO would
        raise OSError("fileno() not supported")

def _to_py_int_list(ids: Any)-> List[int]:
    # Accept list/tuple/numpy array and coerce to plain ints
    try:
        import numpy as _np  # already available via pre
        if isinstance(ids, _np.ndarray):
            return [int(x) for x in ids.tolist()]
    except Exception:
        pass
    return [int(x) for x in ids]

@contextmanager
def _maybe_status(message: str, enable: bool) -> Iterator[None]:
    """No-op status context when JSON mode is active."""
    if not enable:
        yield
    else:
        with safe_status(message):
            yield
@contextmanager
def _integration_scope(integration: Any ) -> Iterator[None]:
    try:
        yield
    finally:
        try:
            asyncio.run(integration.close())
        except Exception:
            pass

def _json_error(msg: str | None, code: int | None) -> NoReturn:
    payload = {"error": {"message": msg or "Unknown error", "status": code}}
    typer.echo(json.dumps(payload, ensure_ascii=False))
    raise typer.Exit(1)

@app.command("retrieve", help="Retrieve from an XTrace knowledge base (optionally run inference).")
def retrieve(
    kb_id: str = typer.Argument(..., help="Knowledge Base ID to retrieve from"),
    query: str = typer.Argument(..., help="Query to search for"),
    k: int = typer.Option(3, "-k", help="Number of results (top-k)"),
    inference: str | None = typer.Option(None, "--inference", help="ollama | openai | redpill | claude"),
    model: str | None = typer.Option(None, "--model", help="Model name to use with --inference"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON only"),
    api_key_override: str | None = typer.Option(
        None, "-a", "--api_key",
        help="Override API key (bypass XTRACE_API_KEY in .env)"
    ),
)-> None:
    _silence_transformers_future_warning()

    inf = inference.lower() if inference else None
    if inf:
        if inf not in _ALLOWED_INFERENCE:
            raise typer.BadParameter(f"--inference must be one of: {', '.join(sorted(_ALLOWED_INFERENCE))}")
        if not model:
            raise typer.BadParameter("--model is required when using --inference")

    # load environment
    with _maybe_status("Loading environment…", enable=not json_out):
        import dotenv
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
        required = ["XTRACE_ORG_ID", "XTRACE_API_URL", "XTRACE_EXECUTION_CONTEXT_PATH", "XTRACE_PASS_PHRASE", "XTRACE_EMBEDDING_MODEL_PATH"]
        if api_key_override is None:
            required.append("XTRACE_API_KEY")
        if inf:
            required.append("INFERENCE_API_KEY")
        env = _require_env(required)

    if not json_out:
        console.print(f"[dim]KB[/]: [bold blue]{kb_id}[/]  •  [dim]k[/]: [bold]{k}[/]")
        console.print(Rule(style="dim"))

    effective_api_key = resolve_api_key_override(api_key_override, env, console=console)

    # load components
    with _maybe_status("Using preloaded XTrace components…", enable=not json_out):
        pre = get_pre()
        Retriever_        = pre.Retriever
        XTraceIntegration = pre.XTraceIntegration
        ExecutionContext  = pre.ExecutionContext
        Embedding         = pre.Embedding
        Path              = pre.Path
        pkl               = pre.pkl
        np                = pre.np

        # graceful fallback if running outside the session (one-shot mode)
        if not all([Retriever_, XTraceIntegration, ExecutionContext, Embedding, Path, pkl, np]):
            from xtrace_sdk.x_vec.retrievers.retriever import Retriever as _SR
            from xtrace_sdk.integrations.xtrace import XTraceIntegration as _XI
            from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext as _EC
            from xtrace_sdk.x_vec.inference.embedding import Embedding as _EM
            from pathlib import Path as _Path
            import pickle as _pkl
            import numpy as _np

            Retriever_        = Retriever_        or _SR
            XTraceIntegration = XTraceIntegration or _XI
            ExecutionContext  = ExecutionContext  or _EC
            Embedding         = Embedding         or _EM
            Path              = Path              or _Path
            pkl               = pkl               or _pkl
            np                = np                or _np

    # create compute + retriever (use effective_api_key)
    with _maybe_status("Initializing retriever…", enable=not json_out):
        assert XTraceIntegration is not None
        _shared = get_integration()
        if _shared is not None and not api_key_override:
            integration = _shared
            _owned = False
        else:
            integration = XTraceIntegration(
                org_id=env["XTRACE_ORG_ID"],
                api_key=effective_api_key,
                api_url=env["XTRACE_API_URL"],
            )
            _owned = True
        try:
            # Reuse in-session execution context if available; otherwise load from disk
            exec_context = get_exec_context(load_if_missing=False)
            if exec_context is None:
                assert ExecutionContext is not None  # for mypy type narrowing
                exec_context = ExecutionContext.load_from_disk(
                    env["XTRACE_PASS_PHRASE"], env["XTRACE_EXECUTION_CONTEXT_PATH"]
                )

            # Reuse in-session embedding model if available; otherwise load from disk
            embedding_model = get_embed_model(load_if_missing=False)
            if embedding_model is None:
                assert Path is not None  # for mypy
                embed_bytes = Path(env["XTRACE_EMBEDDING_MODEL_PATH"]).read_bytes()
                assert pkl is not None  # for mypy
                obj = pkl.loads(embed_bytes)
                # support both formats: raw object OR {"embedding": obj}
                embedding_model = obj.get("embedding") if isinstance(obj, dict) else obj

            assert Retriever_ is not None
            retriever = Retriever_(exec_context, integration)
        except Exception:
            # ensure we don't leak the underlying aiohttp session on init errors
            if _owned:
                try:
                    asyncio.run(integration.close())
                except Exception:
                    pass
            raise

    with _maybe_status("Searching…", enable=not json_out):
        try:
            assert np is not None
            assert embedding_model is not None
            async def _search() -> Any:
                bits = [int(x) for x in np.asarray(await embedding_model.bin_embed(query)).tolist()]
                return await retriever.nn_search_for_ids(bits, k, kb_id=kb_id)
            ids = asyncio.run(_search())
            ids = _to_py_int_list(ids)
        except Exception as e:
            msg, code = extract_server_error(e)
            if json_out:
                _json_error(msg, code)
            if msg:
                status = f" ({code})" if code is not None else ""
                console.print(f"[red]Search failed{status}:[/] {msg}")
            else:
                console.print(f"[red]Search failed:[/] {e}")
            raise typer.Exit(1)
    try:
        import numpy as _np
        if isinstance(ids, _np.ndarray):
            ids = [int(x) for x in ids.tolist()]
        else:
            ids = [int(x) for x in ids]
    except Exception:
        ids = [int(x) for x in ids]
    if not json_out:
        console.print(f"[bold green]Successfully retrieved chunks[/]: {ids}")

    # decrypt contexts
    with _maybe_status("Decrypting results…", enable=not json_out):
        contexts = asyncio.run(retriever.retrieve_and_decrypt(ids, kb_id=kb_id))

    # format context for optional inference
    assert Retriever_ is not None
    formatted = Retriever_.format_context([c["chunk_content"] for c in contexts])

    # Build table for human mode
    if not json_out:
        table = Table(show_header=True, header_style="bold", expand=True, show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Content")
        for cid, ctx in zip(ids, contexts, strict=False):
            content_full = _clean_full(str(ctx.get("chunk_content", "") or ""))
            table.add_row(str(cid), f"\"{content_full}\"")
        console.print(table)

    result_payload = {
        "kb_id": kb_id,
        "query": query,
        "k": k,
        "ids": ids,
        "chunks": [
            {"id": int(cid), "content": _clean_full(str(ctx.get("chunk_content", "") or ""))}
            for cid, ctx in zip(ids, contexts, strict=False)
        ],
    }

    try:
        # inference
        if inf:
            if not json_out:
                console.print(Rule(style="dim"))
                console.print(f"[dim]Running inference[/]: [bold]{inf}[/]  •  [dim]model[/]: [bold]{model}[/]")

            from xtrace_sdk.x_vec.inference.llm import InferenceClient
            inference_client = InferenceClient(
                inference_provider=inf,
                model_name=cast(str, model),
                api_key=env["INFERENCE_API_KEY"],
            )

            inference_text = ""
            if json_out:
                # Prefer non-streaming if available; fallback to capturing streamed output.
                try:
                    out = inference_client.query(
                        query=query,
                        context=formatted,
                        stream=False,
                        print_response=False,
                    )
                    inference_text = (out or "").strip() if isinstance(out, str) else (str(out) if out is not None else "")
                except TypeError:
                    from contextlib import redirect_stdout, redirect_stderr
                    buf = io.StringIO()
                    try:
                        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
                            inference_client.query(
                                query=query,
                                context=formatted,
                                stream=True,
                                print_response=True,
                            )
                        inference_text = buf.getvalue().strip()
                    except Exception as e:
                        # from ._utils._errors import extract_server_error
                        msg, code = extract_server_error(e)
                        typer.echo(json.dumps({"error": {"message": msg or str(e), "status": code}}, ensure_ascii=False))
                        raise typer.Exit(1)
                except Exception as e:
                    # from ._utils._errors import extract_server_error
                    msg, code = extract_server_error(e)
                    typer.echo(json.dumps({"error": {"message": msg or str(e), "status": code}}, ensure_ascii=False))
                    raise typer.Exit(1)
            else:
                # human mode: stream to console
                try:
                    inference_client.query(
                        query=query,
                        context=formatted,
                        stream=True,
                        print_response=True,
                    )
                except Exception as e:
                    msg, code = extract_server_error(e)
                    if msg:
                        status = f" ({code})" if code is not None else ""
                        console.print(f"[red]Inference failed{status}:[/] {msg}")
                    else:
                        console.print(f"[red]Inference failed:[/] {e}")
                    raise typer.Exit(1)

            if json_out:
                result_payload["inference"] = {
                    "provider": inf,
                    "model": model,
                    "output": inference_text,
                }

        if json_out:
            try:
                typer.echo(json.dumps(result_payload, ensure_ascii=False))
            except Exception as e:
                _json_error(str(e), None)
            return
    finally:
        if _owned:
            try:
                asyncio.run(integration.close())
            except Exception:
                pass

# Alias
app.command(
    "query",
    help="Alias of 'retrieve'. Same signature and behavior."
)(retrieve)
