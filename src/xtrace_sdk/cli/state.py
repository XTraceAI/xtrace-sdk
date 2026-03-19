from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Iterator
import os
from contextlib import contextmanager
from pathlib import Path as _Path

try:
    from rich.console import Console
except Exception:
    Console = None  # type: ignore[assignment,misc]

# --- SESSION CACHE (in-memory for current process only) -----------------------
_SESSION: dict[str, Any] = {}  # NEW

def session_get(key: str, default: Any = None) -> Any:  # NEW
    return _SESSION.get(key, default)

def session_set(key: str, value: Any) -> None:  # NEW
    _SESSION[key] = value

def clear_session_key(key: str) -> None:  # NEW
    _SESSION.pop(key, None)

def get_admin_key_cached(*, json_out: bool, human_prompt_fn: Any) -> str:  # NEW
    """
    Return the admin key for this *xtrace* session.
    1) If already cached in-memory -> return it.
    2) Else, call your existing util get_admin_key (env -> prompt) and cache it.
    """
    cached = session_get("ADMIN_KEY")
    if isinstance(cached, str) and cached:
        return cached

    from .commands._utils._admin_key import get_admin_key as _raw_get_admin_key
    key = _raw_get_admin_key(json_out=json_out, human_prompt_fn=human_prompt_fn)
    if key:  # only cache non-empty
        session_set("ADMIN_KEY", key)
    return key

@contextmanager
def _status(console: Console | None, message: str) -> Iterator[None]:
    if not console:
        yield
        return
    try:
        with console.status(message, spinner="dots"):
            yield
    except TypeError:
        # older rich fallback
        console.print(f"[dim]{message}[/]")
        yield

@dataclass
class Preloaded:
    SimpleRetriever: Any | None = None
    ParallelRetriever: Any | None = None
    RetrieverBase: Any | None = None
    XTraceIntegration: Any | None = None
    ExecutionContext: Any | None = None
    Embedding: Any | None = None
    DataLoaderBase: Any | None = None
    Path: Any | None = None
    pkl: Any | None = None
    np: Any | None = None
    errors: Any | None = None

_pre: Preloaded = Preloaded(errors=[])

def preload_components(console: Console|None) -> None:
    """Import once per session; safe to call multiple times."""
    # If we already imported core pieces, skip
    if _pre.ExecutionContext and _pre.Embedding:
        return
    with _status(console, "Preloading XTrace components…"):
        try:
            # Core classes
            from xtrace_sdk.x_vec.retrievers.retriever import Retriever as SimpleRetriever
            from xtrace_sdk.x_vec.retrievers.retriever import Retriever as ParallelRetriever
            from xtrace_sdk.x_vec.retrievers.retriever import Retriever as RetrieverBase
            from xtrace_sdk.integrations.xtrace import XTraceIntegration
            from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext
            from xtrace_sdk.x_vec.inference.embedding import Embedding
            from xtrace_sdk.x_vec.data_loaders.loader import DataLoader as DataLoaderBase
            # Standard libs / heavy deps used across commands
            from pathlib import Path as _Path
            import pickle as _pkl
            import numpy as _np
            
            _pre.SimpleRetriever = SimpleRetriever
            _pre.ParallelRetriever = ParallelRetriever
            _pre.RetrieverBase = RetrieverBase
            _pre.XTraceIntegration = XTraceIntegration
            _pre.ExecutionContext = ExecutionContext
            _pre.Embedding = Embedding
            _pre.DataLoaderBase = DataLoaderBase
            _pre.Path = _Path
            _pre.pkl = _pkl
            _pre.np = _np
        except Exception as e:
            (_pre.errors or []).append(f"Preload error: {e!r}")

def get_pre() -> Preloaded:
    """Access the cached imports; commands can grab classes from here."""
    return _pre

@dataclass
class CLIState:
    loaded: bool = False
    exec_context: Any = None
    embed_model: Any = None
    exec_path: Any = None
    embed_path: Path | None = None
    notes: list[str] | None = None
    integration: Any = None  # shared XTraceIntegration for shell sessions

state = CLIState(notes=[])

def load_once(console: Console | None = None) -> None:
    """Best-effort warmup: load .env, execution context, embedding model if present."""
    if state.loaded:
        return
    # .env
    
    try:
        import dotenv
        dotenv.load_dotenv()
    except Exception:
        pass

    exec_path = os.getenv("XTRACE_EXECUTION_CONTEXT_PATH")
    embed_path = os.getenv("XTRACE_EMBEDDING_MODEL_PATH")

    #with _status(console, "Importing XTrace SDK modules…"):
    ExecutionContext = None
    try:
        from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext  # type: ignore
    except Exception as e:
        state.notes = (state.notes or []) + [f"Could not import ExecutionContext: {e!r}"]

    # load exec context
    if exec_path and "ExecutionContext" in locals() and locals()["ExecutionContext"]:
        with _status(console, f"Loading execution context from {exec_path}…"):
            try:
                passphrase = os.getenv("XTRACE_PASS_PHRASE")
                if passphrase:
                    state.exec_context = locals()["ExecutionContext"].load_from_disk(passphrase, exec_path)
                    state.exec_path = Path(exec_path)
                else:
                    state.notes = (state.notes or []) + ["No XTRACE_PASS_PHRASE set; cannot load execution context."]
            except Exception as e:
                state.notes = (state.notes or []) + [f"Failed to load execution context: {e!r}"]

    # load embedding model
    if embed_path:
        with _status(console, f"Loading embedding model from {embed_path}…"):
            try:
                import pickle as pkl
                p = Path(embed_path)
                if p.exists():
                    state.embed_model = pkl.loads(p.read_bytes())
                    state.embed_path = p
                else:
                    state.notes = (state.notes or []) + [f"Embedding model not found at {p}."]
            except Exception as e:
                state.notes = (state.notes or []) + [f"Failed to load embedding model: {e!r}"]

    state.loaded = True

def readiness_messages() -> list[str]:
    msgs = []
    if not state.exec_context:
        msgs.append("No execution context found. Run: init")
    if not state.embed_model:
        msgs.append("No embedding model found. Run: init")
    if not msgs:
        msgs.append("XTrace SDK session is ready.")
    return msgs + (state.notes or [])

def get_exec_context(load_if_missing: bool = True) -> Any:
    if state.exec_context or not load_if_missing:
        return state.exec_context
    load_once()
    return state.exec_context

def get_embed_model(load_if_missing: bool = True) -> Any:
    if state.embed_model or not load_if_missing:
        return state.embed_model
    load_once()
    return state.embed_model

def set_exec_context_now(ec: object, path: str | _Path) -> None:
    """Update the current shell session with a fresh execution context."""
    state.exec_context = ec
    state.exec_path = _Path(path)
    state.loaded = True 

def set_embed_model_now(model: object, path: str | _Path) -> None:
    """Update the current shell session with a fresh embedding model."""
    state.embed_model = model
    state.embed_path = _Path(path)
    state.loaded = True

def get_integration() -> Any:
    """Return the shared XTraceIntegration for the current shell session, or None."""
    return state.integration

def set_integration(integration: Any) -> None:
    """Store (or clear) the shared XTraceIntegration for the current shell session."""
    state.integration = integration

def in_shell() -> bool:
    return bool(session_get("IN_SHELL", False))

def set_in_shell(flag: bool) -> None:
    session_set("IN_SHELL", flag)