from __future__ import annotations
from pathlib import Path
import os
import typer
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel
from contextlib import contextmanager
from ._utils._prompt import secret_prompt

import socket
import ssl
from typing import Optional
from typing import Iterator
from typing import Tuple, Type
from typing import cast


import asyncio
import pickle as pkl
import click
from enum import StrEnum
import questionary
import sys
from ._utils._errors import extract_server_error
from xtrace_sdk.cli.state import set_exec_context_now, set_embed_model_now

from typing import Annotated
from pathlib import Path
from typing import NoReturn


app = typer.Typer()
console = Console()

_API_URL = "https://api.production.xtrace.ai"

def _walk_causes(exc: BaseException)-> Iterator[BaseException]:
    """Yield exc and its chained causes/contexts (cause or context)."""
    seen = set()
    cur: BaseException | None = exc
    while cur and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = (cur.__cause__ or cur.__context__)

def _network_error_details(exc: Exception) -> str | None:
    """
    If this looks like a connectivity/DNS/SSL/timeout error, return a short
    human summary like 'gaierror: [Errno 8] ...' or 'ConnectionRefusedError: ...'.
    """
    try:
        import aiohttp  
        AIOHTTP_TYPES: tuple[type[BaseException], ...] = (
            aiohttp.ClientConnectorError,
            aiohttp.ClientConnectionError,
        )
    except Exception:
        AIOHTTP_TYPES = cast(tuple[type[BaseException], ...], ())

    try:
        import httpx
        HTTPX_TYPES: tuple[type[BaseException], ...] = (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.NetworkError,
        )
    except Exception:
        HTTPX_TYPES = cast(tuple[type[BaseException], ...], ())

    text_needles = [
        "Failed to establish a new connection",
        "Cannot connect to host",
        "Connection refused",
        "getaddrinfo failed",
        "timed out",
        "SSL",
        "Temporary failure in name resolution",
        "Name or service not known",
        "nodename nor servname provided",
    ]

    for e in _walk_causes(exc):
        if isinstance(e, (ConnectionRefusedError, TimeoutError, socket.gaierror, socket.timeout, ssl.SSLError)):
            return f"{e.__class__.__name__}: {e}" if str(e) else e.__class__.__name__
        if isinstance(e, AIOHTTP_TYPES + HTTPX_TYPES):
            return f"{e.__class__.__name__}: {e}" if str(e) else e.__class__.__name__
        s = str(e)
        if s and any(n.lower() in s.lower() for n in text_needles):
            return f"{e.__class__.__name__}: {s}"
    return None

def _graceful_fail(exc: BaseException, *, default_code: int = 1, context: str | None = None) -> NoReturn: 
    if isinstance(exc, (typer.Abort, KeyboardInterrupt)):
        console.print("[yellow]Canceled by user.[/]")
        raise typer.Exit(1)

    if isinstance(exc, PermissionError):
        console.print("[red]Permission denied.[/] [dim]Check write access and try again.[/]")
        raise typer.Exit(4)

    # prefer structured message if available
    assert isinstance(exc, Exception)
    msg, code = extract_server_error(exc)

    if not msg:
        assert isinstance(exc, Exception)
        net = _network_error_details(exc)
        if net:
            msg = f"[dim]Network error:[/] {net}"
    if not msg:
        s = str(exc) or ""
        # only rewrite the padding error if this call-site told us it's the load-context flow
        if context == "load_exec_context" and "Padding is incorrect" in s:
            msg = "Decryption failed. Check your pass phrase and execution context id."
        else:
            msg = s or "Unexpected error."

    console.print(f"[red]Error:[/] {msg}")
    raise typer.Exit(code or default_code)

def _mkdirp(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        _graceful_fail(e)

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except PermissionError as e:
        _graceful_fail(e)
    except Exception as e:
        # clean up tmp if something odd happened
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        finally:
            _graceful_fail(e)

@contextmanager
def safe_status(message: str)-> Iterator[None]:
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
            + "\n[dim]Tip: run `xtrace init` and set execution context to create .env[/]"
        )
        raise typer.Exit(2)
    return {n: os.environ[n] for n in names}

def _resolve_env_target(env_file: Path, prompt_overwrite: bool = True) -> Path:
    p = env_file.expanduser()

    if p.exists():
        if p.is_dir():
            if prompt_overwrite and not _ask_confirm(f"Generate .env file in directory {p.resolve()}?", default=True):
                console.print("[yellow]Aborted. No files were changed.[/]")
                raise typer.Exit(1)
            return p / ".env"
        else:
            if p.name != ".env":
                console.print(
                    "[red]The path you provided is a file that is not named '.env'.[/]\n"
                    "[dim]Pass a directory or a path to a file named exactly '.env' (e.g., /path/to/.env).[/]"
                )
                raise typer.Exit(2)
            if prompt_overwrite and not _ask_confirm(f"{p.resolve()} exists. Update it?", default=True):
                console.print("[yellow]Aborted. No files were changed.[/]")
                raise typer.Exit(1)
            return p

    if p.suffix and p.name != ".env":
        console.print(
            "[red]The path looks like a file that is not named '.env'.[/]\n"
            "[dim]Pass a directory or a path to a file named exactly '.env' (e.g., /path/to/.env).[/]"
        )
        raise typer.Exit(2)

    if p.name == ".env":
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if prompt_overwrite and not _ask_confirm(f"Generate .env file in directory {p.resolve()}?", default=True):
        console.print("[yellow]Aborted. No files were changed.[/]")
        raise typer.Exit(1)
    p.mkdir(parents=True, exist_ok=True)
    return p / ".env"

def _ensure_env_exec_block(
    target: Path,
    *,
    exec_path: str,
    api_url: str,
    passphrase: str | None = None,
    api_key: str | None = None,
    org_id: str | None = None,
    context_id: str | None = None,
) -> None:
    """
    Ensure .env contains the minimal keys to recognize the execution context.
    Merge with any existing values; only set/overwrite the keys we know now.
    """
    updates: dict[str, str] = {
        "XTRACE_API_URL": api_url,
        "XTRACE_EXECUTION_CONTEXT_PATH": exec_path,
    }
    if api_key:
        updates["XTRACE_API_KEY"] = api_key
    if org_id:
        updates["XTRACE_ORG_ID"] = org_id
    if passphrase:
        updates["XTRACE_PASS_PHRASE"] = passphrase
    if context_id:
        updates["XTRACE_CONTEXT_ID"] = context_id

    skipped = _append_env_keys(target, updates)
    _warn_env_conflicts(skipped)
    # update process env with only the keys that were actually written
    written = {k: v for k, v in updates.items() if k not in skipped}
    os.environ.update(written)

# --- validation helpers ---
def _as_int(name: str, value: str | int) -> int:
    try:
        return int(value)
    except Exception:
        raise typer.BadParameter(f"{name} must be an integer.")

def _validate_keys(key_len: int, embed_len: int | None = None) -> None:
    if key_len < 1024:
        raise typer.BadParameter("key-length should be at least 1024 bits for security.")
    if embed_len is not None and embed_len > key_len:
        raise typer.BadParameter("embedding-length cannot be greater than key-length.")
    
def _merge_env_kv(target: Path, kv: dict[str, str]) -> None:
    filtered = {k: v for k, v in kv.items() if v is not None}
    skipped = _append_env_keys(target, filtered)
    _warn_env_conflicts(skipped)

def _ask_int(
    label: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    max_digits: int | None = 6,
    default: int | None = None,
    warn_threshold: int | None = None,
    warn_text: str = "Large values may significantly slow operations. Continue?"
) -> int:
    while True:
        s = _ask_text(label, default=str(default) if default is not None else None)
        if not s.isdigit():
            console.print("[red]Please enter digits only.[/]")
            continue
        if max_digits is not None and len(s) > max_digits:
            console.print(f"[red]Too many digits (>{max_digits}).[/]")
            continue
        val = int(s)
        if min_value is not None and val < min_value:
            console.print(f"[red]Value must be at least {min_value}.[/]")
            continue
        if max_value is not None and val > max_value:
            console.print(f"[red]Value must be at most {max_value}.[/]")
            continue
        if warn_threshold is not None and val >= warn_threshold and not _ask_confirm_indented(f"{val} selected. {warn_text}", default=False):
            continue
        return val


def _ollama_running(url: str = "http://localhost:11434") -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(url + "/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

# --- questionary style (tames default highlighting / colors) ---
try:
    from questionary import Style 
    _Q_ORANGE = "#ffaf00"

    _qstyle_select = Style([
        ("qmark",       "fg:#cccccc"),
        ("question",    "fg:#cccccc"),  
        ("pointer",     "fg:#cccccc"),
        ("selected",    ""),             
        ("highlighted", "fg:#cccccc"),
        ("instruction", "fg:#aaaaaa"),
        ("text",        ""),
        ("answer",      f"fg:{_Q_ORANGE}"), 
    ])

    _qstyle_text = Style([
        ("qmark",       "fg:#cccccc"),
        ("question",    "fg:#cccccc"),
        ("pointer",     "fg:#cccccc"),
        ("selected",    ""),
        ("highlighted", "fg:#cccccc"),
        ("instruction", "fg:#aaaaaa"),
        ("text",        ""),
        ("answer",      "fg:#cccccc"),   
    ])

    _qstyle_pw = Style([
        ("qmark",       "fg:#cccccc"),
        ("question",    "fg:#cccccc"),
        ("pointer",     "fg:#cccccc"),
        ("selected",    ""),
        ("highlighted", "fg:#cccccc"),
        ("instruction", "fg:#aaaaaa"),
        ("text",        ""),
        ("answer",      ""),           
    ])
except Exception:
    _qstyle_select = None
    _qstyle_text   = None
    _qstyle_pw     = None

def _must_answer(ans: str | None, what: str = "selection") -> str:
    # convert a None (Ctrl+C / Esc) into a clean abort
    if ans is None:
        raise typer.Abort()
    return ans

def _ask_text(label: str, default: str | None = None, *, allow_empty: bool = False) -> str:
    while True:
        if not sys.stdin.isatty() or questionary is None or _qstyle_text is None:
            ans = typer.prompt(label, default=default or "")
        else:
            kwargs = {"qmark": " ▶", "style": _qstyle_text}
            if default is not None:
                kwargs["default"] = default
            ans = questionary.text(label, **kwargs).unsafe_ask()
        if ans is None:
            raise typer.Abort()
        if not allow_empty and not ans.strip():
            console.print("[red]Input cannot be empty.[/]")
            continue
        return ans

def _ask_confirm(message: str, default: bool = False) -> bool:
    if not sys.stdin.isatty() or questionary is None or _qstyle_select is None:
        return typer.confirm(message, default=default)
    ans = questionary.confirm(message, default=default, qmark="▶", style=_qstyle_select).unsafe_ask()
    if ans is None:
        raise typer.Abort()
    return bool(ans)

def _ask_confirm_indented(message: str, default: bool = False) -> bool:
    if not sys.stdin.isatty() or questionary is None or _qstyle_select is None:
        return typer.confirm(message, default=default)
    ans = questionary.confirm(message, default=default, qmark=" ▶", style=_qstyle_select).unsafe_ask()
    if ans is None:
        raise typer.Abort()
    return bool(ans)

def _ask_secret(label: str, *, allow_empty: bool = False) -> str:
    while True:
        if not sys.stdin.isatty() or questionary is None or _qstyle_pw is None:
            ans = secret_prompt(label)
        else:
            ans = questionary.password(label, qmark=" ▶", style=_qstyle_text).unsafe_ask()
        if ans is None:
            raise typer.Abort()
        if not allow_empty and not ans.strip():
            console.print("[red]Input cannot be empty.[/]")
            continue
        return ans

def _ask_passphrase_confirm(
    label: str = "Create Pass Phrase:",
    confirm_label: str = "Confirm Pass Phrase:",
    *,
    allow_empty: bool = False
) -> str:
    """
    Ask for a pass phrase twice and loop until they match.
    Ctrl+C / Esc still respected via _ask_secret (raises typer.Abort).
    """
    while True:
        p1 = _ask_secret(label)
        if not allow_empty and not p1:
            console.print("[red]Pass phrase cannot be empty.[/]")
            continue
        p2 = _ask_secret(confirm_label)
        if p1 != p2:
            console.print("[yellow]Pass phrases do not match. Please try again.[/]")
            continue
        return p1

class homoClient(StrEnum):
    paillier_lookup = "paillier_lookup"
    paillier = "paillier"

def ask_homo_client_type(default: homoClient | None = None) -> homoClient:
    # if not a TTY or questionary missing, fallback to typed prompt
    if not sys.stdin.isatty() or questionary is None:
        return typer.prompt(
            "Select homomorphic client type:",
            type=click.Choice([e.value for e in homoClient], case_sensitive=False),
            default=(default.value if default else None),
        )
    # Arrow-key select
    answer = questionary.select(
        "Select homomorphic client type:",
        choices=[e.value for e in homoClient],
        qmark=" ▶",
        pointer=" ➤",
        use_shortcuts=False,
        style=_qstyle_select,
    ).unsafe_ask()
    answer = _must_answer(answer, "homomorphic client type")
    return homoClient(answer)

class execSetType(StrEnum):
    new = "New"
    existing = "Existing"

def ask_exec_set_type(default: execSetType | None = None) -> execSetType:
    # if not a TTY or questionary missing, fallback to typed prompt
    if not sys.stdin.isatty() or questionary is None:
        return typer.prompt(
            "Create New or Load Existing?",
            type=click.Choice([e.value for e in execSetType], case_sensitive=False),
            default=(default.value if default else None),
        )
    # Arrow-key select
    answer = questionary.select(
        "Create New or Load Existing?",
        choices=[e.value for e in execSetType],
        qmark=" ▶",
        pointer=" ➤",
        use_shortcuts=False,
        style=_qstyle_select,
    ).unsafe_ask()
    answer = _must_answer(answer, "execution context selection")
    return execSetType(answer)

class embedType(StrEnum):
    sentence_transformer = "Sentence Transformer"
    ollama = "Ollama"
    openai = "Open Ai"

def ask_embed_type(default: embedType | None = None) -> embedType:
    # if not a TTY or questionary missing, fallback to typed prompt
    if not sys.stdin.isatty() or questionary is None:
        return typer.prompt(
            "Choose Embedding Provider:",
            type=click.Choice([e.value for e in embedType], case_sensitive=False),
            default=(default.value if default else None),
        )
    # Arrow-key select
    answer = questionary.select(
        "Choose Embedding Provider:",
        choices=[e.value for e in embedType],
        qmark=" ▶",
        pointer="  ➤",
        use_shortcuts=False,
        style=_qstyle_select,
    ).unsafe_ask()
    answer = _must_answer(answer, "embedding provider")
    return embedType(answer)

def _load_env_dict(path: Path) -> dict[str, str]:
    """Load active (uncommented) key=value pairs from an env file."""
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def _env_keys_in_file(path: Path) -> set[str]:
    """Return all env key names present in the file — both active AND commented-out."""
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        # handle commented-out vars like: # FOO=bar  or  #FOO=bar
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
            if not stripped or "=" not in stripped:
                continue
        k, _ = stripped.split("=", 1)
        keys.add(k.strip())
    return keys


def _append_env_keys(
    target: Path,
    updates: dict[str, str],
    *,
    overwrite_active: bool = False,
) -> list[str]:
    """Append new key=value pairs to an env file, preserving ALL existing content.

    - Never modifies or removes existing lines (comments, blank lines, active vars).
    - If a key already exists as an active var and *overwrite_active* is False, the
      key is skipped and its name is returned in the conflict list so the caller can
      warn the user.
    - If a key appears only in a commented-out line, it is also treated as a conflict
      (the user may have intentionally commented it out).

    Returns the list of key names that were **skipped** due to conflicts.
    """
    if not updates:
        return []

    existing_text = target.read_text(encoding="utf-8") if target.exists() else ""
    active_keys = _load_env_dict(target)
    all_keys = _env_keys_in_file(target)

    skipped: list[str] = []
    new_lines: list[str] = []

    for k, v in updates.items():
        if k in active_keys and not overwrite_active:
            skipped.append(k)
        elif k in all_keys and k not in active_keys:
            # key exists only as a comment — also skip to avoid confusion
            skipped.append(k)
        else:
            new_lines.append(f"{k}={v}")

    if new_lines:
        # ensure we start on a fresh line
        if existing_text and not existing_text.endswith("\n"):
            existing_text += "\n"
        existing_text += "\n".join(new_lines) + "\n"
        _mkdirp(target.parent)
        _atomic_write_text(target, existing_text)

    return skipped


def _warn_env_conflicts(skipped: list[str]) -> None:
    """Print a warning for env keys that were not written due to conflicts."""
    if not skipped:
        return
    console.print(
        "[yellow]The following keys already exist in your .env and were NOT overwritten:[/] "
        + ", ".join(f"[bold]{k}[/]" for k in skipped)
    )
    console.print("[dim]If these values are outdated, update or remove them from your .env manually.[/]")

def _cleanup_failed_exec_context(path: Path) -> None:
    """Remove a locally created exec context that was never saved to remote."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

@app.command("init", help="Initializes XTrace SDK")
def init(
    env_file: Annotated[Path, typer.Option(..., "--env-file", "-f", help="Path to '.env'")] = Path(".env"),
    admin: bool = typer.Option(False, "--admin", help="Prompt for XTrace Admin key and save it into the .env"),
    inference: bool = typer.Option(False, "--inference", help="Prompt for inference key and save it into the .env")
) -> None:
    try:
        homomorphic_client = homoClient.paillier_lookup

        with safe_status("Loading XTrace components…"):
            from xtrace_sdk.x_vec.data_loaders.loader import DataLoader as DataLoaderBase
            from xtrace_sdk.integrations.xtrace import XTraceIntegration
            from xtrace_sdk.x_vec.inference.embedding import Embedding
            from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext

        exec_context: ExecutionContext | None = None
        embedding_model: Embedding | None = None
        passphrase = None
        xtrace_api_key = None
        xtrace_org_id = None
        admin_key_value = None
        inference_api_key = None

        target_env = _resolve_env_target(env_file, prompt_overwrite=False)
        env_written = False
        if not target_env.exists():
            _mkdirp(target_env.parent)
            _atomic_write_text(target_env, "")

        DATA_DIR   = Path("data")
        EXEC_PATH  = DATA_DIR / "exec_context"
        EMBED_PATH = DATA_DIR / "embed_model"
        _mkdirp(DATA_DIR)


        # ── Check existing artefacts ───────────────────────────────────
        _first_run = not EMBED_PATH.exists() and not EXEC_PATH.exists()
        if _first_run:
            console.print(
                "[bold]Configure your local XTrace environment.[/] "
                "[dim]This sets up the credentials, embedding model, and encryption keys needed to use XTrace. "
                "Your pass phrase and encryption keys never leave this machine — only encrypted data is sent to XTrace.[/]"
            )
            console.print(Rule(style="dim"))

        # Embedding model check (asked first since user sets it first)
        if EMBED_PATH.exists():
            console.print(Panel.fit(
                "[yellow]An embedding model already exists at[/] "
                f"[bold]{EMBED_PATH.resolve()}[/]\n\n"
                "[yellow]Any data uploaded to XTrace with one model must be retrieved with the same model.[/]",
                title="Warning: Existing Embedding Model", border_style="yellow"
            ))
            if not _ask_confirm("Replace existing embedding model with a NEW one?", default=False):
                console.print("[yellow]Keeping existing embedding model.[/]")
                regenerate_embedding = False
            else:
                regenerate_embedding = True
            console.print(Rule(style="dim"))
        else:
            regenerate_embedding = True

        # Execution context check
        if EXEC_PATH.exists():
            console.print(Panel.fit(
                "[yellow]An execution context already exists at[/] "
                f"[bold]{EXEC_PATH.resolve()}[/]\n\n"
                "[yellow]Overwriting will generate NEW homomorphic key material.[/]\n"
                "[yellow]Any data uploaded to XTrace with the old keys will be unreadable.[/]",
                title="Warning: Existing Execution Context", border_style="yellow"
            ))
            if not _ask_confirm("Overwrite the existing execution context with a NEW one?", default=False):
                console.print("[yellow]Keeping existing execution context.[/]")
                regenerate_context = False
            else:
                regenerate_context = True
            console.print(Rule(style="dim"))
        else:
            regenerate_context = True

        # ── Load existing artefacts we're keeping ──────────────────────
        if not regenerate_context:
            # ensure .env has exec context info
            needs_write = (not target_env.exists())
            if not needs_write:
                env_d = _load_env_dict(target_env)
                needs_write = not all(k in env_d for k in ("XTRACE_EXECUTION_CONTEXT_PATH", "XTRACE_PASS_PHRASE"))

            if needs_write:
                assumed_exec_path = str(EXEC_PATH.resolve())
                console.print("[dim]Your .env is missing execution-context info for the existing context.[/]")
                passphrase = _ask_secret("Enter Pass Phrase to save into .env:")
                try:
                    _ensure_env_exec_block(
                        target_env,
                        exec_path=assumed_exec_path,
                        api_url=_API_URL,
                        passphrase=passphrase,
                    )
                    env_written = True
                    console.print("[dim]Wrote minimal execution-context settings to .env.[/]")
                except Exception as e:
                    _graceful_fail(e, default_code=3)

            # load from disk (needed so embedding setup can read embed_len)
            with safe_status("Loading existing execution context…"):
                import dotenv
                dotenv.load_dotenv()
                required = ["XTRACE_PASS_PHRASE", "XTRACE_EXECUTION_CONTEXT_PATH"]
                env_vars = _require_env(required)
                passphrase = env_vars["XTRACE_PASS_PHRASE"]
                exec_path_str = env_vars["XTRACE_EXECUTION_CONTEXT_PATH"]
                exec_context = ExecutionContext.load_from_disk(passphrase, exec_path_str)
                set_exec_context_now(exec_context, exec_path_str)

        if not regenerate_embedding and EMBED_PATH.exists():
            with safe_status("Loading existing embedding model…"):
                embedding_model = pkl.loads(EMBED_PATH.read_bytes())
            # ensure .env has embedding model info
            env_d = _load_env_dict(target_env) if target_env.exists() else {}
            if not env_d.get("XTRACE_EMBEDDING_MODEL_PATH"):
                try:
                    embed_updates: dict[str, str] = {
                        "XTRACE_EMBEDDING_MODEL_PATH": str(EMBED_PATH.resolve()),
                    }
                    if hasattr(embedding_model, "provider") and embedding_model.provider:
                        embed_updates["XTRACE_EMBEDDING_PROVIDER"] = embedding_model.provider
                    if hasattr(embedding_model, "model_name") and embedding_model.model_name:
                        embed_updates["XTRACE_EMBEDDING_MODEL_NAME"] = embedding_model.model_name
                    _merge_env_kv(target_env, embed_updates)
                    os.environ.update(embed_updates)
                    env_written = True
                    console.print("[dim]Wrote embedding model settings to .env.[/]")
                except Exception as e:
                    _graceful_fail(e, default_code=3)

        # ── Credentials ────────────────────────────────────────────────
        if not regenerate_context:
            # ensure credentials are present in .env even when keeping existing artefacts
            env_d = _load_env_dict(target_env) if target_env.exists() else {}
            _existing_key = env_d.get("XTRACE_API_KEY", "").strip()
            _existing_org = env_d.get("XTRACE_ORG_ID", "").strip()
            if not _existing_key or not _existing_org:
                console.print("[dim]Your .env is missing XTrace credentials.[/]")
                console.print("[bold green]Enter XTrace Credentials:[/]")
                console.print("[dim]Find your credentials at https://app.xtrace.ai[/]")
                if not _existing_key:
                    xtrace_api_key = _ask_secret("Enter XTrace API Key:")
                if not _existing_org:
                    xtrace_org_id = _ask_text("Enter XTrace Org ID:")
                try:
                    cred_updates: dict[str, str] = {"XTRACE_API_URL": _API_URL}
                    if xtrace_api_key:
                        cred_updates["XTRACE_API_KEY"] = xtrace_api_key
                    if xtrace_org_id:
                        cred_updates["XTRACE_ORG_ID"] = xtrace_org_id
                    _merge_env_kv(target_env, cred_updates)
                    os.environ.update(cred_updates)
                    env_written = True
                    console.print("[dim]Saved XTrace credentials to .env.[/]")
                except Exception as e:
                    _graceful_fail(e, default_code=3)
                console.print(Rule(style="dim"))

        if regenerate_context:
            # detect existing credentials in .env
            existing_env = _load_env_dict(target_env) if target_env.exists() else {}
            _existing_key = existing_env.get("XTRACE_API_KEY", "").strip()
            _existing_org = existing_env.get("XTRACE_ORG_ID", "").strip()

            if _existing_key and _existing_org:
                _masked_key = _existing_key[:6] + "…" if len(_existing_key) > 6 else _existing_key
                console.print(
                    f"[bold green]Existing credentials detected.[/] "
                    f"Organization [bold]{_existing_org}[/], API key [bold]{_masked_key}[/]"
                )
                if _ask_confirm("Use existing credentials?", default=True):
                    xtrace_api_key = _existing_key
                    xtrace_org_id = _existing_org
                    _reused_creds = True
                else:
                    console.print("[bold green]Enter XTrace Credentials:[/]")
                    console.print("[dim]Find your credentials at https://app.xtrace.ai[/]")
                    xtrace_api_key = _ask_secret("Enter XTrace API Key:")
                    xtrace_org_id = _ask_text("Enter XTrace Org ID:")
                    _reused_creds = False
            else:
                console.print("[bold green]Enter XTrace Credentials:[/]")
                console.print("[dim]Find your credentials at https://app.xtrace.ai[/]")
                xtrace_api_key = _ask_secret("Enter XTrace API Key:")
                xtrace_org_id = _ask_text("Enter XTrace Org ID:")
                _reused_creds = False
            admin_key_value = None
            if admin:
                admin_key_value = _ask_secret("Enter XTrace Admin key (will be saved to .env):")
            if inference:
                inference_api_key = _ask_secret("Enter Inference API Key:")

            try:
                early_updates: dict[str, str] = {"XTRACE_API_URL": _API_URL}
                if not _reused_creds:
                    early_updates["XTRACE_API_KEY"] = xtrace_api_key
                    early_updates["XTRACE_ORG_ID"] = xtrace_org_id
                if admin and admin_key_value:
                    early_updates["XTRACE_ADMIN_KEY"] = admin_key_value
                if inference and inference_api_key:
                    early_updates["XTRACE_INFERENCE_API_KEY"] = inference_api_key
                    early_updates["TOKENIZERS_PARALLELISM"] = "false"

                skipped = _append_env_keys(target_env, early_updates)
                _warn_env_conflicts(skipped)
                written = {k: v for k, v in early_updates.items() if k not in skipped}
                os.environ.update(written)
                env_written = True
                console.print("[dim]Saved XTrace credentials to .env.[/]")
            except Exception as e:
                _graceful_fail(e, default_code=3)
            console.print(Rule(style="dim"))

        # ── Embedding Model Setup ──────────────────────────────────────
        if regenerate_embedding:
            console.print("[bold green]Set Embedding Model: [/][dim]Data uploaded with one model must be retrieved with the same model.[/]")
            provider = ask_embed_type(default=embedType.sentence_transformer)
            if provider.name == 'sentence_transformer':
                try:
                    import sentence_transformers  # noqa: F401
                except ImportError:
                    console.print(
                        "[red]sentence-transformers is not installed.[/]\n"
                        "[dim]Install it with:[/]  pip install 'xtrace-ai-sdk\\[embedding]'"
                    )
                    raise typer.Exit(2)
            elif provider.name == 'openai':
                existing_openai_key = os.getenv("OPENAI_API_KEY", "").strip()
                if existing_openai_key:
                    _masked = existing_openai_key[:6] + "…" if len(existing_openai_key) > 6 else existing_openai_key
                    console.print(f"[dim]Using existing OpenAI API key ({_masked}) from environment.[/]")
                    openai_key = existing_openai_key
                else:
                    openai_key = _ask_secret("Enter OpenAI API Key:")
                os.environ["OPENAI_API_KEY"] = openai_key
                try:
                    _merge_env_kv(target_env, {"OPENAI_API_KEY": openai_key})
                except Exception as e:
                    _graceful_fail(e, default_code=3)
            elif provider.name == "ollama" and not _ollama_running("http://localhost:11434"):
                console.print("[yellow]Could not reach Ollama at http://localhost:11434[/] "
                  "[dim]Tip: open the Ollama app and enable 'Allow connections from other devices', "
                  "or run: `ollama serve`[/]")
            model_name = _ask_text("Provide model name:")

            # determine embedding dimension
            if not regenerate_context:
                # dim is fixed by the existing execution context
                assert exec_context is not None
                embedding_dim = exec_context.embed_len()
                console.print(f"[dim]Using dimension {embedding_dim} from existing execution context.[/]")
            else:
                embedding_dim = _ask_int(
                    "Set dimension of embeddings:",
                    min_value=1,
                    max_digits=5,
                )

            # create Embedding with retry on bad model name
            ok = False
            while not ok:
                try:
                    with safe_status("Loading new embedding model..."):
                        embedding_model = Embedding(
                            provider=provider.name,
                            model_name=model_name,
                            dim=embedding_dim,
                        )
                        ok = True
                except (KeyboardInterrupt, typer.Abort):
                    raise
                except (ImportError, ModuleNotFoundError) as e:
                    # missing dependency — not recoverable by changing model name
                    _graceful_fail(e, default_code=3)
                except Exception as e:
                    console.print(Panel.fit(f"[red]Failed to load embedding model[/]: {e}", border_style="red"))
                    new_name = _ask_text("Provide a different model name (or Ctrl+C to abort):")
                    model_name = new_name.strip()

            _mkdirp(EMBED_PATH.parent)
            with safe_status("Saving model instance to disk..."):
                try:
                    EMBED_PATH.write_bytes(pkl.dumps(embedding_model))
                except PermissionError as e:
                    _graceful_fail(e)
                except Exception as e:
                    _graceful_fail(e)
            if provider.name == 'sentence_transformer':
                console.print("[dim]Model loaded successfully.[/]")
            else:
                console.print(f"[dim]Embedding provider set to {provider.value} ({model_name}).[/]")
            try:
                embed_updates = {
                    "XTRACE_EMBEDDING_MODEL_PATH": str(EMBED_PATH.resolve()),
                    "XTRACE_EMBEDDING_PROVIDER": provider.name,
                    "XTRACE_EMBEDDING_MODEL_NAME": model_name,
                }
                skipped = _append_env_keys(target_env, embed_updates)
                _warn_env_conflicts(skipped)
                written = {k: v for k, v in embed_updates.items() if k not in skipped}
                os.environ.update(written)
                env_written = True
            except Exception as e:
                _graceful_fail(e, default_code=3)
            set_embed_model_now(embedding_model, EMBED_PATH)
            console.print("[dim]Path written to .env.[/]")
            console.print(Rule(style="dim"))

        # resolve embedding dim for exec context creation
        embedding_dim_resolved: int | None = None
        if embedding_model is not None:
            embedding_dim_resolved = int(embedding_model.dim)

        # ── Execution Context Setup ────────────────────────────────────
        if regenerate_context:
            # --- set execution context: create new or load existing from remote ---
            console.print("[bold green]Set Execution Context:[/][dim] This is a unique fingerprint to upload/retrieve your data.[/]")
            set_type = ask_exec_set_type(default=execSetType.new)

            if set_type == execSetType.existing:
                context_id = _ask_text("Enter Context Id:")
                passphrase = _ask_secret("Enter Pass Phrase:")
                with safe_status("Loading execution context…"):
                    integration = XTraceIntegration(org_id=xtrace_org_id, api_key=xtrace_api_key, api_url=_API_URL)
                    try:
                        exec_context = asyncio.run(ExecutionContext.load_from_remote(passphrase, context_id, integration))
                        _mkdirp(EXEC_PATH.parent)
                        exec_context.save_to_disk(str(EXEC_PATH))
                        set_exec_context_now(exec_context, EXEC_PATH)
                        try:
                            _ensure_env_exec_block(
                                target_env,
                                exec_path=str(EXEC_PATH.resolve()),
                                api_url=_API_URL,
                                passphrase=passphrase,
                                api_key=xtrace_api_key,
                                org_id=xtrace_org_id,
                                context_id=exec_context.id,
                            )
                            env_written = True
                        except Exception as e:
                            _graceful_fail(e, default_code=3)
                    except (PermissionError, KeyboardInterrupt, typer.Abort) as e:
                        _graceful_fail(e, context="load_exec_context")
                    except Exception as e:
                        _graceful_fail(e, default_code=3, context="load_exec_context")
                    finally:
                        try:
                            asyncio.run(integration.close())
                        except Exception:
                            pass
                # validate dim match with embedding model
                if embedding_dim_resolved is not None:
                    ec_dim = exec_context.embed_len()
                    if embedding_dim_resolved != ec_dim:
                        raise typer.BadParameter(
                            f"Embedding model dim ({embedding_dim_resolved}) does not match execution context dim ({ec_dim}). "
                            "Either replace the embedding model or load a matching execution context."
                        )
                console.print("[dim]Context loaded successfully.[/]")
                console.print(Rule(style="dim"))

            if set_type == execSetType.new:
                assert embedding_dim_resolved is not None
                console.print(f"[dim]Embedding dimension: {embedding_dim_resolved} (must match your embedding model)[/]")
                min_required = max(embedding_dim_resolved + 1, 1024)
                suggested_key = min_required
                key_length = _ask_int(
                    f"Set key length (min. {min_required}) [smaller = faster, larger = more secure]:",
                    min_value=min_required,
                    default=suggested_key,
                    max_digits=5,
                    warn_threshold=4096,
                    warn_text="Very large keys can slow encryption and increase memory. Continue?"
                )
                _validate_keys(int(key_length), embedding_dim_resolved)
                console.print(
                    "[dim]Your pass phrase encrypts all data in this context. "
                    "If forgotten, all data encrypted under this context will be permanently lost.[/]"
                )
                passphrase = _ask_passphrase_confirm("Create Pass Phrase:", "Confirm Pass Phrase:")

                with safe_status("Generating new execution context…"):
                    try:
                        _mkdirp(EXEC_PATH.parent)
                        exec_context = ExecutionContext.create(
                            passphrase=passphrase,
                            homomorphic_client_type=homomorphic_client,
                            embedding_length=embedding_dim_resolved,
                            key_len=int(key_length),
                            path=str(EXEC_PATH)
                        )
                    except (PermissionError, KeyboardInterrupt, typer.Abort) as e:
                        _graceful_fail(e)
                    except Exception as e:
                        _graceful_fail(e, default_code=3)

                    integration = XTraceIntegration(api_key=xtrace_api_key, org_id=xtrace_org_id, api_url=_API_URL)
                    try:
                        asyncio.run(exec_context.save_to_remote(integration))
                        try:
                            _ensure_env_exec_block(
                                target_env,
                                exec_path=str(EXEC_PATH.resolve()),
                                api_url=_API_URL,
                                passphrase=passphrase,
                                api_key=xtrace_api_key,
                                org_id=xtrace_org_id,
                                context_id=exec_context.id,
                            )
                            env_written = True
                        except Exception as e:
                            _graceful_fail(e, default_code=3)
                        set_exec_context_now(exec_context, EXEC_PATH)
                    except (KeyboardInterrupt, typer.Abort) as e:
                        _cleanup_failed_exec_context(EXEC_PATH)
                        _graceful_fail(e)
                    except Exception as e:
                        _cleanup_failed_exec_context(EXEC_PATH)
                        _graceful_fail(e, default_code=3)
                    finally:
                        try:
                            asyncio.run(integration.close())
                        except Exception:
                            pass
                console.print(f"[dim]Context created with id:[/] [bold magenta]{exec_context.id}[/]")
                console.print("[dim]** Save your id to access your context in the future. **[/]")
                console.print(Rule(style="dim"))

        # ── Admin / inference key updates (existing context only) ──────
        if not regenerate_context and (admin or inference):
            console.print("[bold green]Update Keys for Existing Context:[/]")
            try:
                updates: dict[str, str] = {}

                if admin:
                    admin_key_value = _ask_secret("Enter XTrace Admin key (will be saved to .env):")
                    if admin_key_value:
                        updates["XTRACE_ADMIN_KEY"] = admin_key_value

                if inference:
                    inference_api_key = _ask_secret("Enter Inference API Key:")
                    if inference_api_key:
                        updates["XTRACE_INFERENCE_API_KEY"] = inference_api_key
                        updates["TOKENIZERS_PARALLELISM"] = "false"

                if updates:
                    _merge_env_kv(target_env, updates)
                    env_written = True
                    console.print("[dim]Saved key(s) to .env.[/]")
                console.print(Rule(style="dim"))
            except Exception as e:
                _graceful_fail(e, default_code=3)

        # ── Summary ────────────────────────────────────────────────────
        exec_path = str(EXEC_PATH.resolve())
        embed_path = str(EMBED_PATH.resolve())

        console.print("[bold green]Setup Successful![/]")
        console.print(f"[dim]Embedding model[/]:    {embed_path} " + ("[dim](kept existing)[/]" if not regenerate_embedding else "[dim](set new)[/]"))
        console.print(f"[dim]Execution context[/]: {exec_path} " + ("[dim](kept existing)[/]" if not regenerate_context else "[dim](set new)[/]"))
        if env_written:
            console.print(f"[dim]Wrote env[/]:         {target_env.resolve()}")
        console.print()
        console.print("[dim]Next:[/]  xtrace kb create --help   •   xtrace xvec load --help")
    except (KeyboardInterrupt, typer.Abort) as e:
        _graceful_fail(e)
    except PermissionError as e:
        _graceful_fail(e)
    except Exception as e:
        _graceful_fail(e, default_code=3)
