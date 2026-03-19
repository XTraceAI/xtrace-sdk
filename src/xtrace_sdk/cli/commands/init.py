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
        s = str(exc) or ""
        # only rewrite the padding error if this call-site told us it's the load-context flow
        if context == "load_exec_context" and "Padding is incorrect" in s:
            msg = "Decryption failed. Check your pass phrase and execution context id."
        else:
            msg = s or "Unexpected error."
    if not msg:
        assert isinstance(exc, Exception)
        net = _network_error_details(exc)
        if net:
            msg = f"[dim]Network error:[/] {net}"
            

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
            if prompt_overwrite and not _ask_confirm(f"{p.resolve()} exists. Overwrite?", default=False):
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
) -> None:
    """
    Ensure .env contains the minimal keys to recognize the execution context.
    Merge with any existing values; only set/overwrite the keys we know now.
    """
    current = _load_env_dict(target) if target.exists() else {}
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

    merged = {**current, **updates}
    _mkdirp(target.parent)
    _atomic_write_text(target, _dump_env_dict(merged))
    os.environ.update(merged)

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
    cur = _load_env_dict(target) if target.exists() else {}
    cur.update({k: v for k, v in kv.items() if v is not None})
    _mkdirp(target.parent)
    _atomic_write_text(target, _dump_env_dict(cur))

def _ask_int(
    label: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    max_digits: int | None = 6,
    warn_threshold: int | None = None,
    warn_text: str = "Large values may significantly slow operations. Continue?"
) -> int:
    while True:
        s = _ask_text(label)
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

def _ask_text(label: str, default: str | None = None) -> str:
    if not sys.stdin.isatty() or questionary is None or _qstyle_text is None:
        return typer.prompt(label, default=default or "")
    kwargs = {"qmark": " ▶", "style": _qstyle_text}
    if default is not None:
        kwargs["default"] = default
    ans = questionary.text(label, **kwargs).unsafe_ask()
    if ans is None:
        raise typer.Abort()
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

def _ask_secret(label: str) -> str:
    # Use Questionary password to mask input; fall back to secret_prompt
    if not sys.stdin.isatty() or questionary is None or _qstyle_pw is None:
        return secret_prompt(label)
    ans = questionary.password(label, qmark=" ▶", style=_qstyle_text).unsafe_ask()
    if ans is None:
        raise typer.Abort()
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
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v
    return data

def _dump_env_dict(env: dict[str, str]) -> str:
    lines = [f"{k}={v}" for k, v in env.items()]
    return "\n".join(lines) + "\n"

@app.command("init", help="Initializes XTrace SDK")
def init(
    homomorphic_client: str = typer.Option(None, "--homomorphic-client", "-H", case_sensitive=False, help="paillier | paillier_lookup"),
    #embedding_length: int = typer.Option(None, "--embedding-length", "-d", help="Dimension of (future) binary vectors you plan to use"),
    #key_length: int = typer.Option(None, "--key-length", "-k", help="Homomorphic key length in bits (>= 1024)"),
    env_file: Annotated[Path, typer.Option(..., "--env-file", "-f", help="Path to '.env'")] = Path(".env"),
    admin: bool = typer.Option(False, "--admin", help="Prompt for XTrace Admin key and save it into the .env"),
    inference: bool = typer.Option(False, "--inference", help="Prompt for inference key and save it into the .env")
) -> None:
    try:
        if homomorphic_client:
            hc  = homomorphic_client.lower()
            if hc and hc not in {"paillier", "paillier_lookup"}:
                raise typer.BadParameter("homomorphic-client must be 'paillier' or 'paillier_lookup'")
        # if key_length and key_length < 1024:
        #     raise typer.BadParameter("key-length should be at least 1024 bits for security.")
        # if embedding_length and key_length and embedding_length > key_length:
        #     raise typer.BadParameter("embedding-length cannot be greater than key-length.")

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
            _atomic_write_text(target_env, _dump_env_dict({}))

        # if exec already exists... replace it?
        exec_file = Path("data/exec_context")
        if exec_file.exists():
            console.print(Panel.fit(
                "[yellow]An execution context already exists at[/] "
                f"[bold]{exec_file.resolve()}[/]\n\n"
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

        if not regenerate_context:
            target_env = _resolve_env_target(env_file, prompt_overwrite=False)
            # if .env is missing or missing the exec keys, ask for passphrase once and write the block.
            needs_write = (not target_env.exists())
            if not needs_write:
                env_d = _load_env_dict(target_env)
                needs_write = not all(k in env_d for k in ("XTRACE_EXECUTION_CONTEXT_PATH", "XTRACE_PASS_PHRASE"))

            if needs_write:
                assumed_exec_path = str((Path("data") / "exec_context").resolve())
                console.print("[dim]Your .env is missing execution-context info for the existing context.[/]")
                # prompt once so later steps (embedding) don't fail on missing env
                passphrase = _ask_secret("Enter Pass Phrase to save into .env:")
                try:
                    _ensure_env_exec_block(
                        target_env,
                        exec_path=assumed_exec_path,
                        api_url=_API_URL,
                        passphrase=passphrase,
                        # TODO: need api keys?
                    )
                    env_written = True
                    console.print("[dim]Wrote minimal execution-context settings to .env.[/]")
                except Exception as e:
                    _graceful_fail(e, default_code=3)

        # if embedding model already exists... replace it?
        embed_file = Path("data/embed_model")
        if embed_file.exists():
            console.print(Panel.fit(
                "[yellow]An embedding model already exists at[/] "
                f"[bold]{embed_file.resolve()}[/]\n\n"
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
        
        DATA_DIR   = Path("data")
        EXEC_PATH  = DATA_DIR / "exec_context"
        EMBED_PATH = DATA_DIR / "embed_model"

        _mkdirp(DATA_DIR)
        # if setting execution context (both if user chose to replace existing and if no existing exec found)
        if regenerate_context:
            with safe_status("Loading environment…"):
                embedding_model = None
                if not regenerate_embedding and embed_file.exists():
                    file_content = embed_file.read_bytes()
                    embedding_model = pkl.loads(file_content)
            # --- input xtrace credentials + store immediately ---
            console.print("[bold green]Enter XTrace Credentials:[/]")
            xtrace_api_key    = _ask_secret("Enter XTrace API Key:")
            xtrace_org_id     = _ask_text("Enter XTrace Org ID:")
            admin_key_value = None
            if admin:
                admin_key_value = _ask_secret("Enter XTrace Admin key (will be saved to .env):")
            if inference: 
                    inference_api_key = _ask_secret("Enter Inference API Key:")

            # target_env = _resolve_env_target(env_file, prompt_overwrite=False)
            try:
                current_env = _load_env_dict(target_env) if target_env.exists() else {}

                early_updates: dict[str, str] = {
                    "XTRACE_API_URL": _API_URL,
                    "XTRACE_API_KEY": xtrace_api_key,
                    "XTRACE_ORG_ID": xtrace_org_id,
                }
                if admin and admin_key_value:
                    early_updates["XTRACE_ADMIN_KEY"] = admin_key_value
                if inference and inference_api_key:
                    early_updates["XTRACE_INFERENCE_API_KEY"] = inference_api_key
                    # keeps tokenizers from forking too many threads by default
                    early_updates["TOKENIZERS_PARALLELISM"] = "false"

                merged_env = {**current_env, **early_updates}
                _mkdirp(target_env.parent)
                _atomic_write_text(target_env, _dump_env_dict(merged_env))
                os.environ.update(merged_env)
                env_written = True
                console.print("[dim]Saved XTrace credentials to .env.[/]")
            except Exception as e:
                _graceful_fail(e, default_code=3)
            console.print(Rule(style="dim"))
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
                        #target_env = _resolve_env_target(env_file, prompt_overwrite=False)
                        try:
                            _ensure_env_exec_block(
                                target_env,
                                exec_path=str(EXEC_PATH.resolve()),
                                api_url=_API_URL,
                                passphrase=passphrase,
                                api_key=xtrace_api_key,
                                org_id=xtrace_org_id,
                            )
                            env_written = True
                        except Exception as e:
                            _graceful_fail(e, default_code=3)
                    except (PermissionError, KeyboardInterrupt, typer.Abort) as e:
                        _graceful_fail(e, context="load_exec_context")
                    except Exception as e:
                        # network errors, 4xx/5xx, bad passphrase/decrypt, etc.
                        _graceful_fail(e, default_code=3, context="load_exec_context")
                    finally:
                        try:
                            asyncio.run(integration.close())
                        except Exception:
                            pass
                if not regenerate_embedding and embedding_model is not None:
                    ec_dim = exec_context.embed_len()
                    if int(embedding_model.dim) != ec_dim:
                        raise typer.BadParameter(
                            f"Existing embedding model dim ({embedding_model.dim}) does not match execution context dim ({ec_dim}). "
                            "Either replace the embedding model or load a matching execution context."
                        )
                console.print("[dim]Context loaded successfully.[/]")
                console.print(Rule(style="dim"))

            if set_type == execSetType.new:
                if not homomorphic_client: 
                    homomorphic_client = ask_homo_client_type(default=homoClient.paillier_lookup)

                if not regenerate_embedding:
                    # NOTE: if key not valid reprompt. also, enforce numeric-only input and limit length of input.
                    # NOTE: if key length long (large integer), warn user about loss of performance
                    # NOTE: if supported by typer, add an instruction to the key length input, telling the user that smaller key lengths are faster but less secure.

                    assert embedding_model is not None
                    min_required = max(int(embedding_model.dim), 1024)
                    key_length = _ask_int(
                        f"Set key length (min. {min_required}) [smaller = faster, larger = more secure]:",
                        min_value=min_required,
                        max_digits=5,
                        warn_threshold=4096,
                        warn_text="Very large keys can slow encryption and increase memory. Continue?"
                    )
                    assert embedding_model is not None
                    _validate_keys(int(key_length), int(embedding_model.dim))
                    embedding_length = int(embedding_model.dim)
                else: 
                    # NOTE: if key not valid reprompt. also, enforce numeric-only input and limit length of input.
                    # NOTE: if key length long (large integer), warn user about loss of performance
                    # NOTE: if supported by typer, add an instruction to the key length input, telling the user that smaller key lengths are faster but less secure.
                    key_length = _ask_int(
                        "Set key length (min. 1024) [smaller = faster, larger = more secure]:",
                        min_value=1024,
                        max_digits=5,
                        warn_threshold=4096,
                        warn_text="Very large keys can slow encryption and increase memory. Continue?"
                    )
                    embedding_length = _ask_int(
                        f"Set dimension of embeddings you plan to store (≤ {key_length}):",
                        min_value=1,
                        max_value=key_length,   # enforce upper bound -> re-prompt if exceeded
                        max_digits=5
                    )
                    _validate_keys(int(key_length), int(embedding_length))
                passphrase = _ask_passphrase_confirm("Create Pass Phrase:", "Confirm Pass Phrase:")

                with safe_status("Generating new execution context…"):
                    try:
                        _mkdirp(EXEC_PATH.parent)
                        exec_context = ExecutionContext.create(
                            passphrase=passphrase,
                            homomorphic_client_type=homomorphic_client,
                            embedding_length=int(embedding_length),
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
                        #target_env = _resolve_env_target(env_file, prompt_overwrite=False)
                        try:
                            _ensure_env_exec_block(
                                target_env,
                                exec_path=str(EXEC_PATH.resolve()),
                                api_url=_API_URL,
                                passphrase=passphrase,
                                api_key=xtrace_api_key,
                                org_id=xtrace_org_id,
                            )
                            env_written = True
                        except Exception as e:
                            _graceful_fail(e, default_code=3)
                        set_exec_context_now(exec_context, EXEC_PATH)
                    except (KeyboardInterrupt, typer.Abort) as e:
                        _graceful_fail(e)
                    except Exception as e:
                        _graceful_fail(e, default_code=3)
                    finally:
                        try:
                            asyncio.run(integration.close())
                        except Exception:
                            pass
                console.print(f"[dim]Context created with id:[/] [bold magenta]{exec_context.id}[/]")
                console.print("[dim]** Save your id to access your context in the future. **[/]")
                console.print(Rule(style="dim"))
        elif regenerate_embedding: 
            with safe_status("Loading environment…"):
                import dotenv
                dotenv.load_dotenv()
                required = ["XTRACE_PASS_PHRASE", "XTRACE_EXECUTION_CONTEXT_PATH"]
                env_vars = _require_env(required)
                passphrase  = env_vars["XTRACE_PASS_PHRASE"]
                exec_path = env_vars["XTRACE_EXECUTION_CONTEXT_PATH"]
                exec_context = ExecutionContext.load_from_disk(passphrase, exec_path)
                set_exec_context_now(exec_context, exec_path)
        
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
                        # keeps tokenizers from forking too many threads by default
                        updates["TOKENIZERS_PARALLELISM"] = "false"

                if updates:
                    _merge_env_kv(target_env, updates)
                    env_written = True
                    console.print("[dim]Saved key(s) to .env.[/]")
                console.print(Rule(style="dim"))
            except Exception as e:
                _graceful_fail(e, default_code=3)
        
        if regenerate_embedding: 
            console.print("[bold green]Set Embedding Model: [/][dim]Data uploaded with one model must be retrieved with the same model.[/]")
            provider = ask_embed_type(default=embedType.sentence_transformer)
            if provider.name == 'openai':
                openai_key = os.getenv("OPENAI_API_KEY") or _ask_secret("Enter OpenAi Api Key:")
                os.environ["OPENAI_API_KEY"] = openai_key
                try:
                    _merge_env_kv(target_env, {"OPENAI_API_KEY": openai_key})
                    #console.print("[dim]Saved key to .env.[/]")
                except Exception as e:
                    _graceful_fail(e, default_code=3)
            if provider.name == "ollama" and not _ollama_running("http://localhost:11434"):
                console.print("[yellow]Could not reach Ollama at http://localhost:11434[/] "
                  "[dim]Tip: open the Ollama app and enable 'Allow connections from other devices', "
                  "or run: `ollama serve`[/]")
            model_name = _ask_text("Provide model name:")
            ok = False
            while not ok:
                try:
                    if exec_context is None:
                        raise RuntimeError("Execution context not loaded")
                    with safe_status("Loading new embedding model..."):
                        embedding_model = Embedding(
                            provider=provider.name,          # "sentence_transformer" | "ollama" | "openai"
                            model_name=model_name,
                            dim=exec_context.embed_len(),
                        )
                        ok = True
                except (KeyboardInterrupt, typer.Abort):
                    raise
                except Exception as e:
                    console.print(Panel.fit(f"[red]Failed to load embedding model[/]: {e}", border_style="red"))
                    # re-prompt loop
                    new_name = _ask_text("Provide a different model name (or Ctrl+C to abort):")
                    while not new_name.strip():
                        console.print("[red]Model name cannot be empty.[/]")
                        new_name = _ask_text("Provide a different model name (or Ctrl+C to abort):")
                    model_name = new_name.strip()

            output_file = EMBED_PATH
            _mkdirp(output_file.parent)
            with safe_status("Saving model instance to disk..."):
                try:
                    output_file.write_bytes(pkl.dumps(embedding_model))
                except PermissionError as e:
                    _graceful_fail(e)
                except Exception as e:
                    _graceful_fail(e)
            console.print("[dim]Model loaded successfully.[/]")
            try:
                #target_env = _resolve_env_target(env_file, prompt_overwrite=False)
                current_env = _load_env_dict(target_env) if target_env.exists() else {}
                embed_updates = {
                    "XTRACE_EMBEDDING_MODEL_PATH": str(EMBED_PATH.resolve()),
                }
                provider_kv = {
                    "XTRACE_EMBEDDING_PROVIDER": provider.name,
                    "XTRACE_EMBEDDING_MODEL_NAME": model_name,
                }
                merged_env = {**current_env, **embed_updates, **provider_kv}
                _mkdirp(target_env.parent)
                _atomic_write_text(target_env, _dump_env_dict(merged_env))
                os.environ.update(merged_env)
                env_written = True
            except Exception as e:
                _graceful_fail(e, default_code=3)
            set_embed_model_now(embedding_model, EMBED_PATH)
            console.print("[dim]Path written to .env.[/]")
            console.print(Rule(style="dim"))
            
        exec_path = str(EXEC_PATH.resolve())
        embed_path = str(EMBED_PATH.resolve())

        console.print("[bold green]Setup Successful![/]")
        console.print(f"[dim]Execution context[/]: {exec_path} " + ("[dim](kept existing)[/]" if not regenerate_context else "[dim](set new)[/]"))
        console.print(f"[dim]Embedding model[/]: {embed_path} " + ("[dim](kept existing)[/]" if not regenerate_embedding else "[dim](set new)[/]"))
        if env_written: 
            console.print(f"[dim]Wrote env[/]:         {target_env.resolve()}")
        console.print()
        console.print("[dim]Next:[/]  xtrace create-kb --help   •   xtrace load --help")
    except (KeyboardInterrupt, typer.Abort) as e:
        _graceful_fail(e)
    except PermissionError as e:
        _graceful_fail(e)
    except Exception as e:
        _graceful_fail(e, default_code=3)
