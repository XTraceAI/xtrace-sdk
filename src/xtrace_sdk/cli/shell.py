from __future__ import annotations
import os
import re
import shlex
import click 
from rich.console import Console
from rich.rule import Rule

from typing import Iterable
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent, Completion
from typing import Any
from typer import Typer

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import Completer, Completion, PathCompleter, WordCompleter, merge_completers
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.shortcuts import CompleteStyle
    _HAS_PT = True
except Exception:
    _HAS_PT = False

import asyncio
from .state import in_shell, set_in_shell, load_once, readiness_messages, preload_components, get_pre, set_integration

console = Console()

def _get_app() -> Typer:
    from .main import app
    return app

def _registered_commands() -> list[str]:
    """
    Collect top-level command names from the Typer app.
    Keeps it simple: suggest 'help', 'quit' and the top-level subcommands.
    """
    app = _get_app()
    names = set(["help", "quit"])
    click_app: Any = app
    if isinstance(click_app, click.Group):
        for name in click_app.commands:
            names.add(name)
    return sorted(names)

class HybridCompleter(Completer):
    """
    Command-aware completion:
      - First token -> complete top-level commands.
      - Any subsequent token -> complete filesystem paths.
      - Also complete a path if the *only* token looks like a path (e.g., starting with '/', './', '../', '~').
    """
    def __init__(self) -> None:
        self._cmds = WordCompleter(_registered_commands(), ignore_case=True, sentence=True)

    def _iter_path_completions(self, token: str) -> Iterable[Completion]:
        """
        Minimal readline-like path completion for the last token.
        Expands ~, handles directories, and appends '/' for directories.
        """
        # treat blank token (e.g., just after a space) as current directory
        raw = token or ""
        expanded = os.path.expanduser(raw)

        # if it's a directory with trailing slash, keep as base dir
        if expanded.endswith(os.path.sep):
            base_dir = expanded
            prefix = ""
        else:
            base_dir = os.path.dirname(expanded) or "."
            prefix = os.path.basename(expanded)

        try:
            entries = os.listdir(base_dir)
        except Exception:
            return  # no completions

        for name in entries:
            if prefix and not name.startswith(prefix):
                continue
            full = os.path.join(base_dir, name)
            # Reconstruct the candidate in the *same style* the user typed (respect ~ if present)
            # Use os.path.relpath if the user didn't start from '/', '~', or '.'
            display = name + (os.path.sep if os.path.isdir(full) else "")
            # What we insert should replace the last token exactly:
            # - If user typed '~', keep '~' where possible
            candidate = full
            if raw.startswith("~"):
                # Convert full back to a ~-based path if it lives under home
                home = os.path.expanduser("~")
                if full.startswith(home):
                    candidate = "~" + full[len(home):]
            elif raw.startswith("./") or raw.startswith("../") or raw.startswith("."):
                # Keep relative style if they started relative
                candidate = os.path.relpath(full, os.getcwd())
            # If it's a directory, add trailing slash so further Tab keeps completing inside it
            if os.path.isdir(full) and not candidate.endswith(os.path.sep):
                candidate += os.path.sep

            yield Completion(
                text=candidate,
                start_position=-(len(raw)),
                display=display
            )

    def get_completions(self, document: Document, complete_event: CompleteEvent)-> Iterable[Completion]:
        text = document.text_before_cursor
        tokens = text.split()

        # Determine last token safely (what we're completing)
        last_token = tokens[-1] if tokens else ""

        # If there is exactly one token so far:
        # - complete commands, unless it clearly looks like a path start
        if len(tokens) <= 1:
            looks_like_path = last_token.startswith(("/", ".", "~")) or ("/" in last_token)
            if looks_like_path:
                yield from self._iter_path_completions(last_token)
            else:
                # If the cursor is after a space (`"load "`), tokens may be 1 (["load"]) but last_token == "load".
                # Only complete commands when cursor is still inside that first token.
                # If the user typed a space (text.endswith(' ')), we switch to path completion for the *next* arg.
                if text.endswith(" "):
                    # complete first argument as a path
                    yield from self._iter_path_completions("")
                else:
                    # still typing the command
                    yield from self._cmds.get_completions(document, complete_event)
            return

        # 2+ tokens -> complete paths for arguments
        yield from self._iter_path_completions(last_token)


def _run_one(line: str) -> int:
    args = shlex.split(line)
    app = _get_app()
    try:
        app(prog_name="xtrace", args=args, standalone_mode=False)
        return 0
    except click.ClickException as e:
        e.show()
        return 2
    except click.Abort:
        console.print("[yellow]Aborted[/]")
        return 130
    except SystemExit as e:
        return int(e.code or 1)
    except KeyboardInterrupt:
        console.print("[dim]^C[/]")  # cancel current command, keep shell
        return 130
    except Exception as e:
        console.print(f"[red]Unexpected error:[/] {e!r}")
        return 1

def run_shell() -> int:
    if in_shell():
        console.print("[yellow]Already in a shell. Type 'quit' to exit.[/]")
        return 0

    set_in_shell(True)
    _integration = None
    try:
        load_once(console=console)
        preload_components(console=console)

        # Create one persistent XTraceIntegration for the whole shell session
        import dotenv
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
        _org_id  = os.getenv("XTRACE_ORG_ID")
        _api_key = os.getenv("XTRACE_API_KEY")
        _api_url = (os.getenv("XTRACE_API_URL") or "https://api.production.xtrace.ai").rstrip("/")
        if _org_id and _api_key:
            pre = get_pre()
            _XI = pre.XTraceIntegration
            if _XI is None:
                from xtrace_sdk.integrations.xtrace import XTraceIntegration as _XI
            assert _XI is not None
            _integration = _XI(org_id=_org_id, api_key=_api_key, api_url=_api_url)
            set_integration(_integration)

        console.print(Rule(style="dim"))
        for msg in readiness_messages():
            console.print(f"[dim]-[/] {msg} [dim]-[/]")
        console.print("[dim]Type 'help' for help, 'quit' to exit.[/]")
        console.print(Rule(style="dim"))

        # session with history + completion
        session: PromptSession | None = None
        if _HAS_PT:
            history_path = os.path.expanduser("~/.xtrace_cli_history")
            completer = HybridCompleter()
            session = PromptSession(
                history=FileHistory(history_path),
                completer=completer,
                complete_style=CompleteStyle.READLINE_LIKE,  
                complete_while_typing=False,                
            )

        while True:
            try:
                if session:
                    prompt_text = HTML("<ansiblue><b>&gt;</b></ansiblue> ")
                    line = session.prompt(prompt_text).strip()
                else:
                    # if prompt_toolkit unavailable
                    line = input("> ").strip()
            except EOFError:
                console.print("")  # newline on Ctrl+D
                break
            except KeyboardInterrupt:
                console.print("")  # ignore ^C at the prompt
                continue

            if not line:
                continue
            if line in {"quit", "exit", ":q"}:
                break
            if line in {"help", "--help", "-h"}:
                _run_one("--help")
                continue

            _run_one(line)

        console.print("[dim]Session closed.[/]")
        return 0
    finally:
        set_in_shell(False)
        if _integration is not None:
            try:
                asyncio.run(_integration.close())
            except Exception:
                pass
        set_integration(None)
