from __future__ import annotations
import os
import sys
import json
from getpass import getpass
import typer
from typing import Any
from rich.console import Console

_stderr_console = Console(stderr=True)

def resolve_api_key_override(
    api_key_override: str | None,
    env: dict[str, str],
    *,
    console: Any = None,
) -> str | None:
    """
    Resolve the effective API key when ``-a`` / ``--api_key`` is used.

    * ``None``  → user didn't pass ``-a`` at all  → fall back to env silently.
    * non-empty → use as-is.
    * empty str → user passed ``-a ""`` → ask whether to use the .env key.
    """
    if api_key_override is None:
        # flag was not supplied
        return env.get("XTRACE_API_KEY")

    if api_key_override.strip():
        return api_key_override

    # user explicitly passed -a with an empty value
    env_key = env.get("XTRACE_API_KEY", "").strip()
    if not env_key:
        c = console or _stderr_console
        c.print("[red]No API key provided via -a and no XTRACE_API_KEY in .env.[/]")
        raise typer.Exit(2)

    masked = env_key[:5] + "…" if len(env_key) > 5 else env_key
    c = console or _stderr_console
    import questionary
    try:
        answer = questionary.confirm(
            f"Use .env API key ({masked})?", default=True, qmark=" ▶"
        ).unsafe_ask()
    except Exception:
        answer = True
    if answer is None:
        raise typer.Abort()
    if not answer:
        c.print("[yellow]Aborted.[/]")
        raise typer.Exit(1)
    return env_key


def get_admin_key(json_out: bool, human_prompt_fn: Any) -> str:
    """
    Returns admin key. If XTRACE_ADMIN_KEY exists, use it.
    In --json mode, prompt to STDERR and confirm capture to STDERR.
    In human mode, use your existing secret prompt function.
    """
    env_key = os.getenv("XTRACE_ADMIN_KEY")
    if env_key:
        return env_key

    _stderr_console.print(
        "[dim]Knowledge base operations require an admin key. "
        "Find yours at https://app.xtrace.ai\n"
        "Tip: set XTRACE_ADMIN_KEY in your .env or run `xtrace init --admin` to skip this prompt.[/]"
    )

    if json_out:
        try:
            key = getpass("Enter XTrace Admin key (input hidden; press Enter): ")
        except (Exception, KeyboardInterrupt):
            typer.echo(json.dumps({"error": {"message": "XTRACE_ADMIN_KEY required", "status": None}}, ensure_ascii=False))
            raise typer.Exit(1)

        if not key:
            typer.echo(json.dumps({"error": {"message": "XTRACE_ADMIN_KEY required", "status": None}}, ensure_ascii=False))
            raise typer.Exit(1)

        print("✓ Admin key captured", file=sys.stderr, flush=True)  # feedback to user, not JSON
        return key

    # human (non-JSON) mode
    return human_prompt_fn("Enter XTrace Admin key")
