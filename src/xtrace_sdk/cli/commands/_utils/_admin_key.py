from __future__ import annotations
import os
import sys
import json
from getpass import getpass
import typer
from typing import Any

def get_admin_key(json_out: bool, human_prompt_fn: Any) -> str:
    """
    Returns admin key. If XTRACE_ADMIN_KEY exists, use it.
    In --json mode, prompt to STDERR and confirm capture to STDERR.
    In human mode, use your existing secret prompt function.
    """
    env_key = os.getenv("XTRACE_ADMIN_KEY")
    if env_key:
        return env_key

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
