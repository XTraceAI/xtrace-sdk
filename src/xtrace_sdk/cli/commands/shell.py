from __future__ import annotations
import typer
from ..shell import run_shell
from ..state import in_shell
from rich.console import Console
from typing import NoReturn

console = Console()
app = typer.Typer(name="shell", help="Open the interactive XTrace shell.")

@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context)-> NoReturn:
    if in_shell():
        console.print("[yellow]You're already in the shell. Type 'quit' to exit.[/]")
        raise typer.Exit(0)
    raise typer.Exit(run_shell())