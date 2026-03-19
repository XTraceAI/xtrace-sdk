from __future__ import annotations
import typer
from importlib.metadata import version, PackageNotFoundError

app = typer.Typer()

@app.command("version", help="Prints version number.")
def version_cmd() -> None:
    try:
        typer.echo(version("xtrace-ai-sdk"))
    except PackageNotFoundError:
        typer.echo("0.0.0")