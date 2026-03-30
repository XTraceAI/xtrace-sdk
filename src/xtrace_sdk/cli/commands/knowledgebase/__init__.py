from __future__ import annotations
import typer
from .create import app as create_app
from .delete import app as delete_app
from .describe import app as describe_app
from .list import app as list_app

app = typer.Typer()

app.add_typer(create_app)
app.add_typer(delete_app)
app.add_typer(describe_app)
app.add_typer(list_app)