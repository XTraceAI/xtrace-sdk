from __future__ import annotations
import typer

from .commands.init import app as init_app
from .commands.version import app as version_app
from .commands.shell import app as shell_app
from .commands.knowledgebase import app as kb_app

# x-vec commands
from .commands.load import app as load_app
from .commands.head import app as head_app
from .commands.fetch import app as fetch_app
from .commands.retrieve import app as retrieve_app
from .commands.upsert import app as upsert_app
from .commands.upsert_file import app as upsert_file_app

app = typer.Typer(no_args_is_help=True, add_completion=False, help="XTrace SDK CLI")

# ── Shared top-level commands ───────────────────────────────────────────────
app.add_typer(version_app)
app.add_typer(init_app)
app.add_typer(shell_app, name="shell")

# ── Knowledge-base admin (shared across x-vec and x-mem) ───────────────────
app.add_typer(kb_app, name="kb", help="Knowledge base management (create, delete, list, describe).")

# ── x-vec subgroup ─────────────────────────────────────────────────────────
xvec_app = typer.Typer(
    no_args_is_help=True,
    help="Encrypted vector search commands (load, retrieve, head, fetch, upsert).",
)
xvec_app.add_typer(load_app)
xvec_app.add_typer(retrieve_app)
xvec_app.add_typer(head_app)
xvec_app.add_typer(fetch_app)
xvec_app.add_typer(upsert_app)
xvec_app.add_typer(upsert_file_app)
app.add_typer(xvec_app, name="xvec")

# ── x-mem subgroup (placeholder — coming soon) ──────────────────────────────
xmem_app = typer.Typer(
    no_args_is_help=True,
    help="Encrypted memory database commands (coming soon).",
)
app.add_typer(xmem_app, name="xmem")

if __name__ == "__main__":
    app()
