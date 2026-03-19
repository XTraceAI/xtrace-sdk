from __future__ import annotations
import typer

def secret_prompt(label: str, optional: bool = False) -> str:
    """
    Ask for a secret, masking with '*' when prompt_toolkit is present.
    Falls back to typer.prompt(hide_input=True) otherwise.
    """
    try:
        # optional dependency path
        from prompt_toolkit import prompt as pt_prompt
        value = pt_prompt(f"{label}: ", is_password=True)
    except Exception:
        # Fallback: no masking chars, just hidden input
        default = "" if optional else None  # allow empty if optional
        value = typer.prompt(label, hide_input=True, default=default)

    if not optional and not value:
        typer.echo("This value is required.")
        return secret_prompt(label, optional=False)
    return value
