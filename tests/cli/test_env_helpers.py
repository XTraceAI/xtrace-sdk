"""Tests for the .env file helpers in init.py — verifying additive-only behavior."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from xtrace_sdk.cli.commands.init import (
    _append_env_keys,
    _env_keys_in_file,
    _load_env_dict,
)


# ── _load_env_dict ──────────────────────────────────────────────────────────


def test_load_env_dict_skips_comments(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(dedent("""\
        # This is a comment
        FOO=bar
        # BAZ=should_skip
        HELLO=world
    """))
    assert _load_env_dict(env) == {"FOO": "bar", "HELLO": "world"}


def test_load_env_dict_empty_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("")
    assert _load_env_dict(env) == {}


# ── _env_keys_in_file ──────────────────────────────────────────────────────


def test_env_keys_in_file_includes_commented(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(dedent("""\
        ACTIVE=1
        # COMMENTED_OUT=old_value
        #NO_SPACE=also_commented
        # pure comment with no equals
        ANOTHER=2
    """))
    keys = _env_keys_in_file(env)
    assert keys == {"ACTIVE", "COMMENTED_OUT", "NO_SPACE", "ANOTHER"}


def test_env_keys_in_file_nonexistent(tmp_path: Path) -> None:
    assert _env_keys_in_file(tmp_path / "nope") == set()


# ── _append_env_keys ───────────────────────────────────────────────────────


def test_append_preserves_comments_and_blanks(tmp_path: Path) -> None:
    """The original bug: comments and blank lines were being destroyed."""
    env = tmp_path / ".env"
    original = dedent("""\
        # Database settings
        # DB_HOST=localhost
        DB_PORT=5432

        # API config
        API_URL=https://example.com
    """)
    env.write_text(original)

    skipped = _append_env_keys(env, {"NEW_KEY": "new_value"})

    result = env.read_text()
    # original content is fully preserved
    assert "# Database settings" in result
    assert "# DB_HOST=localhost" in result
    assert "DB_PORT=5432" in result
    assert "# API config" in result
    assert "API_URL=https://example.com" in result
    # new key is appended
    assert "NEW_KEY=new_value" in result
    assert skipped == []


def test_append_skips_active_key_conflict(tmp_path: Path) -> None:
    """If a key already exists as an active var, skip it and report conflict."""
    env = tmp_path / ".env"
    env.write_text("FOO=original\nBAR=keep\n")

    skipped = _append_env_keys(env, {"FOO": "overwritten", "NEW": "added"})

    result = env.read_text()
    # FOO was NOT overwritten
    assert "FOO=original" in result
    assert "FOO=overwritten" not in result
    # NEW was added
    assert "NEW=added" in result
    assert skipped == ["FOO"]


def test_append_skips_commented_key_conflict(tmp_path: Path) -> None:
    """If a key exists only as a commented-out var, skip it — user may have
    intentionally disabled it."""
    env = tmp_path / ".env"
    env.write_text("# MY_VAR=old\nOTHER=1\n")

    skipped = _append_env_keys(env, {"MY_VAR": "new_val", "BRAND_NEW": "yes"})

    result = env.read_text()
    assert "# MY_VAR=old" in result
    assert "MY_VAR=new_val" not in result
    assert "BRAND_NEW=yes" in result
    assert skipped == ["MY_VAR"]


def test_append_to_empty_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("")

    skipped = _append_env_keys(env, {"A": "1", "B": "2"})

    result = env.read_text()
    assert "A=1" in result
    assert "B=2" in result
    assert skipped == []


def test_append_creates_file_if_missing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    assert not env.exists()

    skipped = _append_env_keys(env, {"X": "42"})

    assert env.exists()
    assert "X=42" in env.read_text()
    assert skipped == []


def test_append_no_updates_no_write(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "KEEP=me\n"
    env.write_text(original)

    skipped = _append_env_keys(env, {})

    assert env.read_text() == original
    assert skipped == []


def test_append_all_conflicts_no_write(tmp_path: Path) -> None:
    """If every key conflicts, the file should not be modified at all."""
    env = tmp_path / ".env"
    original = "A=1\nB=2\n"
    env.write_text(original)

    skipped = _append_env_keys(env, {"A": "new", "B": "new"})

    assert env.read_text() == original
    assert set(skipped) == {"A", "B"}


def test_append_with_overwrite_active(tmp_path: Path) -> None:
    """When overwrite_active=True, active keys should be appended (resulting in
    duplicate lines — the last one wins for most .env loaders)."""
    env = tmp_path / ".env"
    env.write_text("FOO=old\n")

    skipped = _append_env_keys(env, {"FOO": "new"}, overwrite_active=True)

    result = env.read_text()
    assert "FOO=old" in result
    assert "FOO=new" in result
    assert skipped == []
