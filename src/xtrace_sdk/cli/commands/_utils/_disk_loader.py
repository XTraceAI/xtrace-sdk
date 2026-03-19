from __future__ import annotations
"""Minimal file loader for the CLI load/upsert-file commands.

Supports plain text (.txt, .md), JSON (.json), and CSV (.csv).
Returns a list of simple string objects so that _to_chunk_collection works
with `getattr(doc, "page_content", doc)`.
"""
from pathlib import Path  # noqa: E402


_SUPPORTED = {".txt", ".md", ".json", ".csv"}


class LocalDiskConnector:
    @staticmethod
    def load_data_from_file(file_path: str) -> list[str]:
        p = Path(file_path)
        ext = p.suffix.lower()
        if ext not in _SUPPORTED:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(_SUPPORTED))}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        return [text]
