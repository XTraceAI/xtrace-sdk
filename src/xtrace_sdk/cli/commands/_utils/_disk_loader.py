"""Minimal file loader with page-based chunking for the CLI.

Supports plain text (.txt, .md), JSON (.json), and CSV (.csv).
Returns a list of string chunks so that _to_chunk_collection works
with `getattr(doc, "page_content", doc)`.

Chunking strategy: approximate one page (~3000 characters) per chunk,
with a small overlap so context isn't lost at boundaries. Splits at
sentence boundaries where possible.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path


_SUPPORTED = {".txt", ".md", ".json", ".csv"}

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# ~3000 chars ≈ one printed page
PAGE_SIZE = 3000
PAGE_OVERLAP = 200


class LocalDiskConnector:
    @staticmethod
    def load_data_from_file(
        file_path: str,
        page_size: int = PAGE_SIZE,
        page_overlap: int = PAGE_OVERLAP,
    ) -> list[str]:
        p = Path(file_path)
        ext = p.suffix.lower()
        if ext not in _SUPPORTED:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(_SUPPORTED))}")
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []

        if ext in (".txt", ".md"):
            return _chunk_into_pages(text, page_size, page_overlap)
        elif ext == ".json":
            return _chunk_json(text, page_size, page_overlap)
        elif ext == ".csv":
            return _chunk_csv(text, page_size, page_overlap)
        return [text.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    parts = _SENTENCE_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def _chunk_into_pages(text: str, page_size: int, overlap: int) -> list[str]:
    """Split text into ~page_size char chunks at sentence boundaries with overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    # If everything fits in one page, return as-is
    if len(text.strip()) <= page_size:
        return [text.strip()]

    pages: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)

        # Single sentence bigger than a page — give it its own chunk
        if s_len > page_size:
            if current:
                pages.append(" ".join(current))
                current = []
                current_len = 0
            pages.append(sentence)
            continue

        if current_len + s_len + 1 > page_size and current:
            pages.append(" ".join(current))

            # Carry overlap: take sentences from the tail of the current page
            if overlap > 0:
                tail: list[str] = []
                tail_len = 0
                for s in reversed(current):
                    if tail_len + len(s) + 1 > overlap:
                        break
                    tail.insert(0, s)
                    tail_len += len(s) + 1
                current = tail
                current_len = tail_len
            else:
                current = []
                current_len = 0

        current.append(sentence)
        current_len += s_len + (1 if current_len > 0 else 0)

    if current:
        pages.append(" ".join(current))

    return [p for p in pages if p.strip()]


def _chunk_json(text: str, page_size: int, overlap: int) -> list[str]:
    """JSON arrays: each element is a chunk. Non-arrays: chunk as text."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _chunk_into_pages(text, page_size, overlap)

    if not isinstance(data, list):
        return _chunk_into_pages(text, page_size, overlap)

    chunks: list[str] = []
    for item in data:
        if isinstance(item, dict):
            chunk = json.dumps(item, ensure_ascii=False)
        elif isinstance(item, str):
            chunk = item
        else:
            chunk = str(item)
        if chunk.strip():
            chunks.append(chunk)
    return chunks if chunks else [text.strip()]


def _chunk_csv(text: str, page_size: int, overlap: int) -> list[str]:
    """Group CSV rows into page-sized chunks. Each row formatted as 'header: value. header: value.'"""
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 2:
        return [text.strip()]

    headers = [h.strip() for h in rows[0]]
    formatted_rows: list[str] = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        parts = [f"{h}: {v.strip()}" for h, v in zip(headers, row, strict=False) if v.strip()]
        formatted_rows.append(". ".join(parts))

    if not formatted_rows:
        return [text.strip()]

    # Group rows into pages
    return _chunk_into_pages("\n\n".join(formatted_rows), page_size, overlap)
