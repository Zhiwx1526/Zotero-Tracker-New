from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TextChunk:
    chunk_id: str
    source_path: str
    title: str
    text: str
    metadata: dict[str, Any] | None = None


def chunk_text(
    *,
    source_path: str,
    title: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    raw = (text or "").strip()
    if not raw:
        return []
    size = max(100, int(chunk_size))
    overlap = max(0, min(int(chunk_overlap), size // 2))
    step = max(1, size - overlap)

    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    while start < len(raw):
        piece = raw[start : start + size].strip()
        if piece:
            chunks.append(
                TextChunk(
                    chunk_id=f"{source_path}::{idx}",
                    source_path=source_path,
                    title=title,
                    text=piece,
                    metadata=None,
                )
            )
            idx += 1
        start += step
    return chunks
