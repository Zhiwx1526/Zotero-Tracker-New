from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
from openai import OpenAI

from .chunking import TextChunk


@dataclass
class RagIndex:
    chunks: list[TextChunk]
    matrix: np.ndarray
    metadata: dict[str, Any]


def _embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    safe_batch_size = max(1, min(int(batch_size), 10))
    if safe_batch_size != int(batch_size):
        logger.warning(
            "RAG embedding_batch_size={} 超出当前接口上限，已自动降为 {}。",
            int(batch_size),
            safe_batch_size,
        )
    vectors: list[list[float]] = []
    for i in range(0, len(texts), safe_batch_size):
        batch = texts[i : i + safe_batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        vectors.extend([item.embedding for item in resp.data])
    return np.asarray(vectors, dtype=np.float32)


def save_index(index_path: str, index: RagIndex) -> None:
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": index.metadata,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "source_path": c.source_path,
                "title": c.title,
                "text": c.text,
                "metadata": c.metadata or {},
            }
            for c in index.chunks
        ],
        "embeddings": index.matrix.tolist(),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_index(index_path: str) -> RagIndex | None:
    p = Path(index_path)
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    chunks = [
        TextChunk(
            chunk_id=item["chunk_id"],
            source_path=item["source_path"],
            title=item.get("title", ""),
            text=item["text"],
            metadata=item.get("metadata", {}) or {},
        )
        for item in raw.get("chunks", [])
    ]
    matrix = np.asarray(raw.get("embeddings", []), dtype=np.float32)
    return RagIndex(chunks=chunks, matrix=matrix, metadata=raw.get("metadata", {}))


def build_index(
    *,
    client: OpenAI,
    embedding_model: str,
    batch_size: int,
    chunks: list[TextChunk],
    metadata: dict[str, Any],
) -> RagIndex:
    texts = [f"{c.title}\n{c.text}".strip() for c in chunks]
    matrix = _embed_texts(client, embedding_model, texts, batch_size)
    return RagIndex(chunks=chunks, matrix=matrix, metadata=metadata)
