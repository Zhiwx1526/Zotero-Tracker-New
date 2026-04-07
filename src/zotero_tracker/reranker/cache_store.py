from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

SCHEMA_VERSION = 1
META_FILENAME = "corpus_embed_meta.json"
VECTORS_FILENAME = "corpus_embed_vectors.npz"


def build_fingerprint(text: str, model: str, text_format_version: str) -> str:
    raw = f"{text}\n{model}\n{text_format_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_dir: Path, model: str, text_format_version: str) -> dict[str, dict[str, Any]]:
    meta_path = cache_dir / META_FILENAME
    vectors_path = cache_dir / VECTORS_FILENAME
    if not meta_path.exists() or not vectors_path.exists():
        return {}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            return {}
        if meta.get("model") != model:
            logger.info("embedding 缓存模型变更，重建缓存。")
            return {}
        if meta.get("text_format_version") != text_format_version:
            logger.info("embedding 文本版本变更，重建缓存。")
            return {}

        vectors_data = np.load(vectors_path)
        vectors = np.asarray(vectors_data["vectors"], dtype=np.float32)
        items = meta.get("items", {})

        cache: dict[str, dict[str, Any]] = {}
        for item_key, item_meta in items.items():
            row = int(item_meta["vector_row"])
            if row < 0 or row >= len(vectors):
                continue
            cache[item_key] = {
                "fingerprint_sha256": item_meta["fingerprint_sha256"],
                "embedding": vectors[row],
            }
        return cache
    except Exception as exc:
        logger.warning(f"读取 embedding 缓存失败，将重建缓存：{exc}")
        return {}


def save_cache(
    cache_dir: Path,
    cache_items: dict[str, dict[str, Any]],
    model: str,
    text_format_version: str,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache_items)

    if keys:
        vectors = np.vstack([np.asarray(cache_items[k]["embedding"], dtype=np.float32) for k in keys])
    else:
        vectors = np.empty((0, 0), dtype=np.float32)

    items_meta = {}
    now = datetime.now(timezone.utc).isoformat()
    for idx, key in enumerate(keys):
        items_meta[key] = {
            "fingerprint_sha256": cache_items[key]["fingerprint_sha256"],
            "vector_row": idx,
            "updated_at": now,
        }

    meta = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "text_format_version": text_format_version,
        "items": items_meta,
    }

    meta_path = cache_dir / META_FILENAME
    vectors_path = cache_dir / VECTORS_FILENAME
    tmp_meta_path = cache_dir / (META_FILENAME + ".tmp")
    tmp_vectors_path = cache_dir / (VECTORS_FILENAME + ".tmp")

    tmp_meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(tmp_vectors_path, "wb") as f:
        np.savez_compressed(f, vectors=vectors)

    os.replace(tmp_vectors_path, vectors_path)
    os.replace(tmp_meta_path, meta_path)
