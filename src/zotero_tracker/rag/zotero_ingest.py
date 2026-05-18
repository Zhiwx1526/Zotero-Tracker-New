from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from loguru import logger

from ..dedupe import corpus_dedupe_keys
from ..protocol import CorpusPaper


_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass
class ZoteroRagDoc:
    item_key: str
    doi: str | None
    title: str
    text: str
    collection_path: str | None


def _clean_text(text: str) -> str:
    t = (text or "").strip()
    t = _MULTI_SPACE_RE.sub(" ", t)
    return t


def build_zotero_rag_docs(
    corpus: list[CorpusPaper],
    *,
    enable_dedupe: bool,
    min_abstract_len: int,
) -> tuple[list[ZoteroRagDoc], dict[str, int]]:
    min_abs = max(0, int(min_abstract_len))
    docs: list[ZoteroRagDoc] = []
    seen_doi: set[str] = set()
    seen_fp: set[tuple[str, str]] = set()
    dropped = 0

    for c in corpus:
        keys = corpus_dedupe_keys(c)
        if enable_dedupe:
            if keys.doi_key and keys.doi_key in seen_doi:
                dropped += 1
                continue
            if keys.norm_abstract_len >= min_abs:
                fp = (keys.title_fp, keys.abstract_fp)
                if fp in seen_fp:
                    dropped += 1
                    continue

        combined = _clean_text(
            "\n".join(
                [
                    f"标题：{c.title}",
                    f"摘要：{c.abstract}",
                    f"集合路径：{'; '.join(c.paths) if c.paths else '未知'}",
                ]
            )
        )
        if not combined:
            continue
        docs.append(
            ZoteroRagDoc(
                item_key=c.item_key,
                doi=c.doi,
                title=c.title,
                text=combined,
                collection_path=(c.paths[0] if c.paths else None),
            )
        )
        if keys.doi_key:
            seen_doi.add(keys.doi_key)
        if keys.norm_abstract_len >= min_abs:
            seen_fp.add((keys.title_fp, keys.abstract_fp))

    stats = {"input": len(corpus), "kept": len(docs), "dedupe_dropped": dropped}
    logger.info(
        "Zotero RAG 入库文献：输入 {}，保留 {}，去重丢弃 {}。",
        stats["input"],
        stats["kept"],
        stats["dedupe_dropped"],
    )
    return docs, stats


def zotero_doc_signature(docs: list[ZoteroRagDoc]) -> str:
    parts: list[str] = []
    for d in docs:
        parts.append(f"{d.item_key}|{d.doi or ''}|{len(d.text)}")
    return str(hash(tuple(parts)))
