from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
from openai import OpenAI

from ..protocol import CorpusPaper
from .chunking import chunk_text
from .ingest import collect_knowledge_chunks
from .store import RagIndex, build_index, load_index, save_index
from .zotero_ingest import build_zotero_rag_docs, zotero_doc_signature


@dataclass
class RagHit:
    source_path: str
    title: str
    score: float
    snippet: str
    item_key: str | None = None
    doi: str | None = None
    collection_path: str | None = None


@dataclass
class RagContext:
    context_text: str
    hits: list[RagHit]


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray([], dtype=np.float32)
    q = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.dot(m, q.T).reshape(-1)


def _hash_inputs(paths: list[str], chunk_size: int, chunk_overlap: int, embedding_model: str) -> str:
    joined = "|".join(sorted(paths)) + f"|{chunk_size}|{chunk_overlap}|{embedding_model}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class RagRetriever:
    def __init__(self, config: Any, openai_client: OpenAI, *, zotero_corpus: list[CorpusPaper] | None = None):
        self.config = config
        self.client = openai_client
        self.rag_cfg = config.executor.get("rag", {})
        self.kb_cfg = config.get("rag", {}).get("knowledge_base", {})
        self.zotero_corpus = zotero_corpus or []
        self.last_index_stats: dict[str, int] = {}
        self._index: RagIndex | None = None

    def _build_or_load_index(self) -> RagIndex | None:
        if self._index is not None:
            return self._index
        paths = [str(p) for p in self.kb_cfg.get("paths", []) if str(p).strip()]
        if not paths:
            logger.info("RAG 未配置知识库路径，跳过。")
            return None
        index_path = str(self.kb_cfg.get("index_path", "cache/rag_index.json"))
        chunk_size = int(self.kb_cfg.get("chunk_size", 1200))
        chunk_overlap = int(self.kb_cfg.get("chunk_overlap", 200))
        embedding_model = str(
            self.kb_cfg.get(
                "embedding_model",
                self.config.reranker.api.model,
            )
        )
        batch_size = int(self.kb_cfg.get("embedding_batch_size", 32))

        use_zotero_corpus = bool(self.rag_cfg.get("use_zotero_corpus", True))
        zotero_docs = []
        zotero_stats = {"input": 0, "kept": 0, "dedupe_dropped": 0}
        if use_zotero_corpus and self.zotero_corpus:
            dedupe_on = bool(self.rag_cfg.get("zotero_dedupe", True))
            min_abs = int(self.config.executor.get("dedupe", {}).get("min_abstract_len", 80))
            zotero_docs, zotero_stats = build_zotero_rag_docs(
                self.zotero_corpus,
                enable_dedupe=dedupe_on,
                min_abstract_len=min_abs,
            )

        signature = _hash_inputs(paths, chunk_size, chunk_overlap, embedding_model) + "|" + zotero_doc_signature(
            zotero_docs
        )
        loaded = load_index(index_path)
        if loaded and loaded.metadata.get("signature") == signature:
            self._index = loaded
            self.last_index_stats = dict(loaded.metadata.get("stats", {}))
            logger.info("RAG 索引加载完成：{} 片段。", len(loaded.chunks))
            return loaded

        chunks = collect_knowledge_chunks(paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for doc in zotero_docs:
            doc_chunks = chunk_text(
                source_path=f"zotero://{doc.item_key}",
                title=doc.title,
                text=doc.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for c in doc_chunks:
                c.metadata = {
                    "item_key": doc.item_key,
                    "doi": doc.doi,
                    "collection_path": doc.collection_path,
                    "source_type": "zotero",
                }
            chunks.extend(doc_chunks)
        if not chunks:
            logger.warning("RAG 未提取到可用知识片段。")
            return None
        stats = {
            "chunks_total": len(chunks),
            "zotero_docs_input": int(zotero_stats["input"]),
            "zotero_docs_kept": int(zotero_stats["kept"]),
            "zotero_docs_dropped": int(zotero_stats["dedupe_dropped"]),
        }
        index = build_index(
            client=self.client,
            embedding_model=embedding_model,
            batch_size=batch_size,
            chunks=chunks,
            metadata={
                "signature": signature,
                "paths": paths,
                "embedding_model": embedding_model,
                "stats": stats,
            },
        )
        save_index(index_path, index)
        self._index = index
        self.last_index_stats = stats
        logger.info("RAG 索引构建完成：{} 片段，存储于 {}", len(index.chunks), index_path)
        return index

    def retrieve(self, *, title: str, abstract: str) -> RagContext | None:
        index = self._build_or_load_index()
        if index is None or index.matrix.size == 0:
            return None
        query = f"{(title or '').strip()}\n{(abstract or '').strip()}".strip()
        if not query:
            return None
        model = str(self.kb_cfg.get("embedding_model", self.config.reranker.api.model))
        response = self.client.embeddings.create(input=[query], model=model)
        query_vec = np.asarray([response.data[0].embedding], dtype=np.float32)
        scores = _cosine_scores(query_vec, index.matrix)
        if scores.size == 0:
            return None

        top_k = max(1, int(self.rag_cfg.get("top_k", 3)))
        min_score = float(self.rag_cfg.get("min_retrieval_score", 0.2))
        idx = np.argsort(-scores)[:top_k]
        selected = [int(i) for i in idx if float(scores[i]) >= min_score]
        if not selected:
            return None

        max_chars = max(200, int(self.rag_cfg.get("max_context_chars", 2200)))
        hits: list[RagHit] = []
        context_parts: list[str] = []
        used = 0
        for rank, i in enumerate(selected, start=1):
            ch = index.chunks[i]
            snippet = " ".join(ch.text.split())
            if used + len(snippet) > max_chars and context_parts:
                break
            if used + len(snippet) > max_chars:
                snippet = snippet[: max(50, max_chars - used)]
            context_parts.append(f"[{rank}] {ch.title} | {Path(ch.source_path).name}\n{snippet}")
            used += len(snippet)
            hits.append(
                RagHit(
                    source_path=ch.source_path,
                    title=ch.title,
                    score=float(scores[i]),
                    snippet=snippet[:200],
                    item_key=(ch.metadata or {}).get("item_key"),
                    doi=(ch.metadata or {}).get("doi"),
                    collection_path=(ch.metadata or {}).get("collection_path"),
                )
            )
        if not hits:
            return None
        context_text = "\n\n".join(context_parts)
        logger.info(
            "RAG 检索命中 {} 条，最高分 {:.3f}，最低分 {:.3f}。",
            len(hits),
            max(h.score for h in hits),
            min(h.score for h in hits),
        )
        return RagContext(context_text=context_text, hits=hits)
