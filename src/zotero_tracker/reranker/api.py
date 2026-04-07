from pathlib import Path

import numpy as np
from loguru import logger
from openai import OpenAI
from omegaconf import DictConfig

from ..protocol import CorpusPaper, Paper
from .base import BaseReranker, _text_for_embedding, register_reranker
from .cache_store import build_fingerprint, load_cache, save_cache


@register_reranker("api")
class ApiReranker(BaseReranker):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        api = config.reranker.api
        self._client = OpenAI(api_key=api.key, base_url=api.base_url)
        self._model = api.model
        self._batch_size = int(api.get("batch_size") or 64)
        cache_cfg = api.get("cache", {})
        self._cache_enabled = bool(cache_cfg.get("enabled", True))
        self._cache_dir = Path(str(cache_cfg.get("dir", "cache")))
        self._text_format_version = str(cache_cfg.get("text_format_version", "title_abstract_v1"))

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        s1_embeddings = self._embed_texts(s1)
        s2_embeddings = self._embed_texts(s2)
        s1_n = s1_embeddings / np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        s2_n = s2_embeddings / np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        return np.dot(s1_n, s2_n.T)

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus, key=lambda x: x.added_date, reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight = time_decay_weight / time_decay_weight.sum()

        candidate_texts = [_text_for_embedding(p.title, p.abstract) for p in candidates]
        corpus_texts = [_text_for_embedding(c.title, c.abstract) for c in corpus]
        candidate_embeddings = self._embed_texts(candidate_texts)
        corpus_embeddings = self._get_corpus_embeddings(corpus, corpus_texts)
        candidate_n = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        corpus_n = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        sim = np.dot(candidate_n, corpus_n.T)
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10
        for s, c in zip(scores, candidates, strict=True):
            c.score = float(s)
        return sorted(candidates, key=lambda x: x.score or 0.0, reverse=True)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            all_embeddings.extend([r.embedding for r in response.data])
        return np.asarray(all_embeddings, dtype=np.float32)

    def _get_corpus_embeddings(self, corpus: list[CorpusPaper], corpus_texts: list[str]) -> np.ndarray:
        if not self._cache_enabled:
            logger.info("书库 embedding 缓存已禁用，重算 {} 条。".format(len(corpus)))
            return self._embed_texts(corpus_texts)
        if any(not c.item_key for c in corpus):
            logger.warning("存在缺失 item_key 的书库条目，跳过缓存并重算 {} 条。".format(len(corpus)))
            return self._embed_texts(corpus_texts)

        cache_items = load_cache(self._cache_dir, self._model, self._text_format_version)
        current_keys = set()
        cache_hits = 0
        to_embed_keys: list[str] = []
        to_embed_texts: list[str] = []
        to_embed_fingerprints: list[str] = []
        ordered_vectors: list[np.ndarray | None] = [None] * len(corpus)

        for idx, (item, text) in enumerate(zip(corpus, corpus_texts, strict=True)):
            item_key = item.item_key
            current_keys.add(item_key)
            fingerprint = build_fingerprint(text, self._model, self._text_format_version)
            cached = cache_items.get(item_key)
            if cached and cached.get("fingerprint_sha256") == fingerprint:
                ordered_vectors[idx] = np.asarray(cached["embedding"], dtype=np.float32)
                cache_hits += 1
                continue
            to_embed_keys.append(item_key)
            to_embed_texts.append(text)
            to_embed_fingerprints.append(fingerprint)

        if to_embed_texts:
            embedded = self._embed_texts(to_embed_texts)
            for key, fp, vec in zip(to_embed_keys, to_embed_fingerprints, embedded, strict=True):
                cache_items[key] = {
                    "fingerprint_sha256": fp,
                    "embedding": np.asarray(vec, dtype=np.float32),
                }
            waiting_idx = 0
            for idx, vec in enumerate(ordered_vectors):
                if vec is None:
                    ordered_vectors[idx] = np.asarray(embedded[waiting_idx], dtype=np.float32)
                    waiting_idx += 1

        stale_keys = [k for k in cache_items if k not in current_keys]
        for k in stale_keys:
            del cache_items[k]

        save_cache(self._cache_dir, cache_items, self._model, self._text_format_version)
        total = len(corpus)
        hit_rate = (cache_hits / total * 100.0) if total else 0.0
        logger.info(
            "书库 embedding 缓存统计：命中 {}/{} ({:.1f}%)，重算 {} 条，清理 {} 条。".format(
                cache_hits,
                total,
                hit_rate,
                len(to_embed_texts),
                len(stale_keys),
            )
        )
        return np.vstack([np.asarray(v, dtype=np.float32) for v in ordered_vectors])
