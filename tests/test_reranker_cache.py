import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from zotero_tracker.protocol import CorpusPaper
from zotero_tracker.reranker.api import ApiReranker
from zotero_tracker.reranker.base import _text_for_embedding


def _build_config(cache_dir: Path):
    return OmegaConf.create(
        {
            "reranker": {
                "api": {
                    "key": "test-key",
                    "base_url": "http://test",
                    "model": "test-model",
                    "batch_size": 64,
                    "cache": {
                        "enabled": True,
                        "dir": str(cache_dir),
                        "text_format_version": "title_abstract_v1",
                    },
                }
            }
        }
    )


def _build_corpus(items: list[tuple[str, str, str]]) -> list[CorpusPaper]:
    base = datetime(2024, 1, 1)
    corpus: list[CorpusPaper] = []
    for i, (key, title, abstract) in enumerate(items):
        corpus.append(
            CorpusPaper(
                item_key=key,
                title=title,
                abstract=abstract,
                added_date=base + timedelta(days=i),
                paths=["p"],
            )
        )
    return corpus


def _patch_embedder(reranker: ApiReranker):
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        vectors = []
        for text in texts:
            base = float(sum(ord(c) for c in text) % 997)
            vectors.append([base, float(len(text)) + 0.5])
        return np.asarray(vectors, dtype=np.float32)

    reranker._embed_texts = fake_embed  # type: ignore[method-assign]
    return calls


def _corpus_texts(corpus: list[CorpusPaper]) -> list[str]:
    return [_text_for_embedding(c.title, c.abstract) for c in corpus]


def test_corpus_embedding_cache_incremental_flow(tmp_path: Path):
    cfg = _build_config(tmp_path)

    # 首次运行：全量计算
    corpus_v1 = _build_corpus([("k1", "t1", "a1"), ("k2", "t2", "a2")])
    reranker = ApiReranker(cfg)
    calls = _patch_embedder(reranker)
    emb_v1 = reranker._get_corpus_embeddings(corpus_v1, _corpus_texts(corpus_v1))
    assert emb_v1.shape == (2, 2)
    assert len(calls) == 1
    assert len(calls[0]) == 2

    # 二次运行无变更：不应触发 embedding API
    reranker2 = ApiReranker(cfg)
    calls2 = _patch_embedder(reranker2)
    emb_v2 = reranker2._get_corpus_embeddings(corpus_v1, _corpus_texts(corpus_v1))
    assert emb_v2.shape == (2, 2)
    assert calls2 == []

    # 新增条目：仅新增触发计算
    corpus_v3 = _build_corpus([("k1", "t1", "a1"), ("k2", "t2", "a2"), ("k3", "t3", "a3")])
    reranker3 = ApiReranker(cfg)
    calls3 = _patch_embedder(reranker3)
    emb_v3 = reranker3._get_corpus_embeddings(corpus_v3, _corpus_texts(corpus_v3))
    assert emb_v3.shape == (3, 2)
    assert len(calls3) == 1
    assert len(calls3[0]) == 1

    # 变更摘要：仅变更条目触发计算
    corpus_v4 = _build_corpus([("k1", "t1", "a1-updated"), ("k2", "t2", "a2"), ("k3", "t3", "a3")])
    reranker4 = ApiReranker(cfg)
    calls4 = _patch_embedder(reranker4)
    emb_v4 = reranker4._get_corpus_embeddings(corpus_v4, _corpus_texts(corpus_v4))
    assert emb_v4.shape == (3, 2)
    assert len(calls4) == 1
    assert len(calls4[0]) == 1

    # 删除条目：缓存中应清理被删除 key
    corpus_v5 = _build_corpus([("k1", "t1", "a1-updated"), ("k3", "t3", "a3")])
    reranker5 = ApiReranker(cfg)
    calls5 = _patch_embedder(reranker5)
    emb_v5 = reranker5._get_corpus_embeddings(corpus_v5, _corpus_texts(corpus_v5))
    assert emb_v5.shape == (2, 2)
    assert calls5 == []

    meta_path = tmp_path / "corpus_embed_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert set(meta["items"].keys()) == {"k1", "k3"}


def test_corpus_embedding_cache_corrupted_fallback(tmp_path: Path):
    cfg = _build_config(tmp_path)
    meta_path = tmp_path / "corpus_embed_meta.json"
    vectors_path = tmp_path / "corpus_embed_vectors.npz"
    tmp_path.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("{invalid json", encoding="utf-8")
    np.savez_compressed(vectors_path, vectors=np.asarray([[1.0, 2.0]], dtype=np.float32))

    corpus = _build_corpus([("k1", "t1", "a1"), ("k2", "t2", "a2")])
    reranker = ApiReranker(cfg)
    calls = _patch_embedder(reranker)
    emb = reranker._get_corpus_embeddings(corpus, _corpus_texts(corpus))
    assert emb.shape == (2, 2)
    assert len(calls) == 1
    assert len(calls[0]) == 2
