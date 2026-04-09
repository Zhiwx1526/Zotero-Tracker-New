from datetime import datetime

import numpy as np
import pytest

from zotero_tracker.protocol import CorpusPaper, Paper
from zotero_tracker.reranker.explain import attach_corpus_explanations


def test_attach_corpus_explanations_top_by_contribution():
    corpus = [
        CorpusPaper(
            item_key="a",
            title="Short A",
            abstract="x",
            added_date=datetime(2024, 1, 1),
            paths=["p/a"],
        ),
        CorpusPaper(
            item_key="b",
            title="Short B",
            abstract="y",
            added_date=datetime(2024, 1, 2),
            paths=["p/b"],
        ),
        CorpusPaper(
            item_key="c",
            title="Short C",
            abstract="z",
            added_date=datetime(2024, 1, 3),
            paths=[],
        ),
    ]
    candidates = [
        Paper(
            source="arxiv",
            title="Cand",
            authors=[],
            abstract="z",
            url="https://example.com",
        )
    ]
    # one row: high sim with index 0, lower with 1 and 2
    sim = np.array([[0.9, 0.5, 0.1]], dtype=np.float32)
    w = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    attach_corpus_explanations(
        candidates,
        corpus,
        w,
        sim,
        top_k=2,
        enabled=True,
        title_max_len=80,
    )
    ex = candidates[0].corpus_explanations
    assert len(ex) == 2
    # contribution = sim * w * 10 — 索引 0 为 1.8，索引 1 为 1.5
    assert ex[0].item_key == "a"
    assert ex[0].contribution == pytest.approx(0.9 * 0.2 * 10)
    assert ex[1].item_key == "b"
    assert ex[1].contribution == pytest.approx(0.5 * 0.3 * 10)


def test_attach_corpus_explanations_disabled_clears():
    p = Paper(
        source="arxiv",
        title="x",
        authors=[],
        abstract="y",
        url="u",
        corpus_explanations=[],
    )
    corpus = [
        CorpusPaper(
            item_key="a",
            title="t",
            abstract="b",
            added_date=datetime(2024, 1, 1),
            paths=[],
        )
    ]
    sim = np.ones((1, 1), dtype=np.float32)
    w = np.ones(1)
    attach_corpus_explanations(
        [p],
        corpus,
        w,
        sim,
        top_k=3,
        enabled=False,
        title_max_len=80,
    )
    assert p.corpus_explanations == []
