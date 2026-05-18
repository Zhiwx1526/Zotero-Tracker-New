from datetime import datetime

from zotero_tracker.protocol import CorpusPaper
from zotero_tracker.rag.zotero_ingest import build_zotero_rag_docs


def test_build_zotero_rag_docs_dedupe_by_doi():
    corpus = [
        CorpusPaper(
            item_key="a1",
            title="Paper A",
            abstract="This is abstract A.",
            added_date=datetime(2026, 1, 1),
            paths=["ml/a"],
            doi="10.1000/xyz123",
        ),
        CorpusPaper(
            item_key="a2",
            title="Paper A duplicate",
            abstract="This is abstract A duplicate.",
            added_date=datetime(2026, 1, 2),
            paths=["ml/b"],
            doi="https://doi.org/10.1000/xyz123",
        ),
    ]
    docs, stats = build_zotero_rag_docs(corpus, enable_dedupe=True, min_abstract_len=20)
    assert len(docs) == 1
    assert stats["dedupe_dropped"] == 1


def test_build_zotero_rag_docs_keep_without_dedupe():
    corpus = [
        CorpusPaper(
            item_key="b1",
            title="Paper B",
            abstract="abstract one",
            added_date=datetime(2026, 1, 1),
            paths=["x"],
            doi=None,
        ),
        CorpusPaper(
            item_key="b2",
            title="Paper B",
            abstract="abstract one",
            added_date=datetime(2026, 1, 2),
            paths=["y"],
            doi=None,
        ),
    ]
    docs, stats = build_zotero_rag_docs(corpus, enable_dedupe=False, min_abstract_len=5)
    assert len(docs) == 2
    assert stats["dedupe_dropped"] == 0
