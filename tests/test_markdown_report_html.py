from datetime import datetime

from zotero_tracker.keywords import KeywordResult
from zotero_tracker.markdown_report import render_html
from zotero_tracker.protocol import CorpusMatchExplain, Paper


def test_render_html_feedback_buttons():
    papers = [
        Paper(
            source="arxiv",
            title="Test Paper",
            authors=["A", "B"],
            abstract="x",
            url="https://example.com/p",
            score=1.23,
            tldr="summary",
            item_id="i1",
            matched_keywords=["ai"],
            corpus_explanations=[
                CorpusMatchExplain(
                    item_key="zk",
                    title="Lib paper",
                    cosine_sim=0.5,
                    time_weight=0.2,
                    contribution=1.0,
                    collection_path="ml/read",
                )
            ],
        )
    ]
    html = render_html(
        papers,
        KeywordResult(terms=["ai"], scores=[1.0]),
        date=datetime(2026, 4, 8),
        feedback_links={"i1": {"rel": "https://x/rel", "irrel": "https://x/irrel"}},
    )
    assert "相关</a>" in html
    assert "不相关</a>" in html
    assert "https://x/rel" in html
    assert "https://x/irrel" in html
    assert "为什么推荐给你" in html
    assert "命中关键词" in html
    assert "Lib paper" in html
