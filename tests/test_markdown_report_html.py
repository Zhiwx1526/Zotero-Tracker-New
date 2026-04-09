from datetime import datetime

from zotero_tracker.keywords import KeywordResult
from zotero_tracker.markdown_report import render_html
from zotero_tracker.protocol import Paper


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
