from datetime import datetime

from zotero_tracker.keywords import KeywordResult
from zotero_tracker.markdown_report import render_html, render_markdown
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
            natural_explain="与书库中机器学习方向条目主题接近。",
            item_id="i1",
            citation_count=100,
            journal_name="Nature",
            journal_sjr=12.3,
            journal_quartile="Q1",
            quality_score=8.1,
            score_breakdown={
                "final_score": 8.5,
                "quality_score": 8.1,
                "relevance": 7.9,
                "citation": 9.0,
                "journal": 8.8,
                "authority": 10.0,
            },
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
        briefing_intro="今日推送以表示学习为主。",
    )
    assert "相关</a>" in html
    assert "不相关</a>" in html
    assert "https://x/rel" in html
    assert "https://x/irrel" in html
    assert "为什么推荐给你" in html
    assert "命中关键词" in html
    assert "Lib paper" in html
    assert "今日简报" in html
    assert "今日推送以表示学习为主" in html
    assert "推荐解读" in html
    assert "机器学习" in html
    assert "质量权重分解" in html
    assert "引用量原值" in html


def test_render_markdown_briefing_and_natural_explain():
    papers = [
        Paper(
            source="arxiv",
            title="P2",
            authors=["C"],
            abstract="y",
            url="https://example.com/q",
            score=0.5,
            tldr="tldr2",
            natural_explain="第二篇的解读。",
            score_breakdown={
                "final_score": 5.0,
                "quality_score": 3.0,
                "relevance": 5.0,
                "citation": 0.0,
                "journal": 0.0,
                "authority": 0.0,
            },
            corpus_explanations=[],
        )
    ]
    md = render_markdown(
        papers,
        KeywordResult(terms=["x"], scores=[1.0]),
        date=datetime(2026, 4, 9),
        briefing_intro="简报开头。",
    )
    assert "## 今日简报" in md
    assert "简报开头" in md
    assert "**推荐解读：**" in md
    assert "第二篇的解读" in md
    assert "#### 质量权重分解" in md
