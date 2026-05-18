from omegaconf import OmegaConf

from zotero_tracker.protocol import Paper
from zotero_tracker.reranker.api import ApiReranker


def _build_cfg(fallback_policy: str = "redistribute"):
    return OmegaConf.create(
        {
            "reranker": {
                "api": {
                    "key": "dummy",
                    "base_url": "https://example.com/v1",
                    "model": "text-embedding-v3",
                    "batch_size": 8,
                    "cache": {"enabled": False},
                }
            },
            "executor": {
                "quality_ranking": {
                    "enabled": True,
                    "fallback_policy": fallback_policy,
                    "weights": {
                        "relevance": 0.9,
                        "authority": 0.1,
                    },
                },
            },
            "quality_data": {
                "source_authority": {
                    "openalex": 1.0,
                    "arxiv": 0.75,
                },
            },
        }
    )


def test_api_reranker_quality_ranking_redistribute():
    reranker = ApiReranker(_build_cfg("redistribute"))
    p = Paper(
        source="openalex",
        title="Paper A",
        authors=[],
        abstract="A",
        url="https://example.com/a",
        score=8.0,
    )
    reranker._apply_quality_ranking([p])
    assert p.score is not None and p.score > 0
    assert p.quality_score is not None
    assert p.score_breakdown["authority"] > 0


def test_api_reranker_quality_ranking_zero_fill():
    reranker = ApiReranker(_build_cfg("zero_fill"))
    p = Paper(
        source="arxiv",
        title="Paper B",
        authors=[],
        abstract="B",
        url="https://example.com/b",
        score=7.0,
    )
    reranker._apply_quality_ranking([p])
    assert p.score is not None
    assert "journal" not in p.score_breakdown
