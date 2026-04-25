from omegaconf import OmegaConf

from zotero_tracker.protocol import Paper
from zotero_tracker.quality_metrics import load_scimago_map, resolve_journal_quality
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
                "explain_enabled": False,
                "quality_ranking": {
                    "enabled": True,
                    "fallback_policy": fallback_policy,
                    "weights": {
                        "relevance": 0.7,
                        "citation": 0.15,
                        "journal": 0.1,
                        "authority": 0.05,
                    },
                },
            },
            "quality_data": {
                "scimago_map_path": None,
                "sjr_log_cap": 10.0,
                "source_authority": {
                    "openalex": 1.0,
                    "arxiv": 0.75,
                },
            },
        }
    )


def test_scimago_map_load_and_resolve(tmp_path):
    csv_path = tmp_path / "sjr.csv"
    csv_path.write_text("journal_name,sjr,quartile\nNature,15.2,Q1\n", encoding="utf-8")
    mapping = load_scimago_map(str(csv_path))
    assert "nature" in mapping
    sjr, quartile, score = resolve_journal_quality("Nature", mapping, sjr_cap=20.0)
    assert sjr == 15.2
    assert quartile == "Q1"
    assert score is not None and 0.0 < score <= 1.0


def test_api_reranker_quality_ranking_redistribute():
    reranker = ApiReranker(_build_cfg("redistribute"))
    p = Paper(
        source="openalex",
        title="Paper A",
        authors=[],
        abstract="A",
        url="https://example.com/a",
        citation_count=200,
        journal_name="Nature",
        score=8.0,
    )
    reranker._scimago_map = {"nature": {"sjr": 12.0, "quartile": "Q1"}}
    reranker._apply_quality_ranking([p])
    assert p.score is not None and p.score > 0
    assert p.quality_score is not None
    assert p.score_breakdown["citation"] > 0
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
    assert p.score_breakdown["citation"] == 0.0
    assert p.score_breakdown["journal"] == 0.0
