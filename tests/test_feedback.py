from pathlib import Path

from omegaconf import OmegaConf

from zotero_tracker.feedback import (
    FeedbackStore,
    LABEL_IRRELEVANT,
    LABEL_RELEVANT,
    apply_feedback_reweight,
    build_feedback_links,
    build_signature_payload,
    sign_feedback_payload,
    verify_feedback_signature,
)
from zotero_tracker.protocol import Paper


def test_signature_roundtrip():
    payload = build_signature_payload("u1", "p1", "i1", LABEL_RELEVANT, 123)
    sig = sign_feedback_payload("secret", payload)
    assert verify_feedback_signature("secret", payload, sig)
    assert not verify_feedback_signature("secret", payload, "bad")


def test_feedback_store_upsert_and_load(tmp_path: Path):
    store = FeedbackStore(str(tmp_path / "feedback.sqlite"))
    store.upsert_feedback(
        user_id="u1",
        push_id="p1",
        item_id="i1",
        label=LABEL_RELEVANT,
        source="arxiv",
        tags=["cs.ai"],
        event_ts=2_000_000_000,
    )
    store.upsert_feedback(
        user_id="u1",
        push_id="p1",
        item_id="i1",
        label=LABEL_IRRELEVANT,
        source="arxiv",
        tags=["cs.ai"],
        event_ts=2_000_000_100,
    )
    rows = store.load_recent_feedback(user_id="u1", lookback_days=99999)
    assert len(rows) == 1
    assert rows[0]["label"] == LABEL_IRRELEVANT


def test_apply_feedback_reweight_changes_order(tmp_path: Path):
    db = tmp_path / "feedback.sqlite"
    cfg = OmegaConf.create(
        {
            "feedback": {
                "enabled": True,
                "store_path": str(db),
                "lookback_days": 365,
                "decay_tau_days": 365,
                "weights": {
                    "item_pos": 3.0,
                    "item_neg": 3.0,
                    "tag_pos": 0.0,
                    "tag_neg": 0.0,
                },
            }
        }
    )
    store = FeedbackStore(str(db))
    store.upsert_feedback(
        user_id="u1",
        push_id="p1",
        item_id="hit",
        label=LABEL_RELEVANT,
        source="arxiv",
        tags=[],
        event_ts=2_000_000_000,
    )
    papers = [
        Paper(source="arxiv", title="A", authors=[], abstract="", url="u1", score=1.0, item_id="miss"),
        Paper(source="arxiv", title="B", authors=[], abstract="", url="u2", score=0.5, item_id="hit"),
    ]
    apply_feedback_reweight(cfg, papers, "u1")
    assert papers[0].item_id == "hit"


def test_build_feedback_links_contains_both_labels():
    links = build_feedback_links(
        base_url="http://localhost:8787",
        secret="secret",
        user_id="u1",
        push_id="p1",
        item_id="i1",
        ts=123,
        source="arxiv",
        tags=["cs.ai"],
    )
    assert LABEL_RELEVANT in links
    assert LABEL_IRRELEVANT in links
    assert "sig=" in links[LABEL_RELEVANT].url
