"""邮件反馈闭环：签名、存储、规则加权。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import math
from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import urlencode

from loguru import logger

from .protocol import Paper

LABEL_RELEVANT = "rel"
LABEL_IRRELEVANT = "irrel"
VALID_LABELS = {LABEL_RELEVANT, LABEL_IRRELEVANT}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def hash_user_id(receiver_email: str, salt: str) -> str:
    raw = f"{(receiver_email or '').strip().lower()}|{salt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_signature_payload(user_id: str, push_id: str, item_id: str, label: str, ts: int) -> str:
    return f"{user_id}|{push_id}|{item_id}|{label}|{ts}"


def sign_feedback_payload(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_feedback_signature(secret: str, payload: str, got_sig: str) -> bool:
    expected = sign_feedback_payload(secret, payload)
    return hmac.compare_digest(expected, got_sig or "")


def paper_item_id(paper: Paper) -> str:
    if paper.item_id:
        return paper.item_id
    basis = (paper.url or "").strip() or (paper.title or "").strip()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def paper_tags(paper: Paper, max_tags: int = 8) -> list[str]:
    tags = [t.strip().lower() for t in (paper.tags or []) if str(t).strip()]
    if tags:
        return tags[:max_tags]
    text = f"{paper.title} {paper.abstract}".lower()
    tokens = []
    for w in text.replace("\n", " ").split():
        w2 = "".join(ch for ch in w if ch.isalnum())
        if len(w2) >= 4:
            tokens.append(w2)
        if len(tokens) >= max_tags:
            break
    return tokens


@dataclass
class FeedbackLink:
    label: str
    url: str


def build_feedback_links(
    *,
    base_url: str,
    secret: str,
    user_id: str,
    push_id: str,
    item_id: str,
    ts: int,
    source: str = "",
    tags: list[str] | None = None,
) -> dict[str, FeedbackLink]:
    links: dict[str, FeedbackLink] = {}
    for label in (LABEL_RELEVANT, LABEL_IRRELEVANT):
        payload = build_signature_payload(user_id, push_id, item_id, label, ts)
        sig = sign_feedback_payload(secret, payload)
        qs = urlencode(
            {
                "u": user_id,
                "p": push_id,
                "i": item_id,
                "l": label,
                "ts": str(ts),
                "sig": sig,
                "s": source.strip().lower(),
                "t": ",".join([x.strip().lower() for x in (tags or []) if x.strip()])[:256],
            }
        )
        links[label] = FeedbackLink(label=label, url=f"{base_url.rstrip('/')}/feedback?{qs}")
    return links


class FeedbackStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    user_id TEXT NOT NULL,
                    push_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source TEXT,
                    tags TEXT,
                    event_ts INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    user_agent TEXT,
                    ip TEXT,
                    PRIMARY KEY (user_id, push_id, item_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_user_time
                ON feedback_events (user_id, event_ts)
                """
            )

    def upsert_feedback(
        self,
        *,
        user_id: str,
        push_id: str,
        item_id: str,
        label: str,
        source: str,
        tags: list[str],
        event_ts: int,
        user_agent: str = "",
        ip: str = "",
    ) -> None:
        now_ts = int(datetime.now(UTC).timestamp())
        tags_s = ",".join([t.strip().lower() for t in tags if t.strip()])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_events
                (user_id, push_id, item_id, label, source, tags, event_ts, created_at, updated_at, user_agent, ip)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, push_id, item_id) DO UPDATE SET
                    label=excluded.label,
                    source=excluded.source,
                    tags=excluded.tags,
                    event_ts=excluded.event_ts,
                    updated_at=excluded.updated_at,
                    user_agent=excluded.user_agent,
                    ip=excluded.ip
                """,
                (
                    user_id,
                    push_id,
                    item_id,
                    label,
                    source,
                    tags_s,
                    event_ts,
                    now_ts,
                    now_ts,
                    user_agent,
                    ip,
                ),
            )

    def load_recent_feedback(self, *, user_id: str, lookback_days: int) -> list[sqlite3.Row]:
        now_ts = int(datetime.now(UTC).timestamp())
        min_ts = now_ts - max(1, lookback_days) * 86400
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, push_id, item_id, label, source, tags, event_ts
                FROM feedback_events
                WHERE user_id = ? AND event_ts >= ?
                ORDER BY event_ts DESC
                """,
                (user_id, min_ts),
            ).fetchall()
        return rows


def apply_feedback_reweight(config: Any, papers: list[Paper], user_id: str) -> None:
    fb_cfg = _cfg_get(config, "feedback")
    if not fb_cfg or not bool(_cfg_get(fb_cfg, "enabled", False)):
        return
    store = FeedbackStore(str(_cfg_get(fb_cfg, "store_path", "data/feedback.sqlite")))
    rows = store.load_recent_feedback(
        user_id=user_id,
        lookback_days=int(_cfg_get(fb_cfg, "lookback_days", 90)),
    )
    if not rows:
        logger.info("未命中历史反馈，跳过反馈重排。")
        return

    tau = float(_cfg_get(fb_cfg, "decay_tau_days", 30))
    now_ts = int(datetime.now(UTC).timestamp())
    wcfg = _cfg_get(fb_cfg, "weights", {})
    item_pos = float(_cfg_get(wcfg, "item_pos", 1.2))
    item_neg = float(_cfg_get(wcfg, "item_neg", 1.6))
    tag_pos = float(_cfg_get(wcfg, "tag_pos", 0.35))
    tag_neg = float(_cfg_get(wcfg, "tag_neg", 0.55))

    item_score: dict[str, float] = {}
    tag_score: dict[str, float] = {}
    for r in rows:
        age_days = max(0.0, (now_ts - int(r["event_ts"])) / 86400.0)
        decay = math.exp(-age_days / max(1.0, tau))
        sign = 1.0 if r["label"] == LABEL_RELEVANT else -1.0
        item_w = item_pos if sign > 0 else item_neg
        tag_w = tag_pos if sign > 0 else tag_neg
        item_id = str(r["item_id"])
        item_score[item_id] = item_score.get(item_id, 0.0) + sign * item_w * decay
        tags = [x.strip() for x in str(r["tags"] or "").split(",") if x.strip()]
        for t in tags:
            tag_score[t] = tag_score.get(t, 0.0) + sign * tag_w * decay

    for p in papers:
        base = float(p.score or 0.0)
        pid = paper_item_id(p)
        tags = paper_tags(p)
        item_adj = item_score.get(pid, 0.0)
        tag_adj = sum(tag_score.get(t, 0.0) for t in tags)
        adj = item_adj + tag_adj
        p.score = base + adj
        title_preview = (p.title or "").replace("\n", " ").strip()
        if len(title_preview) > 60:
            title_preview = title_preview[:57] + "…"
        logger.info(
            "反馈调整 item_id={} title={} 向量分={:.4f} item贡献={:+.4f} tag贡献={:+.4f} 反馈合计={:+.4f} 重排后分={:.4f}",
            pid,
            title_preview or "(无标题)",
            base,
            item_adj,
            tag_adj,
            adj,
            float(p.score or 0.0),
        )

    papers.sort(key=lambda x: x.score or 0.0, reverse=True)
    logger.info("反馈重排完成：加载 {} 条历史反馈记录，已对 {} 篇候选写入分项日志。".format(len(rows), len(papers)))
