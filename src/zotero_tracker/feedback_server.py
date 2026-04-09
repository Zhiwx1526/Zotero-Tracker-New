"""轻量反馈服务：接收邮件反馈链接并入库。"""

from __future__ import annotations

import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import dotenv
from loguru import logger
from omegaconf import OmegaConf

from .feedback import (
    FeedbackStore,
    VALID_LABELS,
    build_signature_payload,
    verify_feedback_signature,
)

dotenv.load_dotenv()


def _is_likely_prefetch(user_agent: str) -> bool:
    ua = (user_agent or "").lower()
    markers = ("bot", "crawler", "spider", "preview", "prefetch", "safe", "scan")
    return any(m in ua for m in markers)


def _render_message(title: str, content: str) -> bytes:
    page = (
        "<html><head><meta charset='utf-8'><title>{}</title></head>"
        "<body><h2>{}</h2><p>{}</p></body></html>"
    ).format(html.escape(title), html.escape(title), html.escape(content))
    return page.encode("utf-8")


def _load_feedback_cfg():
    project_root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "config" / "base.yaml"),
        OmegaConf.load(project_root / "config" / "custom.yaml"),
    )
    return cfg.feedback


def run_server() -> None:
    fb_cfg = _load_feedback_cfg()
    if not bool(fb_cfg.get("enabled", False)):
        logger.warning("feedback.enabled=false，反馈服务仍可启动，但不会被邮件链接使用。")

    base_url = str(fb_cfg.get("base_url", "http://127.0.0.1:8787"))
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8787
    secret = str(fb_cfg.get("secret", "")).strip()
    ttl_seconds = int(fb_cfg.get("ttl_seconds", 604800))
    store = FeedbackStore(str(fb_cfg.get("store_path", "data/feedback.sqlite")))
    prefetch_guard = str(fb_cfg.get("prefetch_guard", "basic")).strip().lower()
    stats = {"total": 0, "accepted": 0, "rejected": 0, "prefetch": 0}

    def _log_stats() -> None:
        total = max(1, stats["total"])
        logger.info(
            "反馈统计 total={} accepted={} rejected={} prefetch={} accept_rate={:.1f}% reject_rate={:.1f}%",
            stats["total"],
            stats["accepted"],
            stats["rejected"],
            stats["prefetch"],
            stats["accepted"] * 100.0 / total,
            stats["rejected"] * 100.0 / total,
        )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            stats["total"] += 1
            parsed_url = urlparse(self.path)
            if parsed_url.path == "/health":
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                self.wfile.write(b"ok")
                return
            if parsed_url.path != "/feedback":
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return

            query = parse_qs(parsed_url.query)
            user_id = (query.get("u", [""])[0] or "").strip()
            push_id = (query.get("p", [""])[0] or "").strip()
            item_id = (query.get("i", [""])[0] or "").strip()
            label = (query.get("l", [""])[0] or "").strip()
            ts_raw = (query.get("ts", [""])[0] or "").strip()
            sig = (query.get("sig", [""])[0] or "").strip()
            source = (query.get("s", [""])[0] or "").strip().lower()[:32]
            tags_raw = (query.get("t", [""])[0] or "").strip()
            tags = [x.strip().lower() for x in tags_raw.split(",") if x.strip()][:8]
            ua = self.headers.get("User-Agent", "")
            ip = self.client_address[0] if self.client_address else ""

            if prefetch_guard == "basic" and _is_likely_prefetch(ua):
                stats["prefetch"] += 1
                logger.info("疑似预取请求，已跳过写入。ua={}", ua)
                body = _render_message("已忽略预取请求", "请从邮件中手动点击一次反馈链接以提交反馈。")
                self.send_response(HTTPStatus.ACCEPTED)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                _log_stats()
                return

            if not all([user_id, push_id, item_id, label, ts_raw, sig, secret]):
                stats["rejected"] += 1
                body = _render_message("反馈失败", "参数不完整，请从邮件中重新点击链接。")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                _log_stats()
                return
            if label not in VALID_LABELS:
                stats["rejected"] += 1
                body = _render_message("反馈失败", "标签非法。")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                _log_stats()
                return
            try:
                ts = int(ts_raw)
            except ValueError:
                stats["rejected"] += 1
                body = _render_message("反馈失败", "时间戳非法。")
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                _log_stats()
                return
            now_ts = int(__import__("time").time())
            if now_ts - ts > ttl_seconds:
                stats["rejected"] += 1
                body = _render_message("反馈链接已过期", "请等待下一封邮件后再次反馈。")
                self.send_response(HTTPStatus.GONE)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                _log_stats()
                return

            payload = build_signature_payload(user_id, push_id, item_id, label, ts)
            if not verify_feedback_signature(secret, payload, sig):
                stats["rejected"] += 1
                body = _render_message("反馈失败", "签名校验未通过。")
                self.send_response(HTTPStatus.FORBIDDEN)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
                _log_stats()
                return

            store.upsert_feedback(
                user_id=user_id,
                push_id=push_id,
                item_id=item_id,
                label=label,
                source=source,
                tags=tags,
                event_ts=ts,
                user_agent=ua,
                ip=ip,
            )
            stats["accepted"] += 1
            logger.info("反馈已记录 user={} item={} label={}", user_id, item_id, label)
            text = "已标记为：相关" if label == "rel" else "已标记为：不相关"
            body = _render_message("感谢反馈", text)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            _log_stats()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("反馈服务启动：http://{}:{}/feedback", host, port)
    server.serve_forever()
