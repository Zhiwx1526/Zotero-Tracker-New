"""通用的 bioRxiv / medRxiv 抓取逻辑."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json

from loguru import logger


def _build_date_range(days: int) -> tuple[str, str]:
    """返回 (from, to) 日期字符串，格式 YYYY-MM-DD，基于 UTC 最近 N 天."""
    days = max(1, days)
    now = datetime.now(timezone.utc)
    today = now.date()
    start = today - timedelta(days=days - 1)
    return start.isoformat(), today.isoformat()


def fetch_biorxiv_like(
    server: str,
    days: int,
    max_results: int,
    timeout_seconds: float = 30.0,
    num_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """从 bioRxiv / medRxiv API 获取最近 N 天的预印本元数据.

    使用日期范围接口：
    https://api.biorxiv.org/details/{server}/{from_yyyy-mm-dd}/{to_yyyy-mm-dd}/{offset}
    """
    base_url = "https://api.biorxiv.org/details"
    days = max(1, days)
    max_results = max(1, max_results)
    timeout_seconds = max(1.0, float(timeout_seconds))
    num_retries = max(1, int(num_retries))
    retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))
    from_date, to_date = _build_date_range(days)

    results: list[dict[str, Any]] = []
    offset = 0
    page_size = 100

    while len(results) < max_results:
        url = f"{base_url}/{server}/{from_date}/{to_date}/{offset}"
        logger.debug(f"{server}: 请求 {url}")
        req = Request(url, headers={"User-Agent": "zotero-tracker/0.1 (+https://www.biorxiv.org)"})
        payload = ""
        for attempt in range(1, num_retries + 1):
            try:
                with urlopen(req, timeout=timeout_seconds) as resp:
                    payload = resp.read().decode("utf-8")
                break
            except (TimeoutError, URLError, HTTPError, OSError) as exc:
                if attempt >= num_retries:
                    raise
                sleep_s = retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    f"{server}: 请求失败（{attempt}/{num_retries}）{exc}，{sleep_s:.1f}s 后重试"
                )
                time.sleep(sleep_s)
        data = json.loads(payload)
        collection = data.get("collection") or []
        if not collection:
            break
        for item in collection:
            results.append(item)
            if len(results) >= max_results:
                break
        if len(collection) < page_size or len(results) >= max_results:
            break
        offset += page_size

    return results[:max_results]

