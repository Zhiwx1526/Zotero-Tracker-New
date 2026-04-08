"""OpenAlex：通过 Works API 检索论文元数据。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import json

from loguru import logger

from ..protocol import Paper
from .base import BaseRetriever, register_retriever


def _decode_inverted_index(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    tokens: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for position in positions:
            tokens.append((position, token))
    tokens.sort(key=lambda x: x[0])
    return " ".join(token for _, token in tokens).strip()


@register_retriever("openalex")
class OpenAlexRetriever(BaseRetriever):
    API_URL = "https://api.openalex.org/works"

    def _publication_date_filter(self) -> str:
        """OpenAlex 按出版日 YYYY-MM-DD 过滤。未写死 from/to 时：UTC 当天往前数 days 个自然日～当天。"""
        cfg = self.retriever_config
        from_raw = cfg.get("from_publication_date")
        to_raw = cfg.get("to_publication_date")
        from_s = str(from_raw).strip() if from_raw not in (None, "") else ""
        to_s = str(to_raw).strip() if to_raw not in (None, "") else ""

        if from_s and to_s:
            return f"from_publication_date:{from_s},to_publication_date:{to_s}"

        now = datetime.now(timezone.utc)
        today_d = now.date()
        days = max(0, int(cfg.get("days", 3)))
        start_d = today_d - timedelta(days=days)
        return (
            f"from_publication_date:{start_d.isoformat()},"
            f"to_publication_date:{today_d.isoformat()}"
        )

    def _resolve_query(self) -> str | None:
        cfg = self.retriever_config
        if cfg.get("query"):
            return str(cfg.query).strip()
        categories = self.config.source.arxiv.get("category")
        if categories:
            return " ".join(str(c) for c in categories)
        return None

    def _http_get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode(params)
        headers = {"User-Agent": "zotero-tracker/0.1 (+https://openalex.org)"}
        request = Request(f"{self.API_URL}?{query}", headers=headers, method="GET")
        try:
            with urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            logger.error(f"OpenAlex 请求失败: {exc.code} {exc.reason}; url={request.full_url}; body={detail}")
            raise
        return json.loads(payload)

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        cfg = self.retriever_config
        query = self._resolve_query()
        if not query:
            logger.warning("OpenAlex 未提供 query，且 source.arxiv.category 为空，已跳过。")
            return []

        per_page = max(1, min(int(cfg.get("per_page", 50)), 200))
        max_results = max(1, int(cfg.get("max_results", 200)))
        if self.config.executor.debug:
            max_results = min(max_results, 20)
        mailto = cfg.get("mailto")
        filter_parts: list[str] = []
        date_filter = self._publication_date_filter()
        if date_filter:
            filter_parts.append(date_filter)
        title_only = bool(cfg.get("search_title_only", False))
        if title_only:
            filter_parts.append(f"title.search:{query}")
        filter_value = ",".join(filter_parts)

        raw_papers: list[dict[str, Any]] = []
        page = 1
        while len(raw_papers) < max_results:
            params: dict[str, Any] = {
                "page": page,
                "per-page": per_page,
                "sort": "publication_date:desc",
            }
            if not title_only:
                params["search"] = query
            if filter_value:
                params["filter"] = filter_value
            if mailto:
                params["mailto"] = str(mailto)
            data = self._http_get_json(params)
            batch = data.get("results", [])
            if not batch:
                break
            raw_papers.extend(batch)
            if len(batch) < per_page:
                break
            page += 1

        return raw_papers[:max_results]

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = str(raw_paper.get("title") or "").strip()
        if not title:
            return None

        authorships = raw_paper.get("authorships", []) or []
        authors = [
            str(authorship.get("author", {}).get("display_name") or "").strip()
            for authorship in authorships
            if authorship.get("author", {}).get("display_name")
        ]

        abstract = _decode_inverted_index(raw_paper.get("abstract_inverted_index"))
        if not abstract:
            abstract = str(raw_paper.get("abstract") or "").strip()

        primary = raw_paper.get("primary_location") or {}
        pdf_url = primary.get("pdf_url") or raw_paper.get("open_access", {}).get("oa_url")
        url = raw_paper.get("id") or primary.get("landing_page_url")
        if not url:
            return None

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=str(url),
            pdf_url=str(pdf_url) if pdf_url else None,
        )
