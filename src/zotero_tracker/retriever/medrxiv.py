"""medRxiv：最近 N 天预印本元数据检索."""

from __future__ import annotations

from typing import Any

from ..protocol import Paper
from .base import BaseRetriever, register_retriever
from .biorxiv_like import fetch_biorxiv_like


@register_retriever("medrxiv")
class MedrxivRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        days = int(self.retriever_config.get("days", 2))
        max_results = int(self.retriever_config.get("max_results", 200))
        timeout_seconds = float(self.retriever_config.get("timeout_seconds", 60))
        num_retries = int(self.retriever_config.get("num_retries", 3))
        retry_backoff_seconds = float(self.retriever_config.get("retry_backoff_seconds", 2))
        return fetch_biorxiv_like(
            "medrxiv",
            days=days,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
            num_retries=num_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = str(raw_paper.get("title") or "").strip()
        if not title:
            return None

        authors_raw = raw_paper.get("authors") or ""
        authors = [a.strip() for a in authors_raw.split(";") if a.strip()] if isinstance(authors_raw, str) else []

        abstract = str(raw_paper.get("abstract") or "").strip()

        doi = str(raw_paper.get("doi") or "").strip()
        rel_doi = str(raw_paper.get("rel_doi") or "").strip()
        url = ""
        if rel_doi:
            url = f"https://doi.org/{rel_doi}"
        elif doi:
            url = f"https://doi.org/{doi}"
        else:
            url = str(raw_paper.get("medrxiv_url") or raw_paper.get("url") or "").strip()
        if not url:
            return None

        pdf_url = str(raw_paper.get("medrxiv_pdf_url") or "").strip() or None

        canon_doi = (rel_doi or doi or "").strip() or None
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            item_id=(doi or rel_doi or None),
            tags=["medrxiv"],
            doi=canon_doi,
        )

