"""arXiv：API 查询 + submittedDate 窗口（UTC，最近 N 天，由 source.arxiv.days 控制）。"""

import threading
from datetime import datetime, timedelta, timezone

import arxiv
from arxiv import Result as ArxivResult
from loguru import logger
from tqdm import tqdm

from ..protocol import Paper
from .base import BaseRetriever, register_retriever

# arxiv 库内部 requests 默认无 timeout
_API_TIMEOUT_S = 120.0
_STUCK_WARN_AFTER_S = 30.0


def _stuck_timer(label: str, seconds: float) -> threading.Timer:
    def _fire() -> None:
        logger.warning(f"{label} 已等待超过 {seconds:.0f}s，可能网络较慢或受限")

    timer = threading.Timer(seconds, _fire)
    timer.daemon = True
    timer.start()
    return timer


def _submitted_date_range_recent_days_utc(days: int) -> str:
    """arXiv API：submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]，见 https://arxiv.org/help/api/user-manual"""
    days = max(1, int(days))
    now = datetime.now(timezone.utc)
    today = now.date()
    start_day = today - timedelta(days=days - 1)
    start = datetime(start_day.year, start_day.month, start_day.day, 0, 0, tzinfo=timezone.utc)
    end = datetime(today.year, today.month, today.day, 23, 59, tzinfo=timezone.utc)
    return f"submittedDate:[{start.strftime('%Y%m%d%H%M')} TO {end.strftime('%Y%m%d%H%M')}]"


def _build_search_query(categories: list[str], days: int) -> str:
    cat_parts = [f"cat:{c}" for c in categories]
    cat_expr = cat_parts[0] if len(cat_parts) == 1 else "(" + " OR ".join(cat_parts) + ")"
    return f"{cat_expr} AND {_submitted_date_range_recent_days_utc(days)}"


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("必须在配置中指定 source.arxiv.category。")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        if self.retriever_config.get("include_cross_list", False):
            logger.warning(
                "source.arxiv.include_cross_list=true：当前基于 API 的 cat 查询无法复刻 RSS 的 cross 过滤，"
                "将仍按分区 cat 检索（含该分区下的交叉列表论文）。"
            )

        client = arxiv.Client(num_retries=10, delay_seconds=10)
        session = client._session
        orig_get = session.get

        def get_with_timeout(*args, **kwargs):
            kwargs.setdefault("timeout", _API_TIMEOUT_S)
            return orig_get(*args, **kwargs)

        session.get = get_with_timeout  # type: ignore[method-assign]

        categories = [str(c) for c in self.config.source.arxiv.category]
        days = int(self.retriever_config.get("days", 2))
        query = _build_search_query(categories, days)
        max_results = int(self.retriever_config.get("max_results", 2000))
        if self.config.executor.debug:
            max_results = min(max_results, 10)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        stuck = _stuck_timer("arXiv API（submittedDate 查询）", _STUCK_WARN_AFTER_S)
        try:
            raw_papers = list(tqdm(client.results(search), desc="arxiv"))
        except Exception as exc:
            logger.error(f"arXiv API 失败：{exc}")
            raise
        finally:
            stuck.cancel()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title.replace("\n", " ").strip()
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        paper_id = ""
        if getattr(raw_paper, "entry_id", None):
            paper_id = str(raw_paper.entry_id).rstrip("/").split("/")[-1]
        tags: list[str] = []
        primary_cat = getattr(raw_paper, "primary_category", None)
        if primary_cat:
            tags.append(str(primary_cat).strip().lower())
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            item_id=paper_id or None,
            tags=tags,
        )
