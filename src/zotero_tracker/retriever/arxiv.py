"""arXiv：RSS 取 ID + arxiv 库拉元数据（仅用摘要，不下载 PDF/HTML）。"""

import arxiv
import feedparser
from arxiv import Result as ArxivResult
from tqdm import tqdm

from ..protocol import Paper
from .base import BaseRetriever, register_retriever


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("必须在配置中指定 source.arxiv.category。")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = "+".join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if "Feed error for query" in getattr(feed.feed, "title", ""):
            raise RuntimeError(f"无效的 arXiv 分区查询：{query}")
        allowed = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        raw_papers: list[ArxivResult] = []
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i : i + 20])
            batch = list(client.results(search))
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title.replace("\n", " ").strip()
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
        )
