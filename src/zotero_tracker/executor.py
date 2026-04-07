"""串联：Zotero 语料 → 关键词 → 各来源检索 → 重排 → TLDR → Markdown 邮件。"""

import random
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .email_smtp import send_markdown_email
from .keywords import extract_keywords_from_corpus
from .markdown_report import render_markdown
from .protocol import CorpusPaper
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .utils_glob import glob_match


def normalize_path_patterns(
    patterns: list[str] | ListConfig | None,
    config_key: str,
) -> list[str] | None:
    if patterns is None:
        return None
    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} 必须为 glob 字符串列表或 null，"
            '例如 ["2026/survey/**"]；不支持单个字符串。'
        )
    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} 中的元素必须均为字符串。")
    return list(patterns)


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("正在从 Zotero 拉取书库…")
        zot = zotero.Zotero(self.config.zotero.user_id, "user", self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c["key"]: c for c in collections}
        corpus = zot.everything(zot.items(itemType="conferencePaper || journalArticle || preprint"))
        corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

        def get_collection_path(col_key: str) -> str:
            if p := collections[col_key]["data"]["parentCollection"]:
                return get_collection_path(p) + "/" + collections[col_key]["data"]["name"]
            return collections[col_key]["data"]["name"]

        for c in corpus:
            paths = [get_collection_path(col) for col in c["data"]["collections"]]
            c["paths"] = paths
        logger.info(f"共拉取 {len(corpus)} 条带摘要的 Zotero 文献")
        return [
            CorpusPaper(
                item_key=c["key"],
                title=c["data"]["title"],
                abstract=c["data"]["abstractNote"],
                added_date=datetime.strptime(c["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"),
                paths=c["paths"],
            )
            for c in corpus
        ]

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"按 include_path 筛选：{self.include_path_patterns}")
            corpus = [
                c
                for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"按 ignore_path 排除：{self.ignore_path_patterns}")
            corpus = [
                c
                for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            sample_text = "\n".join([c.title + " — " + "\n".join(c.paths) for c in samples])
            logger.info(f"筛选后共 {len(corpus)} 条。示例如下：\n{sample_text}\n...")
        return corpus

    def run(self) -> None:
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"未找到可用 Zotero 文献，请检查配置：\n{self.config.zotero}")
            return

        kw = extract_keywords_from_corpus(corpus, self.config.keywords)
        logger.info(
            f"兴趣关键词（展示用）：{', '.join(kw.terms[:10])}{'…' if len(kw.terms) > 10 else ''}"
        )

        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"正在从 {source} 拉取…")
            papers = retriever.retrieve_papers()
            if not papers:
                logger.info(f"{source} 无新稿")
                continue
            logger.info(f"从 {source} 得到 {len(papers)} 篇")
            all_papers.extend(papers)

        logger.info(f"候选论文总数：{len(all_papers)}")
        reranked: list = []
        if all_papers:
            logger.info("按与书库的向量相似度重排中…")
            reranked = self.reranker.rerank(all_papers, corpus)
            reranked = reranked[: int(self.config.executor.max_paper_num)]
            logger.info("正在生成一句话摘要…")
            for p in tqdm(reranked):
                p.generate_tldr(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("无候选论文且 send_empty=false，不发送邮件。")
            return

        md = render_markdown(reranked, kw)
        logger.info("正在发送邮件…")
        send_markdown_email(self.config, md)
        logger.info("全部完成。")
