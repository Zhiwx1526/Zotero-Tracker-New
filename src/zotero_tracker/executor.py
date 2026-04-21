"""串联：Zotero 语料 → 关键词 → 各来源检索 → 重排 → TLDR → Markdown 邮件。"""

import random
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .email_smtp import send_markdown_email
from .feedback import (
    apply_feedback_reweight,
    build_feedback_links,
    hash_user_id,
    paper_item_id,
)
from .dedupe import apply_dedupe_pipeline
from .keywords import KeywordResult, extract_keywords_from_corpus, match_keywords_in_paper
from .markdown_report import render_html, render_markdown
from .protocol import CorpusPaper, Paper, fill_briefing_intro
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .utils_glob import glob_match


def _zotero_raw_doi(data: dict) -> str | None:
    for key in ("DOI", "doi"):
        v = data.get(key)
        if v and str(v).strip():
            return str(v).strip()
    extra = str(data.get("extra") or "")
    m = re.search(r"(?:^|\n)\s*DOI:\s*(10\.\d{4,9}/\S+)", extra, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).rstrip(".,)")
    m = re.search(r"\b(10\.\d{4,9}/[^\s]+)", extra)
    if m:
        return m.group(1).rstrip(".,)")
    return None


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


@dataclass
class RunResult:
    papers: list[Paper]
    keywords: KeywordResult


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {}
        for source in config.executor.source:
            source_cfg = getattr(config.source, source, None)
            enabled = bool(source_cfg.get("enabled", True)) if source_cfg is not None else True
            if not enabled:
                logger.info(f"来源 {source} 已配置为 disabled，跳过。")
                continue
            self.retrievers[source] = get_retriever_cls(source)(config)
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
                doi=_zotero_raw_doi(c["data"]),
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

    def run(self) -> RunResult | None:
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"未找到可用 Zotero 文献，请检查配置：\n{self.config.zotero}")
            return None

        kw = extract_keywords_from_corpus(corpus, self.config.keywords)
        logger.info(
            f"兴趣关键词（展示用）：{', '.join(kw.terms[:10])}{'…' if len(kw.terms) > 10 else ''}"
        )

        all_papers = []
        failed_sources: list[tuple[str, str]] = []
        error_policy = str(self.config.executor.get("source_error_policy", "continue")).strip().lower()
        for source, retriever in self.retrievers.items():
            logger.info(f"正在从 {source} 拉取…")
            try:
                papers = retriever.retrieve_papers()
            except Exception as exc:
                failed_sources.append((source, str(exc)))
                logger.error(f"{source} 拉取失败：{exc}")
                if error_policy == "fail_fast":
                    raise
                logger.warning(f"按 source_error_policy={error_policy} 继续处理其它来源。")
                continue
            if not papers:
                logger.info(f"{source} 无新稿")
                continue
            logger.info(f"从 {source} 得到 {len(papers)} 篇")
            all_papers.extend(papers)

        if failed_sources:
            failed_text = "; ".join([f"{s}: {err}" for s, err in failed_sources])
            logger.warning(f"以下来源失败并已跳过：{failed_text}")

        logger.info(f"候选论文总数：{len(all_papers)}")
        all_papers, _dedupe_stats = apply_dedupe_pipeline(all_papers, corpus, self.config.executor)
        if _dedupe_stats.get("merged") or _dedupe_stats.get("library_dropped"):
            logger.info(
                "去重统计：合并重复 {} 篇，书库剔除 {} 篇；剩余候选 {} 篇。".format(
                    _dedupe_stats.get("merged", 0),
                    _dedupe_stats.get("library_dropped", 0),
                    len(all_papers),
                )
            )
        reranked: list = []
        feedback_links: dict[str, dict[str, str]] = {}
        briefing_intro: str | None = None
        if all_papers:
            logger.info("按与书库的向量相似度重排中…")
            reranked = self.reranker.rerank(all_papers, corpus)
            fb_cfg = self.config.get("feedback")
            if fb_cfg and bool(fb_cfg.get("enabled", False)):
                user_id = hash_user_id(
                    str(self.config.email.receiver),
                    str(fb_cfg.get("user_id_salt", "zotero-tracker")),
                )
                apply_feedback_reweight(self.config, reranked, user_id)
            min_score_cfg = self.config.executor.get("min_score", None)
            if min_score_cfg is not None:
                min_score = float(min_score_cfg)
                before_filter = len(reranked)
                reranked = [p for p in reranked if p.score is not None and p.score >= min_score]
                logger.info(
                    "已按 min_score={} 过滤：{} -> {} 篇。".format(
                        min_score, before_filter, len(reranked)
                    )
                )
            reranked = reranked[: int(self.config.executor.max_paper_num)]
            for p in reranked:
                p.matched_keywords = match_keywords_in_paper(kw.terms, p.title, p.abstract)
            logger.info("正在生成一句话摘要…")
            for p in tqdm(reranked):
                p.generate_tldr(self.openai_client, self.config.llm)
            ne_cfg = self.config.llm.get("natural_explain") or {}
            if bool(ne_cfg.get("enabled", False)):
                max_ne = ne_cfg.get("max_papers")
                to_explain = reranked if max_ne is None else reranked[: int(max_ne)]
                logger.info(f"正在生成自然语言推荐解读（{len(to_explain)} 篇）…")
                for p in tqdm(to_explain):
                    p.fill_natural_explain(self.openai_client, self.config.llm)
            br_cfg = self.config.llm.get("briefing") or {}
            if bool(br_cfg.get("enabled", False)) and reranked:
                logger.info("正在生成今日简报导语…")
                date_label = datetime.now().strftime("%Y-%m-%d")
                briefing_intro = fill_briefing_intro(
                    self.openai_client,
                    self.config.llm,
                    reranked,
                    kw.terms,
                    date_label,
                )
            if fb_cfg and bool(fb_cfg.get("enabled", False)):
                base_url = str(fb_cfg.get("base_url", "")).strip()
                secret = str(fb_cfg.get("secret", "")).strip()
                if base_url and secret:
                    user_id = hash_user_id(
                        str(self.config.email.receiver),
                        str(fb_cfg.get("user_id_salt", "zotero-tracker")),
                    )
                    push_id = datetime.now().strftime("%Y%m%d") + "-" + uuid4().hex[:10]
                    ts = int(datetime.now().timestamp())
                    for p in reranked:
                        item_id = paper_item_id(p)
                        links = build_feedback_links(
                            base_url=base_url,
                            secret=secret,
                            user_id=user_id,
                            push_id=push_id,
                            item_id=item_id,
                            ts=ts,
                            source=p.source,
                            tags=(p.tags or []),
                        )
                        feedback_links[item_id] = {k: v.url for k, v in links.items()}
                else:
                    logger.warning("feedback.enabled=true，但 base_url 或 secret 缺失，已跳过反馈链接生成。")
        elif not self.config.executor.send_empty:
            logger.info("无候选论文且 send_empty=false，不发送邮件。")
            return None

        md = render_markdown(
            reranked,
            kw,
            feedback_links=feedback_links,
            briefing_intro=briefing_intro,
        )
        html = render_html(
            reranked,
            kw,
            feedback_links=feedback_links,
            briefing_intro=briefing_intro,
        )
        logger.info("正在发送邮件…")
        send_markdown_email(self.config, md, html_body=html)
        logger.info("全部完成。")
        return RunResult(papers=reranked, keywords=kw)
