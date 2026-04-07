"""从 Zotero 书库抽取 TF-IDF 关键词（仅用于日报展示，不参与 embedding 打分）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from omegaconf import DictConfig
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .protocol import CorpusPaper


@dataclass
class KeywordResult:
    terms: list[str]
    scores: list[float]


def extract_keywords_from_corpus(
    corpus: list[CorpusPaper],
    kw_config: DictConfig | dict,
) -> KeywordResult:
    """将每条文献的「标题+摘要」拼成文档，返回 TF-IDF 分数最高的若干词/短语。"""
    if not corpus:
        return KeywordResult(terms=[], scores=[])

    top_k = int(kw_config.get("top_k", 20))
    max_features = int(kw_config.get("max_features", 5000))
    ngram_max = int(kw_config.get("ngram_max", 2))

    docs = [f"{c.title}\n{c.abstract}" for c in corpus]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(ENGLISH_STOP_WORDS),
        ngram_range=(1, ngram_max),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]+\b",
    )
    try:
        tfidf = vectorizer.fit_transform(docs)
    except ValueError as e:
        logger.warning(f"TF-IDF 失败（词表为空？）：{e}")
        return KeywordResult(terms=[], scores=[])

    # 对各文档的 TF-IDF 取平均，便于解释「整库」关键词
    mean_scores = np.asarray(tfidf.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(-mean_scores)
    order = order[:top_k]
    picked_terms = terms[order].tolist()
    picked_scores = mean_scores[order].tolist()
    return KeywordResult(terms=picked_terms, scores=picked_scores)
