from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from omegaconf import DictConfig

from ..protocol import CorpusPaper, Paper

registered_rerankers: dict[str, type["BaseReranker"]] = {}


def register_reranker(name: str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls

    return decorator


def get_reranker_cls(name: str) -> Type["BaseReranker"]:
    if name not in registered_rerankers:
        raise ValueError(f"未找到重排器「{name}」。已注册：{list(registered_rerankers)}")
    return registered_rerankers[name]


class BaseReranker(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        # 书库中越新的条目权重越高（与 zotero-arxiv-daily 一致）
        corpus = sorted(corpus, key=lambda x: x.added_date, reverse=True)
        time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        time_decay_weight = time_decay_weight / time_decay_weight.sum()
        sim = self.get_similarity_score([c.abstract for c in candidates], [c.abstract for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        scores = (sim * time_decay_weight).sum(axis=1) * 10
        for s, c in zip(scores, candidates, strict=True):
            c.score = float(s)
        return sorted(candidates, key=lambda x: x.score or 0.0, reverse=True)

    @abstractmethod
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        raise NotImplementedError
