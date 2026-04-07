from abc import ABC, abstractmethod
from time import sleep
from typing import Any, Type

from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from ..protocol import Paper

registered_retrievers: dict[str, type["BaseRetriever"]] = {}


def register_retriever(name: str):
    def decorator(cls):
        registered_retrievers[name] = cls
        cls.name = name
        return cls

    return decorator


def get_retriever_cls(name: str) -> Type["BaseRetriever"]:
    if name not in registered_retrievers:
        raise ValueError(f"未找到来源「{name}」。已注册：{list(registered_retrievers)}")
    return registered_retrievers[name]


class BaseRetriever(ABC):
    name: str

    def __init__(self, config: DictConfig):
        self.config = config
        self.retriever_config = getattr(config.source, self.name)

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[Any]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper: Any) -> Paper | None:
        pass

    def retrieve_papers(self) -> list[Paper]:
        raw_papers = self._retrieve_raw_papers()
        sleep_sec = float(self.config.executor.get("retriever_sleep_seconds", 1))
        logger.info("正在转换原始记录为 Paper…")
        papers: list[Paper] = []
        for raw_paper in tqdm(raw_papers, total=len(raw_papers), desc=f"{self.name}"):
            try:
                paper = self.convert_to_paper(raw_paper)
            except Exception as exc:
                logger.warning(f"跳过论文 {getattr(raw_paper, 'title', raw_paper)}：{exc}")
                continue
            if paper is not None:
                papers.append(paper)
            if sleep_sec > 0:
                sleep(sleep_sec)
        return papers
