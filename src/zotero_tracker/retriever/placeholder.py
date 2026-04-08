"""占位来源：后续再接 bioRxiv / medRxiv 等实现。"""

from typing import Any

from loguru import logger

from ..protocol import Paper
from .base import BaseRetriever, register_retriever


def _stub(name: str):
    @register_retriever(name)
    class _StubRetriever(BaseRetriever):
        def _retrieve_raw_papers(self) -> list[Any]:
            logger.warning(f'来源「{name}」尚未实现，已跳过。')
            return []

        def convert_to_paper(self, raw_paper: Any) -> Paper | None:
            return None

    _StubRetriever.__name__ = f"{name.title()}Placeholder"
    return _StubRetriever


