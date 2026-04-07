from .base import get_retriever_cls
from . import arxiv  # noqa: F401 — 注册 arxiv
from . import openalex  # noqa: F401 — 注册 openalex
from . import placeholder  # noqa: F401 — 注册占位来源

__all__ = ["get_retriever_cls"]
