from omegaconf import OmegaConf
from openai import OpenAI

from zotero_tracker.rag.chunking import chunk_text
from zotero_tracker.rag.ingest import collect_knowledge_chunks
from zotero_tracker.rag.retrieve import RagRetriever


def test_chunk_text_overlap():
    chunks = chunk_text(
        source_path="x.md",
        title="X",
        text="A" * 1000,
        chunk_size=300,
        chunk_overlap=100,
    )
    assert len(chunks) >= 3
    assert chunks[0].source_path == "x.md"


def test_collect_knowledge_chunks_txt(tmp_path):
    f = tmp_path / "guide.txt"
    f.write_text("first paragraph\n\nsecond paragraph", encoding="utf-8")
    chunks = collect_knowledge_chunks([str(f)], chunk_size=80, chunk_overlap=20)
    assert chunks
    assert chunks[0].title == "guide"


def test_rag_retrieve_none_when_empty_paths():
    cfg = OmegaConf.create(
        {
            "executor": {"rag": {"enabled": True, "top_k": 3, "min_retrieval_score": 0.2, "max_context_chars": 500}},
            "rag": {"knowledge_base": {"paths": [], "index_path": "cache/test_rag_index.json", "embedding_model": "x"}},
            "reranker": {"api": {"model": "x"}},
        }
    )
    dummy_client = object.__new__(OpenAI)  # 不触发网络调用，仅用于构造类型
    retriever = RagRetriever(cfg, dummy_client)  # type: ignore[arg-type]
    assert retriever.retrieve(title="t", abstract="a") is None
