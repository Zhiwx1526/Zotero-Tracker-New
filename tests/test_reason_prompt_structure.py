from zotero_tracker.protocol import Paper


def test_natural_explain_prompt_contains_three_sections():
    p = Paper(
        source="arxiv",
        title="New Method",
        authors=["A"],
        abstract="This paper proposes a method.",
        url="https://example.com/p",
    )
    system, user = p._natural_explain_prompt(
        lang_display="简体中文",
        is_zh=True,
        rag_context_text="RAG context",
        corpus_evidence_text="1. 证据A",
    )
    assert "三段结构" in user
    assert "为什么前沿" in user
    assert "研究方向" in user
    assert "参考意义" in user
    assert "RAG context" in user
    assert "证据A" in user
    assert "请严格使用" in system
