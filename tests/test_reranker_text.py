from zotero_tracker.reranker.base import _text_for_embedding


def test_text_for_embedding_normal():
    assert _text_for_embedding("  Title  ", "  Abstract body.  ") == "Title\nAbstract body."


def test_text_for_embedding_empty_title():
    assert _text_for_embedding("", "Only abstract") == "Only abstract"
    assert _text_for_embedding("   ", "Only abstract") == "Only abstract"


def test_text_for_embedding_empty_abstract():
    assert _text_for_embedding("Only title", "") == "Only title"


def test_text_for_embedding_both_empty():
    assert _text_for_embedding("", "") == ""
