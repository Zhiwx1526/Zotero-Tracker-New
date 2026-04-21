"""分级 DOI + 标题/摘要指纹去重。"""

from __future__ import annotations

from datetime import datetime

from zotero_tracker.dedupe import (
    apply_dedupe_pipeline,
    dedupe_enabled,
    filter_against_library,
    merge_paper_list,
    normalize_doi,
    normalize_title,
    paper_dedupe_keys,
)
from zotero_tracker.protocol import CorpusPaper, Paper


def test_normalize_doi_prefixes() -> None:
    assert normalize_doi("10.1000/182") == "10.1000/182"
    assert normalize_doi("  HTTPS://DOI.ORG/10.1000/182  ") == "10.1000/182"
    assert normalize_doi("doi:10.1000/182") == "10.1000/182"
    assert normalize_doi("http://dx.doi.org/10.1000/182?foo=1") == "10.1000/182"
    assert normalize_doi("") is None
    assert normalize_doi(None) is None
    assert normalize_doi("not-a-doi") is None


def test_title_fingerprint_stable_whitespace() -> None:
    a = paper_dedupe_keys(
        Paper(
            source="x",
            title="  Hello   World  ",
            authors=[],
            abstract="Same abstract here.",
            url="u",
        )
    )
    b = paper_dedupe_keys(
        Paper(
            source="x",
            title="hello world",
            authors=[],
            abstract="Same abstract here.",
            url="u",
        )
    )
    assert a.title_fp == b.title_fp


def test_merge_t1_same_doi_different_titles() -> None:
    cfg = {
        "dedupe": {
            "enabled": True,
            "merge_within_run": True,
            "min_abstract_len": 10,
        }
    }
    p1 = Paper(
        source="arxiv",
        title="Title A",
        authors=["A"],
        abstract="Abstract one " * 20,
        url="http://a",
        doi="10.1000/182",
    )
    p2 = Paper(
        source="openalex",
        title="Title B other",
        authors=["B"],
        abstract="Abstract two " * 20,
        url="http://b",
        doi="https://doi.org/10.1000/182",
    )
    out, removed = merge_paper_list([p1, p2], cfg)
    assert len(out) == 1
    assert removed == 1
    assert normalize_doi(out[0].doi) == "10.1000/182"
    assert out[0].source == "openalex"
    assert "merged_from:arxiv,openalex" in out[0].tags


def test_merge_t2_no_doi_matching_fingerprints() -> None:
    long_abs = "x" * 100
    cfg = {
        "dedupe": {
            "enabled": True,
            "merge_within_run": True,
            "min_abstract_len": 80,
        }
    }
    p1 = Paper(source="arxiv", title="Same", authors=[], abstract=long_abs, url="u1", doi=None)
    p2 = Paper(source="openalex", title="same", authors=[], abstract=long_abs, url="u2", doi=None)
    out, removed = merge_paper_list([p1, p2], cfg)
    assert len(out) == 1
    assert removed == 1


def test_merge_t2_short_abstract_no_merge() -> None:
    short_abs = "y" * 40
    cfg = {
        "dedupe": {
            "enabled": True,
            "merge_within_run": True,
            "min_abstract_len": 80,
        }
    }
    p1 = Paper(source="arxiv", title="Same", authors=[], abstract=short_abs, url="u1")
    p2 = Paper(source="openalex", title="same", authors=[], abstract=short_abs, url="u2")
    out, removed = merge_paper_list([p1, p2], cfg)
    assert len(out) == 2
    assert removed == 0


def test_merge_t2_conflicting_doi_no_merge() -> None:
    long_abs = "z" * 100
    cfg = {
        "dedupe": {
            "enabled": True,
            "merge_within_run": True,
            "min_abstract_len": 80,
        }
    }
    p1 = Paper(
        source="a",
        title="Same",
        authors=[],
        abstract=long_abs,
        url="u1",
        doi="10.1000/aaa",
    )
    p2 = Paper(
        source="b",
        title="same",
        authors=[],
        abstract=long_abs,
        url="u2",
        doi="10.1000/bbb",
    )
    out, removed = merge_paper_list([p1, p2], cfg)
    assert len(out) == 2
    assert removed == 0


def test_library_filter_t1() -> None:
    long_abs = "w" * 100
    corpus = [
        CorpusPaper(
            item_key="k1",
            title="In library",
            abstract=long_abs,
            added_date=datetime(2024, 1, 1),
            paths=[],
            doi="10.9999/lib",
        )
    ]
    cand = Paper(
        source="arxiv",
        title="Different title",
        authors=[],
        abstract="other " * 50,
        url="u",
        doi="10.9999/lib",
    )
    cfg = {"dedupe": {"filter_against_library": True, "min_abstract_len": 80}}
    out, n = filter_against_library([cand], corpus, cfg)
    assert out == []
    assert n == 1


def test_library_filter_t2_fingerprint() -> None:
    long_abs = "q" * 100
    corpus = [
        CorpusPaper(
            item_key="k1",
            title="  My Title  ",
            abstract=long_abs,
            added_date=datetime(2024, 1, 1),
            paths=[],
            doi=None,
        )
    ]
    cand = Paper(
        source="arxiv",
        title="my title",
        authors=[],
        abstract=long_abs,
        url="u",
        doi=None,
    )
    cfg = {"dedupe": {"filter_against_library": True, "min_abstract_len": 80}}
    out, n = filter_against_library([cand], corpus, cfg)
    assert out == []
    assert n == 1


def test_apply_disabled_no_change() -> None:
    papers = [
        Paper(source="a", title="t", authors=[], abstract="x" * 100, url="1"),
        Paper(source="b", title="t", authors=[], abstract="x" * 100, url="2"),
    ]
    corpus: list[CorpusPaper] = []
    cfg = {"dedupe": {"enabled": False}}
    out, stats = apply_dedupe_pipeline(papers, corpus, cfg)
    assert len(out) == 2
    assert stats == {"merged": 0, "library_dropped": 0}


def test_dedupe_enabled_helper() -> None:
    assert not dedupe_enabled({"dedupe": {"enabled": False}})
    assert dedupe_enabled({"dedupe": {"enabled": True}})


def test_normalize_abstract_strips_tags() -> None:
    from zotero_tracker.dedupe import normalize_abstract

    assert "script" not in normalize_abstract("<script>x</script>Hello")


def test_allow_title_only_merge() -> None:
    cfg = {
        "dedupe": {
            "enabled": True,
            "merge_within_run": True,
            "min_abstract_len": 80,
            "allow_title_only": True,
            "min_title_len_weak": 4,
        }
    }
    p1 = Paper(source="a", title="UniqueTitle", authors=[], abstract="a" * 30, url="1")
    p2 = Paper(source="b", title="uniquetitle", authors=[], abstract="b" * 30, url="2")
    out, removed = merge_paper_list([p1, p2], cfg)
    assert len(out) == 1
    assert removed == 1
