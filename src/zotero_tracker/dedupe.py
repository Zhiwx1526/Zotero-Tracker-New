"""分级去重：DOI（T1）+ 标题/摘要 SHA256 指纹（T2），可选仅标题/仅摘要弱规则。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

from .protocol import CorpusPaper, Paper

_ZW_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_TAG_RE = re.compile(r"<[^>]+>")


def normalize_doi(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.lower()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    ):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    s = s.split("?", 1)[0].strip().rstrip(" .)")
    if not s.startswith("10.") or "/" not in s:
        return None
    return s


def _strip_zw(s: str) -> str:
    return _ZW_RE.sub("", s)


def normalize_title(raw: str | None) -> str:
    if not raw:
        return ""
    t = unicodedata.normalize("NFKC", str(raw))
    t = _strip_zw(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.casefold()


def normalize_abstract(raw: str | None) -> str:
    if not raw:
        return ""
    t = unicodedata.normalize("NFKC", str(raw))
    t = _strip_zw(t)
    t = _TAG_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fingerprint(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperDedupeKeys:
    doi_key: str | None
    title_fp: str
    abstract_fp: str
    norm_abstract_len: int
    norm_title_len: int


def paper_dedupe_keys(p: Paper) -> PaperDedupeKeys:
    doi_key = normalize_doi(p.doi)
    nt = normalize_title(p.title)
    na = normalize_abstract(p.abstract)
    return PaperDedupeKeys(
        doi_key=doi_key,
        title_fp=fingerprint(nt),
        abstract_fp=fingerprint(na),
        norm_abstract_len=len(na),
        norm_title_len=len(nt),
    )


def corpus_dedupe_keys(c: CorpusPaper) -> PaperDedupeKeys:
    doi_key = normalize_doi(c.doi)
    nt = normalize_title(c.title)
    na = normalize_abstract(c.abstract)
    return PaperDedupeKeys(
        doi_key=doi_key,
        title_fp=fingerprint(nt),
        abstract_fp=fingerprint(na),
        norm_abstract_len=len(na),
        norm_title_len=len(nt),
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))
        self._r = [0] * n

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._r[ra] < self._r[rb]:
            ra, rb = rb, ra
        self._p[rb] = ra
        if self._r[ra] == self._r[rb]:
            self._r[ra] += 1


def _dedupe_cfg_block(executor_cfg: Any) -> dict[str, Any]:
    if executor_cfg is None:
        return {}
    block = executor_cfg.get("dedupe") if hasattr(executor_cfg, "get") else None
    if block is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(block):
            out = OmegaConf.to_container(block, resolve=True)
            return out if isinstance(out, dict) else {}
    except Exception:
        pass
    if isinstance(block, dict):
        return dict(block)
    return {}


def dedupe_enabled(executor_cfg: Any) -> bool:
    b = _dedupe_cfg_block(executor_cfg)
    return bool(b.get("enabled", False))


def _source_rank(source: str, preference: list[str]) -> int:
    try:
        return preference.index(source)
    except ValueError:
        return len(preference)


def _pick_canonical(cluster: list[Paper], preference: list[str]) -> Paper:
    def sort_key(p: Paper) -> tuple[int, int, int, str]:
        has_doi = 1 if normalize_doi(p.doi) else 0
        sr = _source_rank(p.source, preference)
        alen = len(p.abstract or "")
        return (-has_doi, sr, -alen, p.title or "")

    return sorted(cluster, key=sort_key)[0]


def _merge_cluster(cluster: list[Paper], preference: list[str]) -> Paper:
    if len(cluster) == 1:
        return cluster[0]
    base = _pick_canonical(cluster, preference)
    merged_sources = sorted({p.source for p in cluster})
    doi = next((normalize_doi(p.doi) for p in cluster if normalize_doi(p.doi)), normalize_doi(base.doi))
    richest = max(cluster, key=lambda p: len(p.abstract or ""))
    longest_auth = max(cluster, key=lambda p: len(p.authors or []))
    best_pdf = next(
        (p.pdf_url for p in sorted(cluster, key=lambda x: _source_rank(x.source, preference)) if p.pdf_url),
        base.pdf_url,
    )
    tags: set[str] = set()
    for p in cluster:
        tags.update(p.tags or [])
    tags.add(f"merged_from:{','.join(merged_sources)}")
    title = (richest.title or base.title) or base.title
    abstract = (richest.abstract or base.abstract) or base.abstract
    return Paper(
        source=base.source,
        title=title,
        authors=list(longest_auth.authors) if longest_auth.authors else list(base.authors),
        abstract=abstract,
        url=base.url,
        pdf_url=best_pdf,
        tldr=None,
        score=None,
        item_id=base.item_id,
        tags=sorted(tags),
        doi=doi,
        merged_sources=merged_sources,
    )


class LibraryDedupeIndex:
    """书库 DOI 与 (标题指纹, 摘要指纹) 集合，供候选过滤。"""

    def __init__(
        self,
        corpus: Iterable[CorpusPaper],
        *,
        min_abstract_len: int,
        allow_title_only: bool,
        allow_abstract_only: bool,
        min_title_len_weak: int = 24,
    ) -> None:
        self._min_abstract_len = max(0, int(min_abstract_len))
        self._allow_title_only = bool(allow_title_only)
        self._allow_abstract_only = bool(allow_abstract_only)
        self._min_title_len_weak = max(1, int(min_title_len_weak))
        self._doi_set: set[str] = set()
        self._fp_set: set[tuple[str, str]] = set()
        self._title_fp_set: set[str] = set()
        self._abstract_fp_set: set[str] = set()
        for c in corpus:
            k = corpus_dedupe_keys(c)
            if k.doi_key:
                self._doi_set.add(k.doi_key)
            if k.norm_abstract_len >= self._min_abstract_len:
                self._fp_set.add((k.title_fp, k.abstract_fp))
            if k.norm_title_len >= self._min_title_len_weak:
                self._title_fp_set.add(k.title_fp)
            if k.norm_abstract_len >= self._min_abstract_len:
                self._abstract_fp_set.add(k.abstract_fp)

    def is_duplicate(self, p: Paper) -> bool:
        k = paper_dedupe_keys(p)
        if k.doi_key and k.doi_key in self._doi_set:
            return True
        if (
            k.norm_abstract_len >= self._min_abstract_len
            and (k.title_fp, k.abstract_fp) in self._fp_set
        ):
            return True
        if self._allow_title_only and k.norm_title_len >= self._min_title_len_weak:
            if k.title_fp in self._title_fp_set:
                return True
        if self._allow_abstract_only and k.norm_abstract_len >= self._min_abstract_len:
            if k.abstract_fp in self._abstract_fp_set:
                return True
        return False


def merge_paper_list(papers: list[Paper], executor_cfg: Any) -> tuple[list[Paper], int]:
    """合并本轮候选（T1/T2/弱规则）；返回 (合并后列表, 去掉的篇数)。"""
    b = _dedupe_cfg_block(executor_cfg)
    if not papers or not bool(b.get("merge_within_run", True)):
        return papers, 0
    n = len(papers)
    keys = [paper_dedupe_keys(p) for p in papers]
    uf = _UnionFind(n)
    min_abs = max(0, int(b.get("min_abstract_len", 80)))
    allow_title = bool(b.get("allow_title_only", False))
    allow_abs_only = bool(b.get("allow_abstract_only", False))
    min_title_weak = max(1, int(b.get("min_title_len_weak", 24)))

    # T1: same DOI
    doi_buckets: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        if k.doi_key:
            doi_buckets.setdefault(k.doi_key, []).append(i)
    for idxs in doi_buckets.values():
        for j in range(1, len(idxs)):
            uf.union(idxs[0], idxs[j])

    # T2: title + abstract fingerprint, both abstracts long enough; pair must not be T1-only exclusion — union all valid pairs in bucket
    fp_buckets: dict[tuple[str, str], list[int]] = {}
    for i, k in enumerate(keys):
        fp_buckets.setdefault((k.title_fp, k.abstract_fp), []).append(i)
    for idxs in fp_buckets.values():
        if len(idxs) < 2:
            continue
        good = [i for i in idxs if keys[i].norm_abstract_len >= min_abs]
        if len(good) < 2:
            continue
        for a in range(len(good)):
            for bb in range(a + 1, len(good)):
                ia, ib = good[a], good[bb]
                da, db = keys[ia].doi_key, keys[ib].doi_key
                if da and db and da != db:
                    continue
                uf.union(ia, ib)

    if allow_title:
        tb: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            if k.norm_title_len >= min_title_weak:
                tb.setdefault(k.title_fp, []).append(i)
        for idxs in tb.values():
            if len(idxs) < 2:
                continue
            for j in range(1, len(idxs)):
                uf.union(idxs[0], idxs[j])

    if allow_abs_only:
        ab: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            if k.norm_abstract_len >= min_abs:
                ab.setdefault(k.abstract_fp, []).append(i)
        for idxs in ab.values():
            if len(idxs) < 2:
                continue
            for j in range(1, len(idxs)):
                uf.union(idxs[0], idxs[j])

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    preference = list(b.get("source_preference") or ["openalex", "arxiv", "biorxiv", "medrxiv"])
    out: list[Paper] = []
    removed = 0
    for idxs in groups.values():
        cluster = [papers[i] for i in idxs]
        if len(cluster) > 1:
            removed += len(cluster) - 1
            out.append(_merge_cluster(cluster, preference))
        else:
            out.append(cluster[0])

    if removed:
        logger.info(f"去重合并：{n} -> {len(out)} 篇（合并掉 {removed} 篇重复）")
    return out, removed


def filter_against_library(
    papers: list[Paper],
    corpus: list[CorpusPaper],
    executor_cfg: Any,
) -> tuple[list[Paper], int]:
    """剔除与书库 T1/T2（及可选弱规则）匹配的候选。"""
    b = _dedupe_cfg_block(executor_cfg)
    if not papers or not bool(b.get("filter_against_library", True)):
        return papers, 0
    min_abs = max(0, int(b.get("min_abstract_len", 80)))
    idx = LibraryDedupeIndex(
        corpus,
        min_abstract_len=min_abs,
        allow_title_only=bool(b.get("allow_title_only", False)),
        allow_abstract_only=bool(b.get("allow_abstract_only", False)),
        min_title_len_weak=int(b.get("min_title_len_weak", 24)),
    )
    out: list[Paper] = []
    dropped = 0
    for p in papers:
        if idx.is_duplicate(p):
            dropped += 1
            continue
        out.append(p)
    if dropped:
        logger.info(f"书库去重：剔除 {dropped} 篇已在 Zotero 的文献")
    return out, dropped


def apply_dedupe_pipeline(
    papers: list[Paper],
    corpus: list[CorpusPaper],
    executor_cfg: Any,
) -> tuple[list[Paper], dict[str, int]]:
    """enabled 时：先合并候选，再书库过滤。返回 (papers, stats)。"""
    if not dedupe_enabled(executor_cfg):
        return papers, {"merged": 0, "library_dropped": 0}
    merged_list, merged_n = merge_paper_list(papers, executor_cfg)
    filtered, lib_n = filter_against_library(merged_list, corpus, executor_cfg)
    return filtered, {"merged": merged_n, "library_dropped": lib_n}
