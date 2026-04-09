"""根据相似度矩阵与时间衰减权重，为每篇候选论文附加书库分解说明。"""

from __future__ import annotations

import numpy as np

from ..protocol import CorpusMatchExplain, CorpusPaper, Paper


def _truncate_title(title: str, max_len: int) -> str:
    t = (title or "").strip()
    if max_len <= 0 or len(t) <= max_len:
        return t
    return t[: max(0, max_len - 1)] + "…"


def attach_corpus_explanations(
    candidates: list[Paper],
    corpus: list[CorpusPaper],
    time_decay_weight: np.ndarray,
    sim: np.ndarray,
    *,
    top_k: int,
    enabled: bool,
    title_max_len: int,
) -> None:
    """按对总分贡献（sim * w * 10）取 Top-K，写入各 Paper.corpus_explanations。"""
    if not enabled or top_k <= 0:
        for p in candidates:
            p.corpus_explanations = []
        return

    w = np.asarray(time_decay_weight, dtype=np.float64)
    n_corpus = len(corpus)
    if sim.shape != (len(candidates), n_corpus):
        raise ValueError(
            f"sim 形状应为 ({len(candidates)}, {n_corpus})，实际为 {sim.shape}"
        )
    if w.shape != (n_corpus,):
        raise ValueError(f"time_decay_weight 长度应为 {n_corpus}，实际为 {w.shape}")

    for row, paper in enumerate(candidates):
        contrib = sim[row].astype(np.float64) * w * 10.0
        k = min(top_k, n_corpus)
        if k == 0:
            paper.corpus_explanations = []
            continue
        idx = np.argpartition(-contrib, k - 1)[:k]
        idx = idx[np.argsort(-contrib[idx])]
        explains: list[CorpusMatchExplain] = []
        for j in idx:
            j = int(j)
            cp = corpus[j]
            path = cp.paths[0] if cp.paths else None
            explains.append(
                CorpusMatchExplain(
                    item_key=cp.item_key,
                    title=_truncate_title(cp.title, title_max_len),
                    cosine_sim=float(sim[row, j]),
                    time_weight=float(w[j]),
                    contribution=float(contrib[j]),
                    collection_path=path,
                )
            )
        paper.corpus_explanations = explains
