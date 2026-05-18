from __future__ import annotations

import argparse
import json
from pathlib import Path

import dotenv
from loguru import logger
from omegaconf import OmegaConf
from openai import OpenAI

from ..protocol import Paper
from .retrieve import RagRetriever


def _load_samples(path: str) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run_eval(config_path: str, samples_path: str, output_path: str, limit: int | None) -> None:
    dotenv.load_dotenv()
    cfg = OmegaConf.load(config_path)
    client = OpenAI(api_key=cfg.llm.api.key, base_url=cfg.llm.api.base_url)
    rag = RagRetriever(cfg, client)
    samples = _load_samples(samples_path)
    if limit is not None:
        samples = samples[: max(1, int(limit))]

    out_rows: list[dict] = []
    for idx, row in enumerate(samples, start=1):
        title = str(row.get("title") or "").strip()
        abstract = str(row.get("abstract") or "").strip()
        if not title and not abstract:
            continue
        paper = Paper(source="eval", title=title, authors=[], abstract=abstract, url=f"eval://{idx}")
        baseline = paper.generate_tldr(client, cfg.llm)
        rag_ctx = rag.retrieve(title=title, abstract=abstract)
        rag_tldr = paper.generate_tldr(
            client,
            cfg.llm,
            rag_context_text=(rag_ctx.context_text if rag_ctx else None),
            rag_references=[f"{h.title} ({h.score:.3f})" for h in (rag_ctx.hits if rag_ctx else [])],
        )
        out_rows.append(
            {
                "title": title,
                "baseline_tldr": baseline,
                "rag_tldr": rag_tldr,
                "rag_refs": paper.rag_references,
                "rag_used": bool(rag_ctx),
            }
        )
        logger.info("评估样本 {}/{}：rag_used={}", idx, len(samples), bool(rag_ctx))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    logger.info("RAG 评估输出已写入：{}", str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs RAG TLDR generation.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--samples", required=True, help="JSONL with title/abstract fields.")
    parser.add_argument("--output", default="outputs/rag_eval.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_eval(args.config, args.samples, args.output, args.limit)


if __name__ == "__main__":
    main()
