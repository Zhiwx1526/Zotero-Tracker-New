from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import tiktoken
from loguru import logger
from openai import OpenAI


def _lang_is_chinese(lang: Any) -> bool:
    s = str(lang or "").strip().lower()
    return "中文" in str(lang or "") or "chinese" in s or s in ("zh", "cn", "简体", "繁体")


def _llm_get(llm_params: Any, key: str, default: Any = None) -> Any:
    if hasattr(llm_params, "get"):
        return llm_params.get(key, default)
    return getattr(llm_params, key, default)


def _llm_lang_display(lang_raw: Any) -> tuple[str, bool]:
    """返回 (展示用语言说明, 是否按中文提示词)."""
    lang_s = str(lang_raw).strip()
    is_zh = _lang_is_chinese(lang_raw)
    if lang_s.lower() in ("zh", "cn"):
        lang_display = "简体中文"
    elif is_zh:
        lang_display = lang_s
    else:
        lang_display = lang_s
    return lang_display, is_zh


def _llm_chat_completion(
    openai_client: OpenAI,
    llm_params: Any,
    system: str,
    user: str,
    *,
    max_prompt_tokens: int = 4000,
) -> str:
    enc = tiktoken.encoding_for_model("gpt-4o")
    user_tokens = enc.encode(user)[:max_prompt_tokens]
    user = enc.decode(user_tokens)
    gen_kw = _llm_get(llm_params, "generation_kwargs", {}) or {}
    response = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **gen_kw,
    )
    return response.choices[0].message.content or ""


@dataclass
class CorpusMatchExplain:
    """单条书库文献对候选论文得分的解释项（按对总分的贡献排序）。"""

    item_key: str
    title: str
    cosine_sim: float
    time_weight: float
    contribution: float
    collection_path: Optional[str] = None


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    tldr: Optional[str] = None
    score: Optional[float] = None
    item_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    corpus_explanations: list[CorpusMatchExplain] = field(default_factory=list)
    natural_explain: Optional[str] = None
    doi: Optional[str] = None
    merged_sources: list[str] = field(default_factory=list)
    journal_name: Optional[str] = None
    source_authority_score: Optional[float] = None
    quality_score: Optional[float] = None
    score_breakdown: dict[str, float] = field(default_factory=dict)
    rag_used: bool = False
    rag_references: list[str] = field(default_factory=list)

    def _generate_tldr_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: Any,
        *,
        rag_context_text: str | None = None,
    ) -> str:
        # 配置里默认用 zh（避免 OmegaConf 在 ${oc.env:...,中文} 里解析失败）；环境变量可写 简体中文
        lang_raw = _llm_get(llm_params, "language", "zh")
        lang_display, is_zh = _llm_lang_display(lang_raw)

        if not self.abstract:
            logger.warning(f"无摘要，无法生成 TLDR：{self.url}")
            return "无法生成摘要：论文无摘要。" if is_zh else "Failed to generate TLDR: no abstract."

        if is_zh:
            prompt = f"请根据以下论文信息，用「{lang_display}」写**一句**简明摘要（单句，不要分点）：\n\n"
            if self.title:
                prompt += f"标题：\n{self.title}\n\n"
            prompt += f"摘要：\n{self.abstract}\n\n"
            if rag_context_text:
                prompt += (
                    "补充专业知识（来自检索知识库，仅可作为术语与背景参考；若与论文摘要冲突，以论文摘要为准）：\n"
                    f"{rag_context_text}\n\n"
                )
            system = (
                "你是学术文献助手，擅长用一句话概括论文核心贡献与方法。"
                f"请严格使用「{lang_display}」作答，不要输出英文（除非原文专有名词必要）。"
            )
        else:
            prompt = (
                f"Given the following paper information, generate a one-sentence TLDR summary in {lang_display}:\n\n"
            )
            if self.title:
                prompt += f"Title:\n {self.title}\n\n"
            prompt += f"Abstract: {self.abstract}\n\n"
            if rag_context_text:
                prompt += (
                    "Domain context from retrieval (use only as terminology/background hints; "
                    "if it conflicts with paper abstract, trust the paper abstract):\n"
                    f"{rag_context_text}\n\n"
                )
            system = (
                "You are an assistant who summarizes scientific papers in one sentence. "
                f"Answer in {lang_display}."
            )

        return _llm_chat_completion(openai_client, llm_params, system, prompt)

    def generate_tldr(
        self,
        openai_client: OpenAI,
        llm_params: Any,
        *,
        rag_context_text: str | None = None,
        rag_references: list[str] | None = None,
    ) -> str:
        try:
            self.rag_used = bool(rag_context_text)
            self.rag_references = list(rag_references or [])
            self.tldr = self._generate_tldr_with_llm(
                openai_client,
                llm_params,
                rag_context_text=rag_context_text,
            )
            return self.tldr
        except Exception as e:
            logger.warning(f"生成 TLDR 失败 {self.url}: {e}")
            self.tldr = self.abstract[:500] if self.abstract else ""
            return self.tldr

    def _natural_explain_prompt(
        self,
        lang_display: str,
        is_zh: bool,
        *,
        rag_context_text: str | None = None,
        corpus_evidence_text: str | None = None,
    ) -> tuple[str, str]:
        title = (self.title or "").strip()
        abstract = (self.abstract or "").strip()
        kw = ", ".join(self.matched_keywords) if self.matched_keywords else ("（无）" if is_zh else "(none)")

        if self.corpus_explanations:
            if is_zh:
                lines = ["以下为系统根据向量相似度与时间权重选出的书库关联条目（请据此解释，勿编造其它书库文献）：", ""]
                for k, ex in enumerate(self.corpus_explanations, start=1):
                    path_s = f"；集合路径：{ex.collection_path}" if ex.collection_path else ""
                    lines.append(
                        f"{k}. 书库标题：{ex.title}{path_s}；余弦相似度 {ex.cosine_sim:.3f}；"
                        f"时间权重 {ex.time_weight:.4f}；贡献 {ex.contribution:.3f}"
                    )
                corpus_block = "\n".join(lines)
            else:
                lines = [
                    "Library items selected by the system (base explanations only; "
                    "do not invent other library titles):",
                    "",
                ]
                for k, ex in enumerate(self.corpus_explanations, start=1):
                    path_s = f"; collection: {ex.collection_path}" if ex.collection_path else ""
                    lines.append(
                        f"{k}. Title: {ex.title}{path_s}; cosine {ex.cosine_sim:.3f}; "
                        f"time_weight {ex.time_weight:.4f}; contribution {ex.contribution:.3f}"
                    )
                corpus_block = "\n".join(lines)
        else:
            corpus_block = (
                "（当前未提供书库分解条目；请仅依据下方关键词与摘要说明可能的相关性，不要捏造具体书库论文标题。）"
                if is_zh
                else (
                    "(No per-library breakdown was provided; explain only from keywords and abstract below. "
                    "Do not invent specific library paper titles.)"
                )
            )

        if is_zh:
            user = (
                f"请用「{lang_display}」按三段结构写推荐原因（每段 1-2 句）：\n"
                "1) 为什么前沿（问题/方法/趋势）；\n"
                "2) 与用户研究方向的关联；\n"
                "3) 对课题组/后续研究的参考意义（可执行建议）。\n"
                "要求：只能基于下方证据，不要编造未出现的书库文献或结论。\n\n"
                f"候选论文标题：\n{title or '（无）'}\n\n"
                f"候选论文摘要：\n{abstract or '（无）'}\n\n"
                f"命中展示关键词：{kw}\n\n"
                f"{corpus_block}\n"
            )
            if corpus_evidence_text:
                user += f"\n书库前序证据（Top 2-3）：\n{corpus_evidence_text}\n"
            if rag_context_text:
                user += (
                    "\n可用领域知识（检索得到，仅作为术语与背景参考；若与当前论文摘要冲突，以当前论文摘要为准）：\n"
                    f"{rag_context_text}\n"
                )
            system = (
                "你是学术文献推荐助手，擅长用简短自然语言解释个性化推荐依据。"
                f"请严格使用「{lang_display}」作答。"
            )
        else:
            user = (
                f"In {lang_display}, write recommendation reasons in exactly 3 short sections: "
                "(1) why this paper is frontier, (2) relation to user direction, "
                "(3) practical value for the lab/future work.\n"
                "Use only evidence below; do not invent missing library papers.\n\n"
                f"Candidate title:\n{title or '(none)'}\n\n"
                f"Candidate abstract:\n{abstract or '(none)'}\n\n"
                f"Matched keywords: {kw}\n\n"
                f"{corpus_block}\n"
            )
            if corpus_evidence_text:
                user += f"\nTop prior library evidence (2-3):\n{corpus_evidence_text}\n"
            if rag_context_text:
                user += (
                    "\nDomain context from retrieval (terminology/background hints only; "
                    "if conflicts with candidate abstract, trust the candidate abstract):\n"
                    f"{rag_context_text}\n"
                )
            system = (
                "You explain personalized academic paper recommendations in clear, concise language. "
                f"Answer in {lang_display}."
            )
        return system, user

    def generate_natural_explain(
        self,
        openai_client: OpenAI,
        llm_params: Any,
        *,
        rag_context_text: str | None = None,
        rag_references: list[str] | None = None,
        corpus_evidence_text: str | None = None,
    ) -> str:
        lang_raw = _llm_get(llm_params, "language", "zh")
        lang_display, is_zh = _llm_lang_display(lang_raw)
        self.rag_used = bool(rag_context_text)
        self.rag_references = list(rag_references or [])
        system, user = self._natural_explain_prompt(
            lang_display,
            is_zh,
            rag_context_text=rag_context_text,
            corpus_evidence_text=corpus_evidence_text,
        )
        return _llm_chat_completion(openai_client, llm_params, system, user, max_prompt_tokens=6000)

    def fill_natural_explain(
        self,
        openai_client: OpenAI,
        llm_params: Any,
        *,
        rag_context_text: str | None = None,
        rag_references: list[str] | None = None,
        corpus_evidence_text: str | None = None,
    ) -> str:
        try:
            self.natural_explain = self.generate_natural_explain(
                openai_client,
                llm_params,
                rag_context_text=rag_context_text,
                rag_references=rag_references,
                corpus_evidence_text=corpus_evidence_text,
            )
            return self.natural_explain or ""
        except Exception as e:
            logger.warning(f"生成推荐解读失败 {self.url}: {e}")
            self.natural_explain = None
            return ""


def _briefing_venue_label(paper: Paper) -> str:
    """有期刊名用期刊名；否则用数据来源作为「来源」统计桶。"""
    j = (paper.journal_name or "").strip()
    if j:
        return j
    src = (paper.source or "").strip() or "unknown"
    return f"来源（{src}）"


def build_briefing_intro_from_papers(
    papers: list[Paper],
    *,
    date_label: str,
    lang_raw: Any,
) -> str:
    """邮件「今日简报」导语：本次推荐篇数 + 期刊/会议或来源分布（确定性，不调用 LLM）。"""
    if not papers:
        return ""
    _, is_zh = _llm_lang_display(lang_raw)
    counts = Counter(_briefing_venue_label(p) for p in papers)
    total = len(papers)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    if is_zh:
        parts = [f"{name} {cnt} 篇" for name, cnt in ordered]
        detail = "；".join(parts)
        return (
            f"推送日期：{date_label}。\n\n"
            f"本次推荐文献共 {total} 篇。按期刊、会议或来源统计：{detail}。"
        )

    parts_en = [f"{name}: {cnt}" for name, cnt in ordered]
    detail_en = "; ".join(parts_en)
    return (
        f"Digest date: {date_label}.\n\n"
        f"This digest lists {total} recommended papers. By venue or source: {detail_en}."
    )


def fill_briefing_intro(
    openai_client: OpenAI,
    llm_params: Any,
    papers: list[Paper],
    keyword_terms: list[str],
    date_label: str,
) -> str | None:
    try:
        if not papers:
            return None
        lang_raw = _llm_get(llm_params, "language", "zh")
        return build_briefing_intro_from_papers(papers, date_label=date_label, lang_raw=lang_raw)
    except Exception as e:
        logger.warning(f"生成简报导语失败: {e}")
        return None


@dataclass
class CorpusPaper:
    item_key: str
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
    doi: Optional[str] = None
