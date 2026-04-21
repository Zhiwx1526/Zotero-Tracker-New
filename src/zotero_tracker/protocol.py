from __future__ import annotations

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

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: Any) -> str:
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
            system = (
                "You are an assistant who summarizes scientific papers in one sentence. "
                f"Answer in {lang_display}."
            )

        return _llm_chat_completion(openai_client, llm_params, system, prompt)

    def generate_tldr(self, openai_client: OpenAI, llm_params: Any) -> str:
        try:
            self.tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            return self.tldr
        except Exception as e:
            logger.warning(f"生成 TLDR 失败 {self.url}: {e}")
            self.tldr = self.abstract[:500] if self.abstract else ""
            return self.tldr

    def _natural_explain_prompt(self, lang_display: str, is_zh: bool) -> tuple[str, str]:
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
                f"请用「{lang_display}」写 **2～4 句**（连续段落，不要分点列表），说明「为何会向用户推荐这篇候选论文」。\n"
                "要求：紧扣书库证据与关键词；不要复述摘要全文；不要编造未出现在书库条目列表中的文献标题。\n\n"
                f"候选论文标题：\n{title or '（无）'}\n\n"
                f"候选论文摘要：\n{abstract or '（无）'}\n\n"
                f"命中展示关键词：{kw}\n\n"
                f"{corpus_block}\n"
            )
            system = (
                "你是学术文献推荐助手，擅长用简短自然语言解释个性化推荐依据。"
                f"请严格使用「{lang_display}」作答。"
            )
        else:
            user = (
                f"In {lang_display}, write **2–4 sentences** (one short paragraph, no bullet lists) explaining "
                "why this candidate paper was recommended to the user.\n"
                "Ground the answer in the evidence below; do not paraphrase the entire abstract; "
                "do not invent library paper titles not listed.\n\n"
                f"Candidate title:\n{title or '(none)'}\n\n"
                f"Candidate abstract:\n{abstract or '(none)'}\n\n"
                f"Matched keywords: {kw}\n\n"
                f"{corpus_block}\n"
            )
            system = (
                "You explain personalized academic paper recommendations in clear, concise language. "
                f"Answer in {lang_display}."
            )
        return system, user

    def generate_natural_explain(self, openai_client: OpenAI, llm_params: Any) -> str:
        lang_raw = _llm_get(llm_params, "language", "zh")
        lang_display, is_zh = _llm_lang_display(lang_raw)
        system, user = self._natural_explain_prompt(lang_display, is_zh)
        return _llm_chat_completion(openai_client, llm_params, system, user, max_prompt_tokens=6000)

    def fill_natural_explain(self, openai_client: OpenAI, llm_params: Any) -> str:
        try:
            self.natural_explain = self.generate_natural_explain(openai_client, llm_params)
            return self.natural_explain or ""
        except Exception as e:
            logger.warning(f"生成推荐解读失败 {self.url}: {e}")
            self.natural_explain = None
            return ""


def generate_briefing_intro(
    openai_client: OpenAI,
    llm_params: Any,
    papers: list[Paper],
    keyword_terms: list[str],
    date_label: str,
    *,
    max_papers: int,
) -> str:
    if not papers:
        return ""
    lang_raw = _llm_get(llm_params, "language", "zh")
    lang_display, is_zh = _llm_lang_display(lang_raw)
    n = max(1, min(max_papers, len(papers)))
    slice_p = papers[:n]

    if is_zh:
        lines = [
            f"推送日期：{date_label}",
            f"用户书库兴趣关键词（展示用）：{', '.join(keyword_terms) if keyword_terms else '（无）'}",
            "",
            f"以下共 {len(slice_p)} 篇（邮件内顺序，最多列出 {n} 篇供你归纳）：",
            "",
        ]
        for i, p in enumerate(slice_p, start=1):
            tldr = (p.tldr or "").strip().replace("\n", " ")
            lines.append(f"{i}. 【{p.source}】{p.title}")
            lines.append(f"   一句话摘要：{tldr or '（无）'}")
            lines.append("")
        user = (
            "\n".join(lines)
            + f"\n请用「{lang_display}」写 **3～5 句** 简报导语（一段连续文字，不要分点）："
            "概括今日推送的主题倾向、与用户兴趣的关系，并可略微提示阅读顺序。"
            "不必逐篇点名；不要编造列表中未出现的论文内容。"
        )
        system = (
            "你是学术简报编辑，擅长为研究者写当日文献推送的开头导语。"
            f"请严格使用「{lang_display}」作答。"
        )
    else:
        lines = [
            f"Digest date: {date_label}",
            f"Corpus keyword hints: {', '.join(keyword_terms) if keyword_terms else '(none)'}",
            "",
            f"Up to {n} papers (email order):",
            "",
        ]
        for i, p in enumerate(slice_p, start=1):
            tldr = (p.tldr or "").strip().replace("\n", " ")
            lines.append(f"{i}. [{p.source}] {p.title}")
            lines.append(f"   TLDR: {tldr or '(none)'}")
            lines.append("")
        user = (
            "\n".join(lines)
            + f"\nIn {lang_display}, write **3–5 sentences** as an email opening briefing (one paragraph, "
            "no bullets): summarize themes, relation to the user's interests, optional light reading guidance. "
            "You need not mention every paper; do not invent content not implied by the list."
        )
        system = (
            "You write concise opening briefings for daily academic paper digests. "
            f"Answer in {lang_display}."
        )
    return _llm_chat_completion(openai_client, llm_params, system, user, max_prompt_tokens=6000)


def fill_briefing_intro(
    openai_client: OpenAI,
    llm_params: Any,
    papers: list[Paper],
    keyword_terms: list[str],
    date_label: str,
) -> str | None:
    try:
        b_cfg = _llm_get(llm_params, "briefing", {}) or {}
        mp = b_cfg.get("max_papers", 15) if hasattr(b_cfg, "get") else 15
        max_papers = int(mp) if mp is not None else 15
        return generate_briefing_intro(
            openai_client,
            llm_params,
            papers,
            keyword_terms,
            date_label,
            max_papers=max_papers,
        )
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
