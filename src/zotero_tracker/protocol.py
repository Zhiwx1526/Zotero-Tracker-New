from dataclasses import dataclass
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

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: Any) -> str:
        # 配置里默认用 zh（避免 OmegaConf 在 ${oc.env:...,中文} 里解析失败）；环境变量可写 简体中文
        lang_raw = _llm_get(llm_params, "language", "zh")
        lang_s = str(lang_raw).strip()
        is_zh = _lang_is_chinese(lang_raw)
        if lang_s.lower() in ("zh", "cn"):
            lang_display = "简体中文"
        elif is_zh:
            lang_display = lang_s
        else:
            lang_display = lang_s

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

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)[:4000]
        prompt = enc.decode(prompt_tokens)

        gen_kw = _llm_get(llm_params, "generation_kwargs", {}) or {}
        response = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **gen_kw,
        )
        return response.choices[0].message.content or ""

    def generate_tldr(self, openai_client: OpenAI, llm_params: Any) -> str:
        try:
            self.tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            return self.tldr
        except Exception as e:
            logger.warning(f"生成 TLDR 失败 {self.url}: {e}")
            self.tldr = self.abstract[:500] if self.abstract else ""
            return self.tldr


@dataclass
class CorpusPaper:
    item_key: str
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
