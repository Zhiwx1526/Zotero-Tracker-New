# Zotero-Tracker

根据你的 Zotero 书库，对每日新增 arXiv 论文做兴趣对齐排序（embedding 相似度），并从书库中抽取 TF-IDF 兴趣关键词用于展示；最终以 **Markdown 纯文本** 邮件推送。

## 环境准备

```bash
cd Zotero-Tracker
uv sync
```

可参考 `config/custom.yaml` 的写法，或通过环境变量配置（如 `ZOTERO_ID`、`ZOTERO_KEY`、`OPENAI_API_KEY`、`OPENAI_API_BASE`、邮箱与 SMTP 等）。默认 **邮件标题、正文标签与 TLDR** 为中文（配置里用 `zh`，避免 YAML 插值里写中文触发 OmegaConf 报错）。若要用英文摘要，可设 `LLM_LANGUAGE=English` 或 Hydra：`llm.language=English`。也可在 `.env` 里设 `LLM_LANGUAGE=简体中文`（从环境读入，不经 `${oc.env:...,中文}` 解析）。

## 运行

```bash
uv run python -m zotero_tracker.main
```

Hydra 覆盖示例：`uv run python -m zotero_tracker.main executor.debug=true`
