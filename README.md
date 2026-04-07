# Zotero-Tracker

根据你的 Zotero 书库，对新增论文做兴趣对齐排序（embedding 相似度），并从书库中抽取 TF-IDF 兴趣关键词用于展示；最终以 **Markdown 纯文本** 邮件推送。当前支持 `arxiv`，并可扩展 `openalex` 等来源。

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

## 多来源与平台开关

通过两层控制来源是否参与检索：

- `executor.source`：声明要尝试的来源列表。
- `source.<name>.enabled`：平台级开关（`true/false`）。

示例：

```yaml
source:
  arxiv:
    enabled: true
    category: ["cs.AI", "cs.LG"]
  openalex:
    enabled: true
    query: null
    search_title_only: false
    from_publication_date: null
    to_publication_date: null
    per_page: 50
    max_results: 200
    mailto: null

executor:
  source: ["arxiv", "openalex"]
```

说明：
- 若 `executor.source` 包含 `openalex`，但 `source.openalex.enabled=false`，则会被跳过。
- 若 `source.openalex.enabled=true`，但 `executor.source` 不含 `openalex`，同样不会执行（便于按场景组合来源）。
