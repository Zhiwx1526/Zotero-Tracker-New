# Zotero-Tracker

根据你的 Zotero 书库，对新增论文做兴趣对齐排序（embedding 相似度），并从书库中抽取 TF-IDF 兴趣关键词用于展示；最终以 **Markdown 纯文本** 邮件推送。当前支持 `arxiv` / `openalex` / `biorxiv` / `medrxiv` 等来源。

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

## Streamlit 在线入口

安装依赖后可直接启动 Web 界面：

```bash
uv sync
uv run streamlit run src/zotero_tracker/web_app.py
```

或使用脚本入口：

```bash
uv run zotero-tracker-web
```

页面支持配置并保存以下参数（写入 `config/custom.yaml`）：

- `zotero.user_id`：Zotero 用户 ID。
- `zotero.api_key`：Zotero API Key。
- `email.receiver`：接收日报的邮箱。
- `llm.api.key` / `llm.api.base_url` / `llm.generation_kwargs.model`：LLM 接口与模型。
- `source.<platform>.enabled` / `source.<platform>.days`：各来源开关与日期窗口。
- `executor.min_score`：相似度过滤阈值（留空表示不启用）。

注意：敏感字段（API Key）会写入本地 `config/custom.yaml`，请勿提交到公开仓库。

## 云端定时推送（GitHub Actions）

仓库已提供工作流：`.github/workflows/daily-push.yml`，支持两种触发方式：

- `schedule`：每日定时执行（当前为 `0 1 * * *`，即 UTC 01:00）。
- `workflow_dispatch`：在 GitHub 页面手动点击执行。

### 1) 配置仓库 Secrets

在 GitHub 仓库中进入 `Settings -> Secrets and variables -> Actions`，添加以下 `Repository secrets`：

- `ZOTERO_ID`
- `ZOTERO_KEY`
- `RECEIVER_EMAIL`
- `SENDER`
- `SENDER_PASSWORD`
- `SMTP_SERVER`
- `SMTP_PORT`
- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `LLM_MODEL`
- `LLM_LANGUAGE`

说明：

- `LLM_LANGUAGE` 建议填 `zh`（需要英文可填 `English`）。
- GitHub Actions 的 cron 使用 **UTC** 时区；例如 `0 1 * * *` 对应中国时间 `09:00`。
- 若你已在 `config/custom.yaml` 固定了部分配置，也建议在 Secrets 中统一维护敏感信息，避免明文进入仓库历史。

### 2) 手动触发与查看日志

1. 打开仓库 `Actions` 页面，选择 `Daily Literature Push`。
2. 点击 `Run workflow` 手动触发一次。
3. 进入该次运行查看 `Run zotero tracker` 步骤日志，确认抓取、排序与邮件发送流程正常。

## 邮件反馈闭环（最小签名）

支持在邮件每条论文后附加 `相关/不相关` 点击反馈，用于下一轮规则重排。

1) 在环境变量中配置反馈服务：

```bash
set FEEDBACK_BASE_URL=http://127.0.0.1:8787
set FEEDBACK_SECRET=replace-with-a-random-secret
```

2) 开启配置（`config/custom.yaml`）：

```yaml
feedback:
  enabled: true
```

3) 启动反馈服务：

```bash
uv run zotero-tracker-feedback
```

反馈端点为 `/feedback`，并提供 `/health` 健康检查。服务会做 HMAC 签名校验、过期校验（`ttl_seconds`）和幂等写入（同 `user_id+push_id+item_id` 覆盖）。

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
    days: 2
    max_results: 2000
  biorxiv:
    enabled: true
    days: 2
    max_results: 200
  medrxiv:
    enabled: false
    days: 2
    max_results: 200
  openalex:
    enabled: true
    query: null
    search_title_only: false
    days: 3
    from_publication_date: null
    to_publication_date: null
    per_page: 50
    max_results: 200
    mailto: null

executor:
  source: ["arxiv", "biorxiv", "medrxiv", "openalex"]
```

说明：
- 若 `executor.source` 包含 `openalex`，但 `source.openalex.enabled=false`，则会被跳过。
- 若 `source.openalex.enabled=true`，但 `executor.source` 不含 `openalex`，同样不会执行（便于按场景组合来源）。
- 若某个来源拉取失败，`executor.source_error_policy=continue`（默认）会跳过该来源并继续；改为 `fail_fast` 则立即中断。

### 日期窗口（OpenAlex、arXiv 与 bioRxiv/medRxiv）

- **OpenAlex**：使用 `from_publication_date` + `to_publication_date`（OpenAlex 仅支持按日）。**两项均为 null** 时按 **`days`（默认 3）** 自动设为 **UTC「当天往前第 N 个自然日」～当天**。**两项都写明**时则完全按配置日期过滤。
- **arXiv**：使用 `days`（最近 N 天，默认 2）控制 `submittedDate` 查询窗口（UTC 自然日，含今天）。查询为 `(cat:… OR …) AND submittedDate:[…]`，条数上限见 `source.arxiv.max_results`（`debug` 时仍会截断为 10）。
- **bioRxiv / medRxiv**：使用 `days`（最近 N 天，默认 2），语义与示例 `.../details/biorxiv/7d/0` 一致；条数上限见 `max_results`。
