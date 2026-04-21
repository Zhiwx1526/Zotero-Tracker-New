from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import dotenv
import streamlit as st
from loguru import logger
from omegaconf import DictConfig, OmegaConf

try:
    # 包内导入：适用于 `uv run zotero-tracker-web`
    from .executor import Executor, RunResult
except ImportError:
    # 脚本导入：适用于 `uv run streamlit run src/zotero_tracker/web_app.py`
    from zotero_tracker.executor import Executor, RunResult

import hydra  # noqa: F401 — 注册 OmegaConf 的 oc.* 解析器，供 ${oc.env:...} 使用

os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
BASE_CONFIG_PATH = CONFIG_DIR / "base.yaml"
CUSTOM_CONFIG_PATH = CONFIG_DIR / "custom.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
SUPPORTED_SOURCES = ("arxiv", "openalex", "biorxiv", "medrxiv")
SOURCE_LABELS = {
    "arxiv": "arXiv 预印本平台",
    "openalex": "OpenAlex 学术索引平台",
    "biorxiv": "bioRxiv 预印本平台",
    "medrxiv": "medRxiv 预印本平台",
}


def _load_merged_config() -> DictConfig:
    base_cfg = OmegaConf.load(BASE_CONFIG_PATH)
    custom_cfg = OmegaConf.load(CUSTOM_CONFIG_PATH)
    merged = OmegaConf.merge(base_cfg, custom_cfg)
    OmegaConf.resolve(merged)
    return merged


def _format_env_value(val: str) -> str:
    if any(c in val for c in " \t\n\r#\"'\\"):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return val


def _upsert_dotenv(path: Path, updates: dict[str, str]) -> None:
    """写入或更新 .env 中的键，保留原有行顺序与其它键。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    out_lines: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates:
                    out_lines.append(f"{key}={_format_env_value(updates[key])}")
                    seen.add(key)
                    continue
            out_lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            out_lines.append(f"{key}={_format_env_value(value)}")
    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def _build_env_updates_from_form(form_data: dict[str, Any]) -> dict[str, str]:
    """从表单生成待写入 .env 的键值。密码类字段若留空则不改写已有环境变量。"""
    updates: dict[str, str] = {}

    zid = form_data["zotero_user_id"].strip()
    if zid:
        updates["ZOTERO_ID"] = zid

    zk = form_data["zotero_api_key"].strip()
    if zk:
        updates["ZOTERO_KEY"] = zk

    recv = form_data["email_receiver"].strip()
    if recv:
        updates["RECEIVER_EMAIL"] = recv
    # 兼容旧 .env 中的 RECEIVER
    if recv:
        updates["RECEIVER"] = recv

    sender = form_data["email_sender"].strip()
    if sender:
        updates["SENDER"] = sender

    smtp_server = form_data["email_smtp_server"].strip()
    if smtp_server:
        updates["SMTP_SERVER"] = smtp_server

    updates["SMTP_PORT"] = str(int(form_data["email_smtp_port"]))

    spw = form_data["email_sender_password"].strip()
    if spw:
        updates["SENDER_PASSWORD"] = spw

    lk = form_data["llm_key"].strip()
    if lk:
        updates["OPENAI_API_KEY"] = lk

    lb = form_data["llm_base_url"].strip()
    if lb:
        updates["OPENAI_API_BASE"] = lb

    lm = form_data["llm_model"].strip()
    if lm:
        updates["LLM_MODEL"] = lm

    return updates


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text == "???" else text


def _smtp_port_for_form(cfg: DictConfig) -> int:
    raw = cfg.email.get("smtp_port") if cfg.email else None
    if raw is None:
        return 465
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 465


def _get_source(cfg: DictConfig, source: str) -> DictConfig:
    source_cfg = cfg.source.get(source)
    if source_cfg is None:
        source_cfg = OmegaConf.create({})
        cfg.source[source] = source_cfg
    return source_cfg


def _render_form(cfg: DictConfig) -> dict[str, Any]:
    st.subheader("参数配置")
    st.caption(
        "敏感项（Zotero / 邮箱 / LLM）保存到项目根目录 `.env`；"
        "`config/custom.yaml` 仅写入非敏感结构与 `${oc.env:...}` 占位，请勿将 `.env` 提交到仓库。"
    )

    with st.container(border=True):
        st.markdown("### Zotero")
        zotero_user_id = st.text_input(
            "zotero.user_id",
            value=_clean_text(cfg.zotero.get("user_id")),
            help="Zotero 用户 ID，用于访问你的个人文献库。",
        )
        zotero_api_key = st.text_input(
            "zotero.api_key",
            value=_clean_text(cfg.zotero.get("api_key")),
            type="password",
            help="Zotero API Key，用于鉴权读取文献元数据。",
        )

    with st.container(border=True):
        st.markdown("### Email")
        email_sender = st.text_input(
            "email.sender",
            value=_clean_text(cfg.email.get("sender")),
            help="发件邮箱地址（与 SMTP 登录账号通常一致）。保存后写入 `.env` 的 `SENDER`。",
        )
        email_receiver = st.text_input(
            "email.receiver",
            value=_clean_text(cfg.email.get("receiver")),
            help="推送结果的收件邮箱地址。",
        )
        email_smtp_server = st.text_input(
            "email.smtp_server",
            value=_clean_text(cfg.email.get("smtp_server")),
            help="SMTP 服务器主机名，例如 smtp.qq.com。保存后写入 `.env` 的 `SMTP_SERVER`。",
        )
        email_smtp_port = st.number_input(
            "email.smtp_port",
            min_value=1,
            max_value=65535,
            value=_smtp_port_for_form(cfg),
            step=1,
            help="SMTP 端口；465 多为 SSL，587 多为 STARTTLS。保存后写入 `.env` 的 `SMTP_PORT`。",
        )
        email_sender_password = st.text_input(
            "email.sender_password",
            value=_clean_text(cfg.email.get("sender_password")),
            type="password",
            help="SMTP 密码或授权码。",
        )

    with st.container(border=True):
        st.markdown("### LLM")
        llm_key = st.text_input(
            "llm.api.key",
            value=_clean_text(cfg.llm.api.get("key")),
            type="password",
            help="LLM 服务 API Key，用于生成一句话摘要。",
        )
        llm_base_url = st.text_input(
            "llm.api.base_url",
            value=_clean_text(cfg.llm.api.get("base_url")),
            help="OpenAI 兼容 API 网关地址，例如 DashScope 或自建网关。",
        )
        llm_model = st.text_input(
            "llm.generation_kwargs.model",
            value=_clean_text(cfg.llm.generation_kwargs.get("model")),
            help="用于 TLDR、推荐解读与简报导语的模型名（均为 chat 补全）。",
        )
        ne_m = cfg.llm.get("natural_explain") or {}
        natural_explain_enabled = st.toggle(
            "llm.natural_explain.enabled（自然语言推荐解读）",
            value=bool(ne_m.get("enabled", False)),
            help="对邮件中的论文额外调用 LLM，生成 2～4 句「推荐解读」（会增加 API 费用与耗时）。",
            key="llm_natural_explain_enabled",
        )
        ne_max_raw = ne_m.get("max_papers")
        natural_explain_max_papers = st.number_input(
            "llm.natural_explain.max_papers（0=不限制篇数）",
            min_value=0,
            max_value=500,
            value=0 if ne_max_raw is None else int(ne_max_raw),
            step=1,
            help="仅对前 N 篇生成推荐解读；填 0 表示对邮件中的全部候选篇生成。",
            key="llm_natural_explain_max",
        )
        br_m = cfg.llm.get("briefing") or {}
        briefing_enabled = st.toggle(
            "llm.briefing.enabled（今日简报导语）",
            value=bool(br_m.get("enabled", False)),
            help="每封邮件额外 1 次 LLM 调用，在正文开头生成「今日简报」段落。",
            key="llm_briefing_enabled",
        )
        briefing_max_papers = st.number_input(
            "llm.briefing.max_papers",
            min_value=1,
            max_value=50,
            value=max(1, int(br_m.get("max_papers", 15))),
            step=1,
            help="写入简报 prompt 的最多篇数（标题+一句话摘要），避免上下文过长。",
            key="llm_briefing_max",
        )

    with st.container(border=True):
        st.markdown("### Sources")
        st.caption("控制每个平台是否启用，以及检索最近多少天的论文。")
        for source in SUPPORTED_SOURCES:
            source_cfg = _get_source(cfg, source)
            display_name = SOURCE_LABELS.get(source, source)
            col1, col2 = st.columns([2, 1])
            with col1:
                enabled = st.toggle(
                    f"{display_name}（enabled）",
                    value=bool(source_cfg.get("enabled", False)),
                    help=f"是否启用 {display_name}。配置键：source.{source}.enabled。",
                    key=f"enabled_{source}",
                )
            with col2:
                days = st.number_input(
                    f"{display_name}（days）",
                    min_value=1,
                    max_value=30,
                    value=max(1, int(source_cfg.get("days", 2))),
                    step=1,
                    help=f"{display_name} 仅检索最近 N 天的论文。配置键：source.{source}.days。",
                    key=f"days_{source}",
                )
            source_cfg.enabled = enabled
            source_cfg.days = int(days)

    with st.container(border=True):
        st.markdown("### Executor")
        explain_enabled = st.toggle(
            "executor.explain_enabled（推荐原因）",
            value=bool(cfg.executor.get("explain_enabled", True)),
            help="开启后邮件与网页会展示「为什么推荐」：命中关键词与书库相似度分解。",
            key="executor_explain_enabled",
        )
        min_score_raw = cfg.executor.get("min_score")
        min_score_text = "" if min_score_raw is None else str(min_score_raw)
        min_score_input = st.text_input(
            "executor.min_score",
            value=min_score_text,
            help="相似度过滤阈值；留空表示不启用阈值过滤。",
        )

    with st.container(border=True):
        st.markdown("### Feedback")
        fb_cfg = cfg.get("feedback")
        fb_enabled_default = bool(fb_cfg.get("enabled", False)) if fb_cfg is not None else False
        feedback_enabled = st.toggle(
            "feedback.enabled（是否开启推送反馈）",
            value=fb_enabled_default,
            help="开启后邮件中可带「相关 / 不相关」链接，并参与后续重排加权（需配置反馈服务 base_url 与 secret）。",
            key="feedback_enabled",
        )

    return {
        "zotero_user_id": zotero_user_id,
        "zotero_api_key": zotero_api_key,
        "email_sender": email_sender,
        "email_receiver": email_receiver,
        "email_smtp_server": email_smtp_server,
        "email_smtp_port": email_smtp_port,
        "email_sender_password": email_sender_password,
        "llm_key": llm_key,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "natural_explain_enabled": natural_explain_enabled,
        "natural_explain_max_papers": natural_explain_max_papers,
        "briefing_enabled": briefing_enabled,
        "briefing_max_papers": briefing_max_papers,
        "min_score_input": min_score_input,
        "explain_enabled": explain_enabled,
        "feedback_enabled": feedback_enabled,
        "source_values": {source: _get_source(cfg, source) for source in SUPPORTED_SOURCES},
    }


def _parse_min_score(value: str) -> tuple[float | None, str | None]:
    text = value.strip()
    if not text:
        return None, None
    try:
        return float(text), None
    except ValueError:
        return None, "executor.min_score 必须是数字或留空。"


def _apply_form_to_custom(custom_cfg: DictConfig, form_data: dict[str, Any]) -> str | None:
    min_score, min_score_err = _parse_min_score(form_data["min_score_input"])
    if min_score_err:
        return min_score_err

    # 敏感字段通过环境变量注入；YAML 中只保留 oc.env 引用，避免密钥落盘到 custom.yaml
    custom_cfg.zotero.user_id = "${oc.env:ZOTERO_ID,???}"
    custom_cfg.zotero.api_key = "${oc.env:ZOTERO_KEY,???}"
    custom_cfg.email.sender = "${oc.env:SENDER,???}"
    custom_cfg.email.receiver = "${oc.env:RECEIVER_EMAIL,???}"
    custom_cfg.email.smtp_server = "${oc.env:SMTP_SERVER,smtp.qq.com}"
    custom_cfg.email.smtp_port = "${oc.env:SMTP_PORT,465}"
    custom_cfg.email.sender_password = "${oc.env:SENDER_PASSWORD,???}"
    custom_cfg.llm.api.key = "${oc.env:OPENAI_API_KEY,???}"
    custom_cfg.llm.api.base_url = "${oc.env:OPENAI_API_BASE,???}"
    custom_cfg.llm.generation_kwargs.model = "${oc.env:LLM_MODEL,???}"
    if custom_cfg.llm.get("natural_explain") is None:
        custom_cfg.llm.natural_explain = OmegaConf.create({})
    custom_cfg.llm.natural_explain.enabled = bool(form_data["natural_explain_enabled"])
    ne_max = int(form_data["natural_explain_max_papers"])
    custom_cfg.llm.natural_explain.max_papers = None if ne_max <= 0 else ne_max
    if custom_cfg.llm.get("briefing") is None:
        custom_cfg.llm.briefing = OmegaConf.create({})
    custom_cfg.llm.briefing.enabled = bool(form_data["briefing_enabled"])
    custom_cfg.llm.briefing.max_papers = max(1, int(form_data["briefing_max_papers"]))
    custom_cfg.executor.min_score = min_score
    custom_cfg.executor.explain_enabled = bool(form_data["explain_enabled"])

    if custom_cfg.get("feedback") is None:
        custom_cfg.feedback = OmegaConf.create({})
    custom_cfg.feedback.enabled = bool(form_data["feedback_enabled"])

    enabled_sources: list[str] = []
    for source, source_cfg in form_data["source_values"].items():
        custom_cfg.source[source].enabled = bool(source_cfg.enabled)
        custom_cfg.source[source].days = int(source_cfg.days)
        if bool(source_cfg.enabled):
            enabled_sources.append(source)

    custom_cfg.executor.source = enabled_sources
    return None


def _save_custom_config(custom_cfg: DictConfig) -> None:
    CUSTOM_CONFIG_PATH.write_text(OmegaConf.to_yaml(custom_cfg, resolve=False), encoding="utf-8")


def _configure_runtime_logging(log_box: Any) -> None:
    logger.remove()

    def sink(message: Any) -> None:
        text = str(message).rstrip()
        st.session_state["run_logs"].append(text)
        log_box.code("\n".join(st.session_state["run_logs"][-300:]))

    logger.add(
        sink,
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )
    for name in logging.root.manager.loggerDict:
        if "zotero_tracker" in name:
            continue
        logging.getLogger(name).setLevel(logging.WARNING)


def _run_tracker() -> tuple[bool, str, RunResult | None]:
    try:
        runtime_cfg = _load_merged_config()
        result = Executor(runtime_cfg).run()
    except Exception as exc:
        return False, f"运行失败：{exc}", None
    if result is None:
        return True, "运行结束（未发送邮件或未产出结果）。", None
    return True, "运行完成。", result


def main() -> None:
    st.set_page_config(page_title="Zotero Tracker", page_icon="📚", layout="wide")
    st.title("Zotero 文献自动追踪")
    st.caption("在线配置并触发文献追踪推送。")

    if "run_logs" not in st.session_state:
        st.session_state["run_logs"] = []
    if "last_run_result" not in st.session_state:
        st.session_state["last_run_result"] = None

    merged_cfg = _load_merged_config()
    form_data = _render_form(merged_cfg)

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        save_clicked = st.button("保存配置", type="primary", use_container_width=True)
    with action_col2:
        run_clicked = st.button("开始追踪", use_container_width=True)

    if save_clicked or run_clicked:
        custom_cfg = OmegaConf.load(CUSTOM_CONFIG_PATH)
        err = _apply_form_to_custom(custom_cfg, form_data)
        if err:
            st.error(err)
            return
        env_updates = _build_env_updates_from_form(form_data)
        if env_updates:
            _upsert_dotenv(ENV_PATH, env_updates)
            dotenv.load_dotenv(ENV_PATH, override=True)
        _save_custom_config(custom_cfg)
        st.success("配置已保存：敏感项写入 `.env`，结构与占位写入 `config/custom.yaml`。")

    if run_clicked:
        st.session_state["run_logs"] = []
        log_box = st.empty()
        _configure_runtime_logging(log_box)
        with st.spinner("任务运行中，请稍候..."):
            ok, msg, run_result = _run_tracker()
        if ok:
            st.success(msg)
            st.session_state["last_run_result"] = run_result
        else:
            st.error(msg)

    with st.expander("运行日志", expanded=False):
        log_text = "\n".join(st.session_state["run_logs"][-500:])
        st.code(log_text if log_text else "暂无日志")

    last = st.session_state.get("last_run_result")
    if last is not None:
        with st.expander("最近一次推送的论文与「为什么推荐」", expanded=False):
            if last.keywords.terms:
                st.caption("书库兴趣关键词（展示用）")
                st.write(", ".join(last.keywords.terms))
            if not last.papers:
                st.info("本次列表为空（可能当日无候选或已被阈值过滤）。")
            for idx, paper in enumerate(last.papers, start=1):
                with st.container(border=True):
                    st.markdown(f"**{idx}.** [{paper.title}]({paper.url})")
                    sc = f"{paper.score:.3f}" if paper.score is not None else "—"
                    st.caption(f"来源 {paper.source} · 相关度 {sc}")
                    if paper.tldr:
                        st.write(paper.tldr)
                    if (paper.natural_explain or "").strip():
                        st.markdown("**推荐解读**")
                        st.write(paper.natural_explain)
                    st.markdown("**为什么推荐给你**")
                    if paper.matched_keywords:
                        st.write("命中关键词：" + ", ".join(paper.matched_keywords))
                    else:
                        st.caption("未命中展示用关键词（可能与书库语言或分词方式有关）。")
                    if paper.corpus_explanations:
                        rows = [
                            {
                                "书库标题": e.title,
                                "集合路径": e.collection_path or "—",
                                "余弦相似度": round(e.cosine_sim, 3),
                                "时间权重": round(e.time_weight, 4),
                                "贡献": round(e.contribution, 3),
                            }
                            for e in paper.corpus_explanations
                        ]
                        st.dataframe(rows, use_container_width=True, hide_index=True)
                    else:
                        st.caption("无书库分解（未启用或暂无条目）。")


def run_streamlit() -> None:
    from streamlit.web import cli as stcli

    target = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(target)]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
