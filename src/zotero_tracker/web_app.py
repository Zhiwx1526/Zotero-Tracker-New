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
    from .executor import Executor
except ImportError:
    # 脚本导入：适用于 `uv run streamlit run src/zotero_tracker/web_app.py`
    from zotero_tracker.executor import Executor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv.load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
BASE_CONFIG_PATH = CONFIG_DIR / "base.yaml"
CUSTOM_CONFIG_PATH = CONFIG_DIR / "custom.yaml"
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
    return OmegaConf.merge(base_cfg, custom_cfg)


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text == "???" else text


def _get_source(cfg: DictConfig, source: str) -> DictConfig:
    source_cfg = cfg.source.get(source)
    if source_cfg is None:
        source_cfg = OmegaConf.create({})
        cfg.source[source] = source_cfg
    return source_cfg


def _render_form(cfg: DictConfig) -> dict[str, Any]:
    st.subheader("参数配置")
    st.caption("所有参数会写入 `config/custom.yaml`，用于后续默认运行。")

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
        email_receiver = st.text_input(
            "email.receiver",
            value=_clean_text(cfg.email.get("receiver")),
            help="推送结果的收件邮箱地址。",
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
            help="用于 TLDR 生成的模型名。",
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
        min_score_raw = cfg.executor.get("min_score")
        min_score_text = "" if min_score_raw is None else str(min_score_raw)
        min_score_input = st.text_input(
            "executor.min_score",
            value=min_score_text,
            help="相似度过滤阈值；留空表示不启用阈值过滤。",
        )

    return {
        "zotero_user_id": zotero_user_id,
        "zotero_api_key": zotero_api_key,
        "email_receiver": email_receiver,
        "llm_key": llm_key,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "min_score_input": min_score_input,
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

    custom_cfg.zotero.user_id = form_data["zotero_user_id"].strip() or "???"
    custom_cfg.zotero.api_key = form_data["zotero_api_key"].strip() or "???"
    custom_cfg.email.receiver = form_data["email_receiver"].strip() or "???"
    custom_cfg.llm.api.key = form_data["llm_key"].strip() or "???"
    custom_cfg.llm.api.base_url = form_data["llm_base_url"].strip() or "???"
    custom_cfg.llm.generation_kwargs.model = form_data["llm_model"].strip() or "???"
    custom_cfg.executor.min_score = min_score

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


def _run_tracker() -> tuple[bool, str]:
    try:
        runtime_cfg = _load_merged_config()
        Executor(runtime_cfg).run()
    except Exception as exc:
        return False, f"运行失败：{exc}"
    return True, "运行完成。"


def main() -> None:
    st.set_page_config(page_title="Zotero Tracker", page_icon="📚", layout="wide")
    st.title("Zotero 文献自动追踪")
    st.caption("在线配置并触发文献追踪推送。")

    if "run_logs" not in st.session_state:
        st.session_state["run_logs"] = []

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
        _save_custom_config(custom_cfg)
        st.success("配置已保存到 config/custom.yaml")

    if run_clicked:
        st.session_state["run_logs"] = []
        log_box = st.empty()
        _configure_runtime_logging(log_box)
        with st.spinner("任务运行中，请稍候..."):
            ok, msg = _run_tracker()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    with st.expander("运行日志", expanded=False):
        log_text = "\n".join(st.session_state["run_logs"][-500:])
        st.code(log_text if log_text else "暂无日志")


def run_streamlit() -> None:
    from streamlit.web import cli as stcli

    target = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(target)]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
