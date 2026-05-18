from __future__ import annotations

from loguru import logger


class TqdmToLogger:
    """将 tqdm 输出重定向到 loguru，便于 Streamlit 日志区展示。"""

    def write(self, buf: str) -> None:
        text = str(buf).strip()
        if not text:
            return
        # 仅保留进度条样式行，避免把其他噪音写入日志
        if "|" in text and "%" in text:
            logger.info(text)

    def flush(self) -> None:
        return


def get_tqdm_stream() -> TqdmToLogger:
    return TqdmToLogger()
