"""通过 SMTP 发送纯文本（Markdown）邮件。"""

import datetime
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr

from loguru import logger
from omegaconf import DictConfig


def send_markdown_email(config: DictConfig, body: str) -> None:
    sender = config.email.sender
    receiver = config.email.receiver
    password = config.email.sender_password
    smtp_server = config.email.smtp_server
    smtp_port = int(config.email.smtp_port)

    def _format_addr(s: str) -> str:
        name, addr = parseaddr(s)
        return formataddr((str(Header(name, "utf-8")), addr))

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = _format_addr(f"Zotero 文献追踪 <{sender}>")
    msg["To"] = _format_addr(f"收件人 <{receiver}>")
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    msg["Subject"] = Header(f"Zotero 文献追踪 {today}", "utf-8")

    # 465：常见为直接 TLS（如 QQ）；587：明文握手后再 STARTTLS。避免 465 上误用 STARTTLS 的多余失败日志。
    if smtp_port == 465:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
    else:
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        except Exception as e:
            logger.debug(f"STARTTLS 失败：{e}，尝试 SSL。")
            try:
                server = smtplib.SMTP_SSL(smtp_server, smtp_port)
            except Exception as e2:
                logger.debug(f"SSL 失败：{e2}，尝试明文 SMTP。")
                server = smtplib.SMTP(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()
