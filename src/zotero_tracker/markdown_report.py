"""生成用于邮件发送的 Markdown 正文。"""

from datetime import datetime

from .keywords import KeywordResult
from .protocol import Paper


def render_markdown(
    papers: list[Paper],
    keywords: KeywordResult,
    *,
    date: datetime | None = None,
) -> str:
    date = date or datetime.now()
    lines: list[str] = [
        f"# Zotero 文献追踪 — {date.strftime('%Y-%m-%d')}",
        "",
        "## 兴趣关键词（来自你的书库）",
        "",
    ]
    if keywords.terms:
        lines.append(", ".join(f"`{t}`" for t in keywords.terms))
    else:
        lines.append("_未能提取关键词（书库为空或文本过短）。_")
    lines.extend(["", "## 论文列表", ""])

    if not papers:
        lines.append("_今日暂无匹配论文。_")
        return "\n".join(lines)

    for i, p in enumerate(papers, start=1):
        score = f"{p.score:.3f}" if p.score is not None else "无"
        authors = ", ".join(p.authors[:8])
        if len(p.authors) > 8:
            authors += "，…"
        tldr = (p.tldr or "").strip().replace("\n", " ")
        lines.append(f"### {i}. {p.title}")
        lines.append("")
        lines.append(f"- **来源：** {p.source}")
        lines.append(f"- **相关度：** {score}")
        lines.append(f"- **作者：** {authors}")
        lines.append(f"- **链接：** {p.url}")
        if p.pdf_url:
            lines.append(f"- **PDF：** {p.pdf_url}")
        lines.append(f"- **一句话摘要：** {tldr}")
        lines.append("")
    return "\n".join(lines)
