"""生成用于邮件发送的正文（Markdown + HTML）。"""

from datetime import datetime
from html import escape

from .feedback import LABEL_IRRELEVANT, LABEL_RELEVANT, paper_item_id
from .keywords import KeywordResult
from .protocol import Paper


def _markdown_why_lines(p: Paper) -> list[str]:
    lines: list[str] = [
        "#### 为什么推荐给你",
        "",
    ]
    if p.matched_keywords:
        kw_line = ", ".join(f"`{t}`" for t in p.matched_keywords)
        lines.append(f"- **命中关键词：** {kw_line}")
    else:
        lines.append(
            "- **命中关键词：** _未命中展示用关键词（可能与书库语言或分词方式有关）。_"
        )
    lines.extend(
        [
            "- **书库依据：** 以下按对「相关度」总分的**贡献**从高到低排列"
            "（相关度 = 10 × Σ 余弦相似度 × 时间权重）。",
            "",
        ]
    )
    if p.corpus_explanations:
        for k, ex in enumerate(p.corpus_explanations, start=1):
            path_s = f"；路径：`{ex.collection_path}`" if ex.collection_path else ""
            lines.append(
                f"  {k}. {ex.title}{path_s} — 余弦 {ex.cosine_sim:.3f}，"
                f"时间权重 {ex.time_weight:.4f}，贡献 {ex.contribution:.3f}"
            )
    else:
        lines.append("  _（未启用分解或暂无书库条目。）_")
    lines.extend(
        [
            "",
            "_说明：分解基于向量相似度与时间衰减。若启用邮件反馈加权，最终相关度可能已单独调整。_",
            "",
        ]
    )
    return lines


def _html_why_block(p: Paper) -> str:
    if p.matched_keywords:
        kw_html = ", ".join(escape(t) for t in p.matched_keywords)
    else:
        kw_html = (
            "<span style='color:#6b7280;'>未命中展示用关键词"
            "（可能与书库语言或分词方式有关）。</span>"
        )
    rows: list[str] = []
    for ex in p.corpus_explanations:
        path_cell = escape(ex.collection_path or "—")
        rows.append(
            "<tr>"
            f"<td style='padding:6px 8px;border:1px solid #e5e7eb;'>{escape(ex.title)}</td>"
            f"<td style='padding:6px 8px;border:1px solid #e5e7eb;font-size:12px;'>{path_cell}</td>"
            f"<td style='padding:6px 8px;border:1px solid #e5e7eb;'>{ex.cosine_sim:.3f}</td>"
            f"<td style='padding:6px 8px;border:1px solid #e5e7eb;'>{ex.time_weight:.4f}</td>"
            f"<td style='padding:6px 8px;border:1px solid #e5e7eb;'>{ex.contribution:.3f}</td>"
            "</tr>"
        )
    if rows:
        table = (
            "<table style='border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:6px 8px;border:1px solid #e5e7eb;'>书库标题</th>"
            "<th style='text-align:left;padding:6px 8px;border:1px solid #e5e7eb;'>集合路径</th>"
            "<th style='padding:6px 8px;border:1px solid #e5e7eb;'>余弦相似度</th>"
            "<th style='padding:6px 8px;border:1px solid #e5e7eb;'>时间权重</th>"
            "<th style='padding:6px 8px;border:1px solid #e5e7eb;'>贡献</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    else:
        table = "<p style='color:#6b7280;font-size:13px;'>未启用分解或暂无书库条目。</p>"
    note = (
        "<p style='font-size:12px;color:#6b7280;margin:8px 0 0 0;'>"
        "分解基于向量相似度与时间衰减；若启用邮件反馈加权，最终相关度可能已单独调整。"
        "</p>"
    )
    return (
        "<h5 style='margin:12px 0 6px 0;'>为什么推荐给你</h5>"
        f"<p style='margin:4px 0;'><b>命中关键词：</b> {kw_html}</p>"
        "<p style='margin:4px 0;font-size:13px;'>"
        "书库依据（按对「相关度」总分的贡献从高到低；相关度 = 10 × Σ 余弦相似度 × 时间权重）："
        "</p>"
        f"{table}{note}"
    )


def render_markdown(
    papers: list[Paper],
    keywords: KeywordResult,
    *,
    date: datetime | None = None,
    feedback_links: dict[str, dict[str, str]] | None = None,
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
        lines.extend(_markdown_why_lines(p))
        pid = paper_item_id(p)
        item_feedback = (feedback_links or {}).get(pid, {})
        rel_link = item_feedback.get(LABEL_RELEVANT)
        irrel_link = item_feedback.get(LABEL_IRRELEVANT)
        if rel_link or irrel_link:
            parts = []
            if rel_link:
                parts.append(f"[相关]({rel_link})")
            if irrel_link:
                parts.append(f"[不相关]({irrel_link})")
            lines.append(f"- **反馈：** {' / '.join(parts)}")
        lines.append("")
    return "\n".join(lines)


def render_html(
    papers: list[Paper],
    keywords: KeywordResult,
    *,
    date: datetime | None = None,
    feedback_links: dict[str, dict[str, str]] | None = None,
) -> str:
    date = date or datetime.now()
    kws = ", ".join(escape(t) for t in keywords.terms) if keywords.terms else "未能提取关键词。"
    parts: list[str] = [
        "<html><body style='font-family:Arial,sans-serif;line-height:1.55;color:#111;'>",
        f"<h2 style='margin-bottom:8px;'>Zotero 文献追踪 — {date.strftime('%Y-%m-%d')}</h2>",
        "<h3 style='margin-bottom:6px;'>兴趣关键词（来自你的书库）</h3>",
        f"<p style='margin-top:0;'>{kws}</p>",
        "<h3>论文列表</h3>",
    ]
    if not papers:
        parts.append("<p>今日暂无匹配论文。</p></body></html>")
        return "".join(parts)

    for i, p in enumerate(papers, start=1):
        score = f"{p.score:.3f}" if p.score is not None else "无"
        authors = ", ".join(p.authors[:8])
        if len(p.authors) > 8:
            authors += "，…"
        tldr = escape((p.tldr or "").strip().replace("\n", " "))
        title = escape(p.title)
        pid = paper_item_id(p)
        item_feedback = (feedback_links or {}).get(pid, {})
        rel_link = item_feedback.get(LABEL_RELEVANT)
        irrel_link = item_feedback.get(LABEL_IRRELEVANT)
        parts.append("<div style='border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin:10px 0;'>")
        parts.append(f"<h4 style='margin:0 0 8px 0;'>{i}. {title}</h4>")
        parts.append(f"<p style='margin:4px 0;'><b>来源：</b> {escape(p.source)}</p>")
        parts.append(f"<p style='margin:4px 0;'><b>相关度：</b> {score}</p>")
        parts.append(f"<p style='margin:4px 0;'><b>作者：</b> {escape(authors)}</p>")
        parts.append(
            f"<p style='margin:4px 0;'><b>链接：</b> <a href='{escape(p.url)}' target='_blank'>查看论文</a></p>"
        )
        if p.pdf_url:
            parts.append(
                "<p style='margin:4px 0;'><b>PDF：</b> "
                f"<a href='{escape(p.pdf_url)}' target='_blank'>下载 PDF</a></p>"
            )
        parts.append(f"<p style='margin:6px 0 10px 0;'><b>一句话摘要：</b> {tldr}</p>")
        parts.append(_html_why_block(p))
        if rel_link or irrel_link:
            parts.append("<div>")
            if rel_link:
                parts.append(
                    "<a href='{}' target='_blank' style='display:inline-block;"
                    "padding:6px 12px;margin-right:8px;background:#16a34a;color:#fff;"
                    "text-decoration:none;border-radius:6px;'>相关</a>".format(escape(rel_link))
                )
            if irrel_link:
                parts.append(
                    "<a href='{}' target='_blank' style='display:inline-block;"
                    "padding:6px 12px;background:#dc2626;color:#fff;"
                    "text-decoration:none;border-radius:6px;'>不相关</a>".format(escape(irrel_link))
                )
            parts.append("</div>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "".join(parts)
