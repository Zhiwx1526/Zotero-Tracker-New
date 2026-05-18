"""生成用于邮件发送的正文（Markdown + HTML）。"""

from datetime import datetime
from html import escape

from .feedback import LABEL_IRRELEVANT, LABEL_RELEVANT, paper_item_id
from .keywords import KeywordResult
from .protocol import Paper


def _markdown_why_lines(p: Paper) -> list[str]:
    lines: list[str] = []
    lines.extend(
        [
            "以下按对「相关度」总分的**贡献**从高到低排列"
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
    if p.score_breakdown:
        lines.append("#### 质量权重分解")
        lines.append("")
        q_score = p.score_breakdown.get("quality_score")
        parts = [f"最终分数：{p.score_breakdown.get('final_score', p.score or 0.0):.3f}"]
        if q_score is not None:
            parts.append(f"质量分（不含相关性）：{q_score:.3f}")
        parts.append(f"相关性分：{p.score_breakdown.get('relevance', 0.0):.3f}")
        parts.append(f"来源权威度分：{p.score_breakdown.get('authority', 0.0):.3f}")
        lines.append("- " + " | ".join(parts))
        lines.append("")
    return lines


def _html_why_block(p: Paper) -> str:
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
    quality_html = ""
    if p.score_breakdown:
        score_row = (
            "<div style='display:flex;flex-wrap:wrap;gap:10px;font-size:13px;margin:6px 0 8px 0;'>"
            f"<span><b>最终分数：</b>{p.score_breakdown.get('final_score', p.score or 0.0):.3f}</span>"
            f"<span><b>质量分：</b>{float(p.score_breakdown.get('quality_score', 0.0)):.3f}</span>"
            f"<span><b>相关性分：</b>{float(p.score_breakdown.get('relevance', 0.0)):.3f}</span>"
            f"<span><b>来源权威度分：</b>{float(p.score_breakdown.get('authority', 0.0)):.3f}</span>"
            "</div>"
        )
        quality_html = (
            "<h5 style='margin:12px 0 6px 0;'>质量权重分解</h5>"
            f"{score_row}"
        )

    return (
        "<p style='margin:4px 0;font-size:13px;'>"
        "按对「相关度」总分的贡献从高到低（相关度 = 10 × Σ 余弦相似度 × 时间权重）："
        "</p>"
        f"{table}{quality_html}"
    )


def render_markdown(
    papers: list[Paper],
    keywords: KeywordResult,
    *,
    date: datetime | None = None,
    feedback_links: dict[str, dict[str, str]] | None = None,
    briefing_intro: str | None = None,
) -> str:
    date = date or datetime.now()
    lines: list[str] = [
        f"# Zotero 文献追踪 — {date.strftime('%Y-%m-%d')}",
        "",
    ]
    bi = (briefing_intro or "").strip()
    if bi:
        lines.extend(
            [
                "## 今日简报",
                "",
                bi,
                "",
            ]
        )
    lines.extend(
        [
            "## 兴趣关键词（来自你的书库）",
            "",
        ]
    )
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
        ne = (p.natural_explain or "").strip()
        if ne:
            lines.append(f"- **推荐解读：** {ne.replace(chr(10), ' ')}")
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
    briefing_intro: str | None = None,
) -> str:
    date = date or datetime.now()
    kws = ", ".join(escape(t) for t in keywords.terms) if keywords.terms else "未能提取关键词。"
    parts: list[str] = [
        "<html><body style='font-family:Arial,sans-serif;line-height:1.55;color:#111;'>",
        f"<h2 style='margin-bottom:8px;'>Zotero 文献追踪 — {date.strftime('%Y-%m-%d')}</h2>",
    ]
    bi = (briefing_intro or "").strip()
    if bi:
        parts.extend(
            [
                "<h3 style='margin-bottom:6px;'>今日简报</h3>",
                f"<p style='margin:0 0 14px 0;'>{escape(bi).replace(chr(10), '<br/>')}</p>",
            ]
        )
    parts.extend(
        [
            "<h3 style='margin-bottom:6px;'>兴趣关键词（来自你的书库）</h3>",
            f"<p style='margin-top:0;'>{kws}</p>",
            "<h3>论文列表</h3>",
        ]
    )
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
        ne = (p.natural_explain or "").strip()
        if ne:
            ne_html = escape(ne).replace("\n", "<br/>")
            parts.append(
                f"<p style='margin:6px 0 10px 0;'><b>推荐解读：</b> {ne_html}</p>"
            )
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
