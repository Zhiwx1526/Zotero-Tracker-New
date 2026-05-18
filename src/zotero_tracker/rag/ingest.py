from __future__ import annotations

from pathlib import Path

from loguru import logger

from .chunking import TextChunk, chunk_text


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        logger.warning("未安装 pypdf，跳过 PDF 文件：{}", str(path))
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:
        logger.warning("读取 PDF 失败 {}: {}", str(path), exc)
        return ""


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        try:
            return path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8-sig").strip()
    if suffix == ".pdf":
        return _read_pdf_text(path)
    return ""


def collect_knowledge_chunks(
    paths: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    supported = {".md", ".txt", ".pdf"}
    for entry in paths:
        p = Path(entry)
        if p.is_file():
            candidates = [p]
        elif p.is_dir():
            candidates = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in supported]
        else:
            logger.warning("RAG 知识路径不存在：{}", entry)
            continue
        for file in candidates:
            content = _read_document(file)
            if not content:
                continue
            chunks.extend(
                chunk_text(
                    source_path=str(file),
                    title=file.stem,
                    text=content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
    logger.info("RAG 文档切块完成：{} 个片段。", len(chunks))
    return chunks
