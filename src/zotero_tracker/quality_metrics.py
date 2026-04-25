from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from loguru import logger


def normalize_journal_name(name: str | None) -> str:
    if not name:
        return ""
    return " ".join(str(name).strip().lower().split())


def _to_float(v: Any) -> float | None:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_quartile(v: Any) -> str | None:
    if v in (None, ""):
        return None
    q = str(v).strip().upper()
    return q if q in {"Q1", "Q2", "Q3", "Q4"} else None


def load_scimago_map(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("SJR 映射文件不存在：{}", path)
        return {}

    result: dict[str, dict[str, Any]] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            journal_name = normalize_journal_name(
                row.get("journal_name") or row.get("title") or row.get("source_title")
            )
            if not journal_name:
                continue
            result[journal_name] = {
                "sjr": _to_float(row.get("sjr")),
                "quartile": _to_quartile(row.get("quartile")),
            }
    logger.info("已加载 SJR 映射 {} 条：{}", len(result), path)
    return result


def quartile_to_score(quartile: str | None) -> float | None:
    if not quartile:
        return None
    q = quartile.upper()
    mapping = {"Q1": 1.0, "Q2": 0.75, "Q3": 0.5, "Q4": 0.25}
    return mapping.get(q)


def sjr_to_score(sjr: float | None, *, cap: float = 10.0) -> float | None:
    if sjr is None:
        return None
    cap = max(0.1, float(cap))
    return min(1.0, math.log1p(max(0.0, float(sjr))) / math.log1p(cap))


def resolve_journal_quality(
    journal_name: str | None,
    scimago_map: dict[str, dict[str, Any]],
    *,
    sjr_cap: float = 10.0,
) -> tuple[float | None, str | None, float | None]:
    key = normalize_journal_name(journal_name)
    if not key:
        return None, None, None
    row = scimago_map.get(key)
    if not row:
        return None, None, None
    sjr = _to_float(row.get("sjr"))
    quartile = _to_quartile(row.get("quartile"))

    sjr_s = sjr_to_score(sjr, cap=sjr_cap)
    q_s = quartile_to_score(quartile)
    if sjr_s is not None and q_s is not None:
        journal_norm = 0.7 * sjr_s + 0.3 * q_s
    else:
        journal_norm = sjr_s if sjr_s is not None else q_s
    return sjr, quartile, journal_norm


def source_authority_score(source: str, mapping: dict[str, Any]) -> float | None:
    if not source:
        return None
    raw = mapping.get(str(source).strip().lower())
    return _to_float(raw)
