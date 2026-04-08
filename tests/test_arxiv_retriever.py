import re
from datetime import datetime

from zotero_tracker.retriever.arxiv import _build_search_query


def _extract_submitted_range(query: str) -> tuple[datetime, datetime]:
    match = re.search(r"submittedDate:\[(\d{12}) TO (\d{12})\]", query)
    assert match is not None
    start = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    end = datetime.strptime(match.group(2), "%Y%m%d%H%M")
    return start, end


def test_build_search_query_includes_categories():
    query = _build_search_query(["cs.AI", "cs.LG"], days=2)
    assert "(cat:cs.AI OR cat:cs.LG)" in query
    assert "submittedDate:[" in query


def test_build_search_query_days_controls_date_window():
    q1 = _build_search_query(["cs.AI"], days=1)
    q3 = _build_search_query(["cs.AI"], days=3)
    start1, end1 = _extract_submitted_range(q1)
    start3, end3 = _extract_submitted_range(q3)

    assert end1 == end3
    assert (end1.date() - start1.date()).days == 0
    assert (end3.date() - start3.date()).days == 2
