import json

from omegaconf import OmegaConf

from zotero_tracker.retriever.biorxiv import BiorxivRetriever
from zotero_tracker.retriever.medrxiv import MedrxivRetriever
from zotero_tracker.retriever import biorxiv_like


class _FakeResp:
    def __init__(self, payload: str):
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _cfg(days=2, max_results=200):
    return OmegaConf.create(
        {
            "source": {
                "biorxiv": {"enabled": True, "days": days, "max_results": max_results},
                "medrxiv": {"enabled": True, "days": days, "max_results": max_results},
            },
            "executor": {"debug": True},
        }
    )


def test_fetch_biorxiv_like_paginates_and_respects_max_results(monkeypatch):
    calls: list[str] = []

    def fake_urlopen(req, timeout=30):
        calls.append(req.full_url)  # type: ignore[attr-defined]
        # 第 1 页 100 条，第 2 页 50 条
        if calls[-1].endswith("/0"):
            coll = [{"title": f"t{i}", "doi": f"10.1101/{i}"} for i in range(100)]
        else:
            coll = [{"title": f"t{i}", "doi": f"10.1101/{i}"} for i in range(100, 150)]
        return _FakeResp(json.dumps({"collection": coll}))

    monkeypatch.setattr(biorxiv_like, "urlopen", fake_urlopen)
    rows = biorxiv_like.fetch_biorxiv_like("biorxiv", days=7, max_results=120)

    assert len(rows) == 120
    assert "/biorxiv/" in calls[0]
    assert calls[0].endswith("/0")
    assert calls[1].endswith("/100")
    # 日期范围应是 .../biorxiv/YYYY-MM-DD/YYYY-MM-DD/offset（不是 .../7d/offset）
    assert "/7d/" not in calls[0]
    path_parts = calls[0].split("/details/")[1].split("/")
    assert len(path_parts) >= 4
    # path_parts: [server, from_date, to_date, offset]
    assert path_parts[0] == "biorxiv"
    assert path_parts[1].count("-") == 2
    assert path_parts[2].count("-") == 2


def test_fetch_biorxiv_like_retries_then_success(monkeypatch):
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("timed out")
        return _FakeResp(json.dumps({"collection": [{"title": "ok", "doi": "10.1101/ok"}]}))

    def fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(biorxiv_like, "urlopen", fake_urlopen)
    monkeypatch.setattr(biorxiv_like.time, "sleep", fake_sleep)
    rows = biorxiv_like.fetch_biorxiv_like(
        "biorxiv",
        days=2,
        max_results=10,
        timeout_seconds=5,
        num_retries=3,
        retry_backoff_seconds=1,
    )

    assert len(rows) == 1
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_biorxiv_convert_to_paper_basic_mapping():
    r = BiorxivRetriever(_cfg())
    raw = {
        "title": "A Paper",
        "authors": "Alice; Bob",
        "abstract": "Abs",
        "doi": "10.1101/123456",
        "biorxiv_pdf_url": "https://example.org/p.pdf",
    }
    p = r.convert_to_paper(raw)
    assert p is not None
    assert p.source == "biorxiv"
    assert p.title == "A Paper"
    assert p.authors == ["Alice", "Bob"]
    assert p.abstract == "Abs"
    assert p.url == "https://doi.org/10.1101/123456"
    assert p.pdf_url == "https://example.org/p.pdf"


def test_medrxiv_convert_to_paper_basic_mapping():
    r = MedrxivRetriever(_cfg())
    raw = {
        "title": "M Paper",
        "authors": "A;B",
        "abstract": "",
        "rel_doi": "10.1101/999999",
        "medrxiv_pdf_url": "",
    }
    p = r.convert_to_paper(raw)
    assert p is not None
    assert p.source == "medrxiv"
    assert p.title == "M Paper"
    assert p.authors == ["A", "B"]
    assert p.url == "https://doi.org/10.1101/999999"
    assert p.pdf_url is None

    assert r.convert_to_paper({"title": "", "doi": "x"}) is None
    assert r.convert_to_paper({"title": "No URL"}) is None

