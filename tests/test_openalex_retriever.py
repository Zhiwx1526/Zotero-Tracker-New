from omegaconf import OmegaConf

from zotero_tracker.retriever.openalex import OpenAlexRetriever


def _build_config(*, query=None, debug=False, search_title_only=False):
    return OmegaConf.create(
        {
            "source": {
                "arxiv": {"category": ["cs.AI", "cs.LG"]},
                "openalex": {
                    "enabled": True,
                    "query": query,
                    "search_title_only": search_title_only,
                    "from_publication_date": "2026-04-01",
                    "to_publication_date": "2026-04-07",
                    "per_page": 2,
                    "max_results": 5,
                    "mailto": "bot@example.com",
                },
            },
            "executor": {"debug": debug},
        }
    )


def test_openalex_retrieve_uses_query_filter_and_pagination():
    cfg = _build_config(query="multimodal llm")
    retriever = OpenAlexRetriever(cfg)

    calls: list[dict] = []
    responses = [
        {"results": [{"title": "p1"}, {"title": "p2"}]},
        {"results": [{"title": "p3"}]},
    ]

    def fake_http_get(params):
        calls.append(dict(params))
        return responses[len(calls) - 1]

    retriever._http_get_json = fake_http_get  # type: ignore[method-assign]
    rows = retriever._retrieve_raw_papers()

    assert len(rows) == 3
    assert calls[0]["search"] == "multimodal llm"
    assert calls[0]["filter"] == "from_publication_date:2026-04-01,to_publication_date:2026-04-07"
    assert calls[0]["mailto"] == "bot@example.com"
    assert calls[0]["page"] == 1
    assert calls[1]["page"] == 2


def test_openalex_retrieve_falls_back_to_arxiv_category_and_debug_caps_results():
    cfg = _build_config(query=None, debug=True, search_title_only=True)
    retriever = OpenAlexRetriever(cfg)

    calls: list[dict] = []
    # debug 下上限为 min(max_results, 20)，本测试 max_results=5
    responses = [{"results": [{"title": f"p{i}-1"}, {"title": f"p{i}-2"}]} for i in range(3)]

    def fake_http_get(params):
        calls.append(dict(params))
        return responses[len(calls) - 1]

    retriever._http_get_json = fake_http_get  # type: ignore[method-assign]
    rows = retriever._retrieve_raw_papers()

    assert len(rows) == 5
    assert "search" not in calls[0]
    assert "title.search:cs.AI cs.LG" in calls[0]["filter"]
    assert len(calls) == 3


def test_openalex_convert_to_paper_handles_missing_fields():
    cfg = _build_config()
    retriever = OpenAlexRetriever(cfg)

    raw = {
        "id": "https://openalex.org/W123",
        "title": "A Test Paper",
        "authorships": [{"author": {"display_name": "Alice"}}, {"author": {"display_name": "Bob"}}],
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "primary_location": {"pdf_url": None, "landing_page_url": "https://example.org/paper"},
        "open_access": {"oa_url": "https://example.org/paper.pdf"},
    }
    paper = retriever.convert_to_paper(raw)

    assert paper is not None
    assert paper.source == "openalex"
    assert paper.title == "A Test Paper"
    assert paper.authors == ["Alice", "Bob"]
    assert paper.abstract == "Hello world"
    assert paper.url == "https://openalex.org/W123"
    assert paper.pdf_url == "https://example.org/paper.pdf"

    assert retriever.convert_to_paper({"title": "", "id": "x"}) is None
    assert retriever.convert_to_paper({"title": "No URL"}) is None
