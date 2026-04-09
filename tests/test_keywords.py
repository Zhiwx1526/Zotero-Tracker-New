from datetime import datetime

from omegaconf import OmegaConf

from zotero_tracker.keywords import extract_keywords_from_corpus, match_keywords_in_paper
from zotero_tracker.protocol import CorpusPaper


def test_keywords_non_empty():
    corpus = [
        CorpusPaper(
            item_key="k1",
            title="Deep learning for vision",
            abstract="We propose a neural network architecture for image classification tasks.",
            added_date=datetime(2024, 1, 1),
            paths=["ml"],
        ),
        CorpusPaper(
            item_key="k2",
            title="Natural language processing survey",
            abstract="Transformers and attention mechanisms for NLP benchmarks.",
            added_date=datetime(2024, 2, 1),
            paths=["nlp"],
        ),
    ]
    cfg = OmegaConf.create({"top_k": 10, "max_features": 500, "ngram_max": 2})
    result = extract_keywords_from_corpus(corpus, cfg)
    assert result.terms
    assert len(result.terms) <= 10
    assert len(result.scores) == len(result.terms)


def test_match_keywords_single_token():
    terms = ["learning", "transformer"]
    title = "Deep Learning Survey"
    abstract = "No match here."
    assert match_keywords_in_paper(terms, title, abstract) == ["learning"]


def test_match_keywords_phrase_ngram():
    terms = ["machine learning", "foo"]
    title = "Intro"
    abstract = "We study machine learning methods."
    assert match_keywords_in_paper(terms, title, abstract) == ["machine learning"]


def test_match_keywords_no_substring_for_single_token():
    terms = ["net"]
    title = "Neural networks"
    abstract = ""
    assert match_keywords_in_paper(terms, title, abstract) == []


def test_match_keywords_empty_terms():
    assert match_keywords_in_paper([], "a", "b") == []
