from datetime import datetime

from omegaconf import OmegaConf

from zotero_tracker.keywords import extract_keywords_from_corpus
from zotero_tracker.protocol import CorpusPaper


def test_keywords_non_empty():
    corpus = [
        CorpusPaper(
            title="Deep learning for vision",
            abstract="We propose a neural network architecture for image classification tasks.",
            added_date=datetime(2024, 1, 1),
            paths=["ml"],
        ),
        CorpusPaper(
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
