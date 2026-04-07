import numpy as np
from openai import OpenAI
from omegaconf import DictConfig

from .base import BaseReranker, register_reranker


@register_reranker("api")
class ApiReranker(BaseReranker):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        api = config.reranker.api
        self._client = OpenAI(api_key=api.key, base_url=api.base_url)
        self._model = api.model
        self._batch_size = int(api.get("batch_size") or 64)

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        all_texts = s1 + s2
        all_embeddings: list[list[float]] = []
        for i in range(0, len(all_texts), self._batch_size):
            batch = all_texts[i : i + self._batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            all_embeddings.extend([r.embedding for r in response.data])
        s1_embeddings = np.array(all_embeddings[: len(s1)])
        s2_embeddings = np.array(all_embeddings[len(s1) :])
        s1_n = s1_embeddings / np.linalg.norm(s1_embeddings, axis=1, keepdims=True)
        s2_n = s2_embeddings / np.linalg.norm(s2_embeddings, axis=1, keepdims=True)
        return np.dot(s1_n, s2_n.T)
