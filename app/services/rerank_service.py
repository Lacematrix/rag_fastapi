from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import List

import requests

from app.core.config import Settings
from app.models import RetrievedChunk
from sentence_transformers import SentenceTransformer

class LocalQwenRerankService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        self.model = SentenceTransformer(settings.rerank_model_name)

    def _request_rerank(self, query: str, documents: List[str]) -> List[float]:
        scores = []
        query_embeddings = self.model.encode([query], prompt_name="query")
        document_embeddings = self.model.encode(documents)
        similarity = self.model.similarity(query_embeddings, document_embeddings)
        for score in similarity[0]:
            scores.append(float(score))
        return scores

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not chunks:
            return []

        documents = [chunk.text for chunk in chunks]

        try:
            scores = self._request_rerank(query, documents)
        except Exception as e:
            print(e)
            return sorted(chunks, key=lambda x: x.score, reverse=True)[:top_k]

        reranked = []
        for chunk, score in zip(chunks, scores):
            metadata = deepcopy(chunk.metadata)
            metadata["retrieval_score"] = chunk.retrieval_score if chunk.retrieval_score is not None else chunk.score
            metadata["rerank_score"] = float(score)
            reranked.append(
                replace(
                    chunk,
                    score=float(chunk.score),
                    metadata=metadata,
                    rerank_score=float(score),
                    retrieval_score=chunk.retrieval_score if chunk.retrieval_score is not None else chunk.score,
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
