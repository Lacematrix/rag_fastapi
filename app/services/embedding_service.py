from __future__ import annotations

import asyncio
from typing import List, Optional

import ollama

from app.core.service_logging import service_logger

class EmbeddingService:
    def __init__(
        self,
        api_key: str,
        openai_api_base: str,
        model: str,
        chunk_size: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = openai_api_base.rstrip("/")
        host = self.base_url.replace("/v1", "") if "/v1" in self.base_url else self.base_url
        self.model = model
        self.chunk_size = chunk_size
        self.client = ollama.Client(host=host)
        self.async_client = ollama.AsyncClient(host=host)

    @service_logger()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        embeddings: List[List[float]] = []
        
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i : i + self.chunk_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings

    @service_logger()
    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            res = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(res.get("embedding", []))
        return embeddings

    @service_logger("app.services.embedding.EmbeddingService._embed_batch_async")
    async def _embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            res = await self.async_client.embeddings(model=self.model, prompt=text)
            embeddings.append(res.get("embedding", []))
        return embeddings

    @service_logger()
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        embeddings: List[List[float]] = []
        
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i : i + self.chunk_size]
            batch_embeddings = await self._embed_batch_async(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings

    @service_logger()
    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await self.aembed_documents([text])
        return embeddings[0] if embeddings else []

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
