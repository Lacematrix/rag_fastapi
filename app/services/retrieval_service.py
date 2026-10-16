from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
import re
from typing import Dict, List, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.service_logging import service_logger
from app.core.config import Settings
from app.models import NodeScore, RetrievedChunk, RetrievalContext, SubQuestionIntent
from app.prompts import PromptRegistry
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService
from app.services.rerank_service import LocalQwenRerankService
from app.services.vector_store import MilvusVectorStoreService


class SearchChannelType(str, Enum):
    INTENT_DIRECTED = "INTENT_DIRECTED"
    VECTOR_GLOBAL = "VECTOR_GLOBAL"


@dataclass
class SearchContext:
    original_question: str
    rewritten_question: str
    intents: List[SubQuestionIntent]
    top_k: int
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def main_question(self) -> str:
        return self.rewritten_question or self.original_question


@dataclass
class SearchChannelResult:
    channel_type: SearchChannelType
    channel_name: str
    chunks: List[RetrievedChunk]


class SearchChannel(Protocol):
    def get_name(self) -> str: ...

    def get_priority(self) -> int: ...

    def is_enabled(self, context: SearchContext) -> bool: ...

    def search(self, context: SearchContext) -> SearchChannelResult: ...


class IntentDirectedSearchChannel:
    def __init__(
        self,
        settings: Settings,
        vector_store: MilvusVectorStoreService,
        intent_service: IntentService,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.intent_service = intent_service

    def get_name(self) -> str:
        return "IntentDirectedSearch"

    def get_priority(self) -> int:
        return 1

    def is_enabled(self, context: SearchContext) -> bool:
        return bool(self._extract_kb_intents(context))

    def search(self, context: SearchContext) -> SearchChannelResult:
        chunks: List[RetrievedChunk] = []
        kb_intents = self._extract_kb_intents(context)

        for node_score in kb_intents:
            collection = node_score.node.collection_name
            if not collection:
                continue
            base_top_k = node_score.node.top_k or context.top_k
            search_top_k = base_top_k * self.settings.intent_directed_topk_multiplier
            results = self.vector_store.retrieve(collection, context.main_question, search_top_k)
            for result in results:
                result.metadata["intent_id"] = node_score.node.id
                result.metadata["channel"] = self.get_name()
            chunks.extend(results)

        return SearchChannelResult(
            channel_type=SearchChannelType.INTENT_DIRECTED,
            channel_name=self.get_name(),
            chunks=chunks,
        )

    def _extract_kb_intents(self, context: SearchContext) -> List[NodeScore]:
        all_scores = [score for sub in context.intents for score in sub.node_scores]
        return self.intent_service.kb_scores(all_scores, min_score=self.settings.intent_directed_min_intent_score)


class VectorGlobalSearchChannel:
    def __init__(
        self,
        settings: Settings,
        vector_store: MilvusVectorStoreService,
        intent_service: IntentService,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.intent_service = intent_service

    def get_name(self) -> str:
        return "VectorGlobalSearch"

    def get_priority(self) -> int:
        return 10

    def is_enabled(self, context: SearchContext) -> bool:
        all_scores = [score for sub in context.intents for score in sub.node_scores]
        if not all_scores:
            return True

        max_score = max(score.score for score in all_scores)
        if max_score < self.settings.vector_global_confidence_threshold:
            return True

        if len(all_scores) == 1 and max_score < self.settings.vector_global_single_intent_supplement_threshold:
            return True

        return False

    def search(self, context: SearchContext) -> SearchChannelResult:
        collection_set = set(self.settings.global_collections)
        collection_set.update(self.intent_service.list_kb_collections())

        search_top_k = context.top_k * self.settings.vector_global_topk_multiplier
        chunks: List[RetrievedChunk] = []

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.vector_store.retrieve, collection, context.main_question, search_top_k) for collection in collection_set]
            for future in futures:
                try:
                    results = future.result()
                except Exception:
                    continue
                for result in results:
                    result.metadata["channel"] = self.get_name()
                chunks.extend(results)

        return SearchChannelResult(
            channel_type=SearchChannelType.VECTOR_GLOBAL,
            channel_name=self.get_name(),
            chunks=chunks,
        )


class MultiChannelRetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: MilvusVectorStoreService,
        intent_service: IntentService,
        llm_service: LLMService,
        rerank_service: LocalQwenRerankService,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.intent_service = intent_service
        self.llm_service = llm_service
        self.rerank_service = rerank_service

        self.channels: List[SearchChannel] = [
            IntentDirectedSearchChannel(settings, vector_store, intent_service),
            VectorGlobalSearchChannel(settings, vector_store, intent_service),
        ]

    @service_logger()
    def retrieve(self, sub_intents: List[SubQuestionIntent], top_k: int, main_question: str, original_question: str) -> RetrievalContext:
        if not sub_intents:
            return RetrievalContext(kb_context="", intent_chunks={})

        question = main_question
        context = SearchContext(
            original_question=question,
            rewritten_question=question,
            intents=sub_intents,
            top_k=top_k,
        )

        enabled = sorted([ch for ch in self.channels if ch.is_enabled(context)], key=lambda x: x.get_priority())
        if not enabled:
            return RetrievalContext(kb_context="", intent_chunks={})

        with ThreadPoolExecutor(max_workers=len(enabled)) as pool:
            futures = [pool.submit(channel.search, context) for channel in enabled]
            results = [f.result() for f in futures]

        deduped = self._deduplicate(results)
        bm25_chunks = self._bm25_retrieve(original_question, sub_intents, top_k)
        fused = self._rrf_fuse(deduped, bm25_chunks, top_k)
        
        reranked = self.rerank_service.rerank(context.main_question, fused, top_k)

        all_kb_scores = [score for sub in sub_intents for score in self.intent_service.kb_scores(sub.node_scores)]
        intent_chunks: Dict[str, List[RetrievedChunk]] = {}
        if all_kb_scores:
            for score in all_kb_scores:
                intent_chunks[score.node.id] = reranked
        else:
            intent_chunks["multi_channel"] = reranked

        kb_context = self._build_kb_context(sub_intents, reranked)
        return RetrievalContext(kb_context=kb_context, intent_chunks=intent_chunks)

    def _deduplicate(self, results: List[SearchChannelResult]) -> List[RetrievedChunk]:
        priority_map = {
            SearchChannelType.INTENT_DIRECTED: 1,
            SearchChannelType.VECTOR_GLOBAL: 3,
        }

        sorted_results = sorted(results, key=lambda x: priority_map.get(x.channel_type, 99))
        merged: Dict[str, RetrievedChunk] = {}

        for result in sorted_results:
            for chunk in result.chunks:
                key = chunk.id or str(hash(chunk.text))
                if key not in merged or chunk.score > merged[key].score:
                    merged[key] = chunk

        return list(merged.values())

    def _build_kb_context(self, sub_intents: List[SubQuestionIntent], chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return ""

        lines = []
        for sub in sub_intents:
            lines.append("---")
            lines.append(f"子问题: {sub.sub_question}")
            lines.append("相关文档:")
            for idx, chunk in enumerate(chunks, start=1):
                text = chunk.text.strip().replace("\n", " ")
                lines.append(f"[{idx}] {text}")
        return "\n".join(lines)

    def _bm25_retrieve(self, question: str, sub_intents: List[SubQuestionIntent], top_k: int) -> List[RetrievedChunk]:
        collections = self._resolve_bm25_collections(sub_intents)
        if not collections:
            return []

        keywords = self._extract_keywords(question)
        query = " ".join(keywords) if keywords else question
        search_top_k = top_k * self.settings.bm25_topk_multiplier
        chunks: List[RetrievedChunk] = []

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.vector_store.bm25_search, collection, query, search_top_k) for collection in collections]
            for future in futures:
                try:
                    results = future.result()
                except Exception:
                    continue
                for result in results:
                    result.metadata["channel"] = "BM25Search"
                chunks.extend(results)

        return chunks

    def _resolve_bm25_collections(self, sub_intents: List[SubQuestionIntent]) -> List[str]:
        collection_set = set(self.settings.global_collections)
        collection_set.update(self.intent_service.list_kb_collections())
        for sub in sub_intents:
            for score in sub.node_scores:
                collection = score.node.collection_name
                if collection:
                    collection_set.add(collection)
        return sorted(collection_set)

    def _extract_keywords(self, query: str) -> List[str]:
        if not query:
            return []

        llm_keywords = self._extract_keywords_with_llm(query)
        if llm_keywords:
            return llm_keywords

        tokens = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query)
        seen = set()
        keywords: List[str] = []
        for token in tokens:
            if token.isascii() and len(token) < 2:
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords

    def _extract_keywords_with_llm(self, query: str) -> List[str]:
        try:
            system_prompt = PromptRegistry.get_template_text("bm25-keyword-extract")
            raw = self.llm_service.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=query)],
                temperature=0.0,
                top_p=0.1,
            )
            parsed = self.llm_service.try_parse_json(raw)
        except Exception:
            return []

        if isinstance(parsed, dict):
            candidates = parsed.get("keywords", [])
        elif isinstance(parsed, list):
            candidates = parsed
        else:
            candidates = []

        keywords: List[str] = []
        seen = set()
        for item in candidates:
            keyword = str(item).strip()
            if not keyword:
                continue
            if keyword.isascii() and len(keyword) < 2:
                continue
            if keyword in seen:
                continue
            seen.add(keyword)
            keywords.append(keyword)

        return keywords[:8]

    def _rrf_fuse(
        self,
        semantic_chunks: List[RetrievedChunk],
        bm25_chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not bm25_chunks:
            return sorted(semantic_chunks, key=lambda x: x.score, reverse=True)[:top_k]

        rrf_k = max(int(self.settings.rrf_k), 1)
        merged: Dict[str, Dict[str, object]] = {}

        def add_ranked(chunks: List[RetrievedChunk]) -> None:
            ranked = sorted(chunks, key=lambda x: x.score, reverse=True)
            for rank, chunk in enumerate(ranked, start=1):
                key = chunk.id or str(hash(chunk.text))
                score = 1.0 / (rrf_k + rank)
                if key in merged:
                    merged[key]["score"] = float(merged[key]["score"]) + score
                else:
                    merged[key] = {"chunk": chunk, "score": score}

        add_ranked(semantic_chunks)
        add_ranked(bm25_chunks)

        fused = [replace(entry["chunk"], score=float(entry["score"])) for entry in merged.values()]
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]
