from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.agent_service import AgentService
from app.services.chunking import ChunkingService
from app.services.collection_config import CollectionConfigService
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.guidance_service import GuidanceService
from app.services.ingestion_service import IngestionService
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService
from app.services.rerank_service import LocalQwenRerankService
from app.services.retrieval_service import MultiChannelRetrievalService
from app.services.rewrite_service import QueryRewriteService
from app.services.vector_store import MilvusVectorStoreService


class ServiceContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_config = CollectionConfigService(settings)
        self.llm_service = LLMService(settings)
        self.embedding_service = EmbeddingService(
            api_key=settings.embedding_api_key,
            openai_api_base=settings.embedding_base_url,
            model=settings.embedding_model_name,
            chunk_size=settings.chunk_size,
        )
        self.vector_store_service = MilvusVectorStoreService(
            settings,
            self.embedding_service,
            self.collection_config,
        )
        self.chunking_service = ChunkingService(settings)
        self.document_processor = DocumentProcessor(self.chunking_service)
        self.intent_service = IntentService(settings, self.llm_service, self.collection_config)
        self.rewrite_service = QueryRewriteService(settings, self.llm_service)
        self.guidance_service = GuidanceService(settings, self.intent_service)
        self.rerank_service = LocalQwenRerankService(settings)
        self.retrieval_service = MultiChannelRetrievalService(
            settings=settings,
            vector_store=self.vector_store_service,
            intent_service=self.intent_service,
            llm_service=self.llm_service,
            rerank_service=self.rerank_service,
        )
        self.agent_service = AgentService(settings, self.llm_service)
        self.ingestion_service = IngestionService(
            settings=settings,
            document_processor=self.document_processor,
            vector_store=self.vector_store_service,
        )


@lru_cache
def get_container() -> ServiceContainer:
    settings = get_settings()
    return ServiceContainer(settings)
