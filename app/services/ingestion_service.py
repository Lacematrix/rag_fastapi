from __future__ import annotations

from pathlib import Path

from app.core.service_logging import service_logger
from app.core.config import Settings
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import MilvusVectorStoreService


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        document_processor: DocumentProcessor,
        vector_store: MilvusVectorStoreService,
    ) -> None:
        self.settings = settings
        self.document_processor = document_processor
        self.vector_store = vector_store

    @service_logger()
    def ingest_file(
        self,
        file_path: Path,
        collection_name: str,
        chunk_mode: str | None = None,
        bm25_language: str | None = None,
    ) -> int:
        chunks = self.document_processor.chunk_file(file_path, mode=chunk_mode)
        docs = self.document_processor.chunking_service.to_documents(chunks, source=str(file_path.name))
        ids = [chunk.chunk_id for chunk in chunks]
        self.vector_store.add_documents(
            collection_name=collection_name,
            docs=docs,
            ids=ids,
            bm25_language=bm25_language,
        )
        return len(chunks)
