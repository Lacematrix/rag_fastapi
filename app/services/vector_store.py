from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from app.core.config import Settings
from app.core.service_logging import service_logger
from app.models import RetrievedChunk
from app.services.collection_config import CollectionConfigService, CollectionSpec
from app.services.embedding_service import EmbeddingService


class MilvusVectorStoreService:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        collection_config: CollectionConfigService,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.collection_config = collection_config
        self._collections: Dict[str, Collection] = {}
        self._bm25_models: Dict[str, Any] = {}
        self._bm25_corpus: Dict[str, List[str]] = {}
        self._bm25_language: Dict[str, str] = {}
        self._connect()

    def _connect(self) -> None:
        try:
            connection_kwargs = {"uri": self.settings.milvus_uri}
            if self.settings.milvus_token:
                connection_kwargs["token"] = self.settings.milvus_token
            
            # Connect to Milvus
            if self.settings.drop_old_collections:
                db_path = Path(self.settings.milvus_uri)
                if db_path.exists():
                    db_path.unlink()
            connections.connect(alias="default", **connection_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Milvus: {e}") from e

    def _get_collection(self, collection_name: str) -> Collection:
        if collection_name in self._collections:
            return self._collections[collection_name]
        try:
            collection = Collection(collection_name)
            collection.load()
            self._collections[collection_name] = collection
            return collection
        except Exception:
            return self._create_collection(collection_name)

    def _create_collection(self, collection_name: str) -> Collection:
        try:
            spec = self.collection_config.get_spec(collection_name)
            fields = self._build_fields(spec)
            schema = CollectionSchema(fields, consistency_level=self.settings.milvus_consistency_level)
            collection = Collection(collection_name, schema=schema)
            # create index on embedding
            collection.create_index(
                field_name="embedding",
                index_params=self._resolve_index_params(spec),
            )
            # Only create sparse_embedding index if explicitly configured in spec
            if spec and spec.enable_sparse_embedding and any(field.name == "sparse_embedding" for field in fields):
                try:
                    collection.create_index(
                        field_name="sparse_embedding",
                        index_params=self._resolve_sparse_index_params(),
                    )
                except Exception as e:
                    service_logger.warning(f"Failed to create sparse_embedding index for {collection_name}: {e}")
            collection.load()
            self._collections[collection_name] = collection
            return collection
        except Exception as e:
            raise RuntimeError(f"Failed to create collection {collection_name}: {e}") from e

    def _build_fields(self, spec: CollectionSpec | None) -> List[FieldSchema]:
        if not spec or not spec.fields:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.settings.default_embedding_dim,
                ),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            ]
            # Only add sparse_embedding if configured
            if self.settings.enable_sparse_embedding:
                fields.append(FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR))
            return fields

        fields: List[FieldSchema] = []
        for field_spec in spec.fields:
            dtype = self._resolve_dtype(field_spec.dtype)
            kwargs = {
                "name": field_spec.name,
                "dtype": dtype,
                "is_primary": field_spec.is_primary,
                "auto_id": field_spec.auto_id,
            }
            if dtype == DataType.VARCHAR and field_spec.max_length:
                kwargs["max_length"] = int(field_spec.max_length)
            if dtype == DataType.FLOAT_VECTOR:
                dim = field_spec.dim or spec.embedding_dim or self.settings.default_embedding_dim
                kwargs["dim"] = int(dim)
            fields.append(FieldSchema(**kwargs))
        # Only add sparse_embedding if explicitly configured in spec or settings
        if (not any(field.name == "sparse_embedding" for field in fields) and
            (self.settings.enable_sparse_embedding or (spec and spec.enable_sparse_embedding))):
            fields.append(FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR))
        return fields

    def _resolve_index_params(self, spec: CollectionSpec | None) -> Dict[str, object]:
        if spec and spec.index_params:
            return spec.index_params
        return {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}

    def _resolve_sparse_index_params(self) -> Dict[str, object]:
        return {"metric_type": "BM25", "index_type": "SPARSE_INVERTED_INDEX", "params": {}}

    def _ensure_bm25_model(self, collection_name: str, texts: List[str], language: str) -> Any:
        if collection_name in self._bm25_models and self._bm25_language.get(collection_name) == language:
            return self._bm25_models[collection_name]

        try:
            tokenizers_module = import_module("pymilvus.model.sparse.bm25.tokenizers")
            sparse_module = import_module("pymilvus.model.sparse")
        except Exception as e:
            raise RuntimeError(f"Failed to import BM25EmbeddingFunction: {e}") from e

        build_default_analyzer = tokenizers_module.build_default_analyzer
        BM25EmbeddingFunction = sparse_module.BM25EmbeddingFunction
        analyzer = build_default_analyzer(language=language)
        bm25_ef = BM25EmbeddingFunction(analyzer)
        if texts:
            bm25_ef.fit(texts)
        self._bm25_models[collection_name] = bm25_ef
        self._bm25_language[collection_name] = language
        return bm25_ef

    @staticmethod
    def _get_sparse_row(sparse_embeddings: Any, index: int) -> Any:
        if hasattr(sparse_embeddings, "getrow"):
            return sparse_embeddings.getrow(index)
        if hasattr(sparse_embeddings, "__getitem__"):
            return sparse_embeddings[index : index + 1]
        return sparse_embeddings[index]

    @staticmethod
    def _resolve_dtype(raw: str) -> DataType:
        try:
            return getattr(DataType, raw)
        except AttributeError:
            return DataType.VARCHAR

    @service_logger()
    def add_documents(
        self,
        collection_name: str,
        docs: List[Document],
        ids: List[str],
        bm25_language: str | None = None,
    ) -> None:
        if len(docs) != len(ids):
            raise ValueError("docs and ids must have the same length")
        if not docs:
            return
        
        collection = self._get_collection(collection_name)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata or {} for doc in docs]
        
        embeddings = self.embedding_service.embed_documents(texts)
        has_sparse = any(field.name == "sparse_embedding" for field in collection.schema.fields)
        sparse_embeddings = None
        if has_sparse and self.settings.enable_sparse_embedding:
            try:
                language = (bm25_language or self.settings.bm25_language).strip()
                corpus = self._bm25_corpus.setdefault(collection_name, [])
                corpus.extend(texts)
                bm25_ef = self._ensure_bm25_model(collection_name, corpus, language)
                if corpus:
                    bm25_ef.fit(corpus)
                sparse_embeddings = bm25_ef.encode_documents(texts) if texts else None
            except Exception as e:
                service_logger.warning(f"Failed to generate sparse embeddings for {collection_name}: {e}")
                sparse_embeddings = None
        
        data = []
        for i in range(len(ids)):
            row = {
                "id": ids[i],
                "embedding": embeddings[i],
                "text": texts[i],
                "metadata": json.dumps(metadatas[i]),
            }
            if sparse_embeddings is not None:
                try:
                    row["sparse_embedding"] = self._get_sparse_row(sparse_embeddings, i)
                except Exception as e:
                    service_logger.warning(f"Failed to get sparse embedding row at index {i}: {e}")
            data.append(row)
        
        try:
            collection.insert(data)
            collection.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to {collection_name}: {e}") from e

    @service_logger()
    def retrieve(self, collection_name: str, query: str, top_k: int) -> List[RetrievedChunk]:
        collection = self._get_collection(collection_name)
        query_embedding = self.embedding_service.embed_query(query)
        
        try:
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["id", "text", "metadata"],
            )
            chunks: List[RetrievedChunk] = []
            for hits in results:
                for hit in hits:
                    try:
                        metadata = json.loads(hit.entity.get("metadata", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                    distance = hit.distance
                    score = 1 / (1 + distance)
                    chunks.append(
                        RetrievedChunk(
                            id=hit.entity.get("id", ""),
                            text=hit.entity.get("text", ""),
                            score=float(score),
                            metadata=metadata,
                        )
                    )
            return chunks
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve from {collection_name}: {e}") from e

    @service_logger()
    def bm25_search(self, collection_name: str, query: str, top_k: int) -> List[RetrievedChunk]:
        collection = self._get_collection(collection_name)
        has_sparse = any(field.name == "sparse_embedding" for field in collection.schema.fields)
        if not has_sparse:
            return []
        corpus = self._bm25_corpus.get(collection_name, [])
        if not corpus:
            return []
        language = self._bm25_language.get(collection_name, self.settings.bm25_language)
        bm25_ef = self._ensure_bm25_model(collection_name, corpus, language)
        try:
            results = collection.search(
                data=[query],
                anns_field="sparse_embedding",
                param={
                    "metric_type": "BM25",
                    "params": {
                        "k1": float(self.settings.bm25_k1),
                        "b": float(self.settings.bm25_b),
                    },
                },
                limit=top_k,
                output_fields=["id", "text", "metadata"],
            )
            chunks: List[RetrievedChunk] = []
            for hits in results:
                for hit in hits:
                    try:
                        metadata = json.loads(hit.entity.get("metadata", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                    chunks.append(
                        RetrievedChunk(
                            id=hit.entity.get("id", ""),
                            text=hit.entity.get("text", ""),
                            score=float(hit.score),
                            metadata=metadata,
                        )
                    )
            return chunks
        except Exception as e:
            raise RuntimeError(f"Failed to BM25 search from {collection_name}: {e}") from e

    def delete_chunks_by_ids(self, collection_name: str, chunk_ids: List[str]) -> Any:
        if not chunk_ids:
            return None
        collection = self._get_collection(collection_name)
        try:
            filter_expr = " or ".join([f'id == "{chunk_id}"' for chunk_id in chunk_ids])
            if filter_expr:
                result = collection.delete(filter_expr)
                collection.flush()
                return result
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to delete chunks from {collection_name}: {e}") from e

    def delete_chunk_by_id(self, collection_name: str, chunk_id: str) -> Any:
        return self.delete_chunks_by_ids(collection_name, [chunk_id])

    def delete_document_vectors(self, collection_name: str, doc_id: str) -> Any:
        collection = self._get_collection(collection_name)
        try:
            filter_expr = f'metadata LIKE "%\\"doc_id\\": \\"{doc_id}\\"%" OR metadata LIKE "%\\"document_id\\": \\"{doc_id}\\"%"'
            result = collection.delete(filter_expr)
            collection.flush()
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to delete document vectors from {collection_name}: {e}") from e
