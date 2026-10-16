from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "RAG FastAPI Simplified"
    api_prefix: str = "/api/v1"

    milvus_uri: str = "./milvus.db"
    milvus_token: str = ""
    milvus_consistency_level: str = "Session"
    milvus_db_name: str = "my_database_1"
    enable_sparse_embedding: bool = True

    default_collection: str = "rag_default_store"
    global_collections: List[str] = Field(default_factory=list)
    collection_config_path: Path = Path("app/data/collections.sample.json")
    default_embedding_dim: int = 1024

    chat_base_url: str = "http://127.0.0.1:11434/v1"
    chat_api_key: str = "EMPTY"
    chat_model: str = "qwen3:8b"

    embedding_api_key: str = "ollama"
    embedding_base_url: str = "http://127.0.0.1:11434/v1/"
    embedding_model_name: str = "qwen3-embedding:0.6b"
    rerank_base_url: str = "http://127.0.0.1:11434/v1/"
    rerank_api_key: str = "ollama"
    drop_old_collections: bool = True
    chunk_size: int = 512
    rerank_model_name: str = "Qwen/Qwen3-Embedding-0.6B"

    intent_min_score: float = 0.35
    max_intent_count: int = 3
    intent_route_min_confidence: float = 0.55

    vector_global_confidence_threshold: float = 0.6
    vector_global_single_intent_supplement_threshold: float = 0.8
    vector_global_topk_multiplier: int = 3

    bm25_topk_multiplier: int = 2
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60
    bm25_language: str = "en"

    intent_directed_min_intent_score: float = 0.4
    intent_directed_topk_multiplier: int = 2

    guidance_enabled: bool = True
    guidance_ambiguity_score_ratio: float = 0.8
    guidance_max_options: int = 6

    query_rewrite_enabled: bool = True

    chunk_mode: str = "recursive"
    chunk_size: int = 512
    overlap_size: int = 128
    chunk_separators: List[str] = Field(
        default_factory=lambda: [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ".",
            "!",
            "?",
            "；",
            ";",
            "，",
            ",",
            " ",
            "",
        ]
    )
    chunk_is_separator_regex: bool = False
    chunk_tokenizer_name: str = "cl100k_base"

    top_k: int = 8

    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    intent_tree_path: Path = Path("app/data/intent_tree.sample.json")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
