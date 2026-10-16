from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ChatHistoryMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: List[ChatHistoryMessage] = Field(default_factory=list)
    top_k: Optional[int] = None


class IntentScoreView(BaseModel):
    sub_question: str
    intent_id: str
    intent_name: str
    score: float


class RetrievedChunkView(BaseModel):
    id: str
    text: str
    score: float
    intent_id: str
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    rewritten_question: str
    sub_questions: List[str]
    guidance_prompt: Optional[str] = None
    intents: List[IntentScoreView] = Field(default_factory=list)
    chunks: List[RetrievedChunkView] = Field(default_factory=list)
    intent_status: Optional[str] = None
    intent_confidence: Optional[float] = None
    need_rag: Optional[bool] = None
    chat_type: Optional[str] = None


class IngestResponse(BaseModel):
    collection_name: str
    chunk_count: int


class MilvusCollectionView(BaseModel):
    name: str
    num_entities: int = 0


class MilvusFieldView(BaseModel):
    name: str
    dtype: str
    is_primary: bool = False
    auto_id: bool = False
    is_vector: bool = False
    viewable: bool = True


class CollectionConfigView(BaseModel):
    name: str
    description: str = ""


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    service: str
    metadata: Dict[str, str] = Field(default_factory=dict)
