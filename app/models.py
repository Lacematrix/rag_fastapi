from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentKind(str, Enum):
    KB = "KB"
    SYSTEM = "SYSTEM"
    MCP = "MCP"


class IntentLevel(str, Enum):
    DOMAIN = "DOMAIN"
    CATEGORY = "CATEGORY"
    TOPIC = "TOPIC"


@dataclass
class IntentNode:
    id: str
    name: str
    description: str = ""
    level: IntentLevel = IntentLevel.TOPIC
    parent_id: Optional[str] = None
    kind: IntentKind = IntentKind.KB
    collection_name: Optional[str] = None
    top_k: Optional[int] = None
    prompt_template: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    full_path: str = ""

    def is_kb(self) -> bool:
        return self.kind == IntentKind.KB

    def is_system(self) -> bool:
        return self.kind == IntentKind.SYSTEM


@dataclass
class NodeScore:
    node: IntentNode
    score: float


@dataclass
class SubQuestionIntent:
    sub_question: str
    node_scores: List[NodeScore]


@dataclass
class RewriteResult:
    rewritten_question: str
    sub_questions: List[str]


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_score: float | None = None
    rerank_score: float | None = None


@dataclass
class RetrievalContext:
    kb_context: str
    intent_chunks: Dict[str, List[RetrievedChunk]]

    def is_empty(self) -> bool:
        return not any(self.intent_chunks.values())
