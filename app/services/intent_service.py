from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.models import IntentKind, IntentLevel, IntentNode, NodeScore, RewriteResult, SubQuestionIntent
from app.prompts import PromptRegistry
from app.services.collection_config import CollectionConfigService
from app.services.llm_service import LLMService


class IntentCandidate(BaseModel):
    id: str = Field(..., description="Intent id from the tree")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field("", description="Short reason")


class IntentAnalysis(BaseModel):
    chat_type: str = Field(..., description="chitchat | rag | clarify")
    need_rag: bool = Field(..., description="Whether to use retrieval")
    confidence: float = Field(..., ge=0.0, le=1.0)
    intents: List[IntentCandidate] = Field(default_factory=list)
    follow_up: str = Field("", description="Clarifying question when unclear")
    reason: str = Field("", description="Overall routing reason")



class IntentService:
    def __init__(
        self,
        settings: Settings,
        llm_service: LLMService,
        collection_config: CollectionConfigService,
    ) -> None:
        self.settings = settings
        self.llm_service = llm_service
        self.collection_config = collection_config
        self.id2node: Dict[str, IntentNode] = {}
        self.leaf_nodes: List[IntentNode] = []
        self.collection_nodes: List[IntentNode] = []
        self.children: Dict[str, List[str]] = defaultdict(list)
        self._load_tree(settings.intent_tree_path)
        self._load_collection_nodes()
        if self.collection_nodes:
            system_nodes = [node for node in self.leaf_nodes if node.is_system()]
            kb_nodes = [node for node in self.leaf_nodes if node.is_kb()]
            merged_ids = {node.id for node in system_nodes + kb_nodes}
            merged = system_nodes + kb_nodes
            for node in self.collection_nodes:
                if node.id in merged_ids:
                    continue
                merged.append(node)
            self.leaf_nodes = merged

    def _load_tree(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes: List[IntentNode] = []

        for item in data:
            node = IntentNode(
                id=item["id"],
                name=item.get("name", item["id"]),
                description=item.get("description", ""),
                level=IntentLevel(item.get("level", "TOPIC")),
                parent_id=item.get("parent_id"),
                kind=IntentKind(item.get("kind", "KB")),
                collection_name=item.get("collection_name"),
                top_k=item.get("top_k"),
                prompt_template=item.get("prompt_template"),
                examples=item.get("examples", []),
                full_path=item.get("full_path", ""),
            )
            nodes.append(node)
            self.id2node[node.id] = node

        for node in nodes:
            if node.parent_id:
                self.children[node.parent_id].append(node.id)

        for node in nodes:
            if not self.children.get(node.id):
                self.leaf_nodes.append(node)

        for node in nodes:
            if not node.full_path:
                node.full_path = self._build_full_path(node)

    def _build_full_path(self, node: IntentNode) -> str:
        names = [node.name]
        current = node
        while current.parent_id and current.parent_id in self.id2node:
            current = self.id2node[current.parent_id]
            names.append(current.name)
        names.reverse()
        return " > ".join(names)

    def get_node_by_id(self, node_id: str) -> IntentNode | None:
        return self.id2node.get(node_id)

    def resolve(self, rewrite_result: RewriteResult) -> List[SubQuestionIntent]:
        sub_questions = rewrite_result.sub_questions or [rewrite_result.rewritten_question]
        sub_intents: List[SubQuestionIntent] = []

        for sub_question in sub_questions:
            scores = self.classify_targets(sub_question)
            filtered = [x for x in scores if x.score >= self.settings.intent_min_score][: self.settings.max_intent_count]
            sub_intents.append(SubQuestionIntent(sub_question=sub_question, node_scores=filtered))

        return self._cap_total_intents(sub_intents)

    def analyze_intent(self, question: str) -> IntentAnalysis:
        prompt_items = self._build_prompt_items()
        parser = PydanticOutputParser(pydantic_object=IntentAnalysis)
        system_prompt = PromptRegistry.render(
            "intent-classifier",
            intent_list="\n\n".join(prompt_items),
            format_instructions=parser.get_format_instructions(),
        )

        raw = self.llm_service.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=question)],
            temperature=0.1,
            top_p=0.3,
        )

        try:
            return parser.parse(raw)
        except Exception:
            parsed = self.llm_service.try_parse_json(raw)
            if isinstance(parsed, dict):
                try:
                    return IntentAnalysis.model_validate(parsed)
                except Exception:
                    pass

        fallback_scores = self._lexical_fallback(question)
        fallback_candidates = [
            IntentCandidate(id=score.node.id, score=score.score, reason="lexical fallback")
            for score in fallback_scores
        ]
        confidence = max([score.score for score in fallback_scores], default=0.2)
        chat_type = "rag" if fallback_candidates else "clarify"
        need_rag = bool(fallback_candidates)
        follow_up = "你是想闲聊，还是希望我从资料里帮你查找并回答？"
        return IntentAnalysis(
            chat_type=chat_type,
            need_rag=need_rag,
            confidence=confidence,
            intents=fallback_candidates,
            follow_up=follow_up,
            reason="fallback",
        )

    def classify_targets(self, question: str) -> List[NodeScore]:
        analysis = self.analyze_intent(question)
        scores = self._scores_from_candidates(analysis.intents)
        if not scores and analysis.chat_type == "rag":
            scores = self._lexical_fallback(question)
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores

    def _scores_from_candidates(self, candidates: List[IntentCandidate]) -> List[NodeScore]:
        scores: List[NodeScore] = []
        for item in candidates:
            node = self.id2node.get(item.id)
            if not node:
                continue
            scores.append(NodeScore(node=node, score=float(item.score)))
        return scores

    def _lexical_fallback(self, question: str) -> List[NodeScore]:
        q = question.lower()
        out: List[NodeScore] = []

        for node in self.leaf_nodes:
            name_hit = 1.0 if node.name.lower() in q else 0.0
            desc_hit = 0.5 if node.description and any(tok in q for tok in node.description.lower().split()) else 0.0
            example_hit = 0.6 if any(ex.lower() in q for ex in node.examples) else 0.0
            score = min(1.0, 0.25 + name_hit * 0.5 + desc_hit * 0.2 + example_hit * 0.3)
            if score >= 0.4:
                out.append(NodeScore(node=node, score=score))

        return sorted(out, key=lambda x: x.score, reverse=True)

    def _build_prompt_items(self) -> List[str]:
        prompt_items = []
        for node in self.leaf_nodes:
            collection_info = self._format_collection_info(node)
            prompt_items.append(
                "\n".join(
                    [
                        f"- id={node.id}",
                        f"  path={node.full_path}",
                        f"  description={node.description}",
                        collection_info,
                        f"  type={node.kind.value}",
                        f"  examples={' / '.join(node.examples)}" if node.examples else "",
                    ]
                ).strip()
            )
        return prompt_items

    def is_system_only(self, node_scores: List[NodeScore]) -> bool:
        return len(node_scores) == 1 and node_scores[0].node.is_system()

    @staticmethod
    def kb_scores(node_scores: List[NodeScore], min_score: float | None = None) -> List[NodeScore]:
        out = [x for x in node_scores if x.node.is_kb()]
        if min_score is not None:
            out = [x for x in out if x.score >= min_score]
        return out

    def list_kb_collections(self) -> List[str]:
        collections = [n.collection_name for n in self.leaf_nodes if n.is_kb() and n.collection_name]
        return sorted(set(collections))

    def _load_collection_nodes(self) -> None:
        specs = self.collection_config.list_specs()
        if not specs:
            return
        for spec in specs:
            if any(node.collection_name == spec.name for node in self.id2node.values()):
                continue
            node = IntentNode(
                id=spec.name,
                name=spec.name,
                description=spec.description,
                level=IntentLevel.TOPIC,
                kind=IntentKind.KB,
                collection_name=spec.name,
                examples=spec.examples or [],
                full_path=spec.name,
            )
            if node.id in self.id2node:
                continue
            self.id2node[node.id] = node
            self.collection_nodes.append(node)

    def _format_collection_info(self, node: IntentNode) -> str:
        if not node.collection_name:
            return ""
        spec = self.collection_config.get_spec(node.collection_name)
        if not spec:
            return f"  collection={node.collection_name}"
        fields = []
        for field_item in spec.fields:
            item = f"{field_item.name}:{field_item.dtype}"
            if field_item.dtype == "FLOAT_VECTOR" and field_item.dim:
                item = f"{item}({field_item.dim})"
            fields.append(item)
        fields_text = ", ".join(fields) if fields else ""
        parts = [f"  collection={spec.name}"]
        if fields_text:
            parts.append(f"  fields={fields_text}")
        return "\n".join(parts)

    def _cap_total_intents(self, sub_intents: List[SubQuestionIntent]) -> List[SubQuestionIntent]:
        total = sum(len(s.node_scores) for s in sub_intents)
        if total <= self.settings.max_intent_count:
            return sub_intents

        candidates = []
        for i, sub in enumerate(sub_intents):
            for score in sub.node_scores:
                candidates.append((i, score))
        candidates.sort(key=lambda x: x[1].score, reverse=True)

        guaranteed = {}
        for idx, score in candidates:
            if idx not in guaranteed:
                guaranteed[idx] = [score]
            if len(guaranteed) == len(sub_intents):
                break

        remaining = self.settings.max_intent_count - sum(len(v) for v in guaranteed.values())
        if remaining > 0:
            for idx, score in candidates:
                selected = guaranteed.get(idx, [])
                if score in selected:
                    continue
                guaranteed.setdefault(idx, []).append(score)
                remaining -= 1
                if remaining == 0:
                    break

        capped: List[SubQuestionIntent] = []
        for i, sub in enumerate(sub_intents):
            capped.append(SubQuestionIntent(sub_question=sub.sub_question, node_scores=guaranteed.get(i, [])))
        return capped
