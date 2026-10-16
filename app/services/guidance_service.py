from __future__ import annotations

from collections import defaultdict
from typing import List

from app.core.config import Settings
from app.models import IntentLevel, NodeScore, SubQuestionIntent
from app.services.intent_service import IntentService


class GuidanceService:
    def __init__(self, settings: Settings, intent_service: IntentService) -> None:
        self.settings = settings
        self.intent_service = intent_service

    def detect_ambiguity(self, question: str, sub_intents: List[SubQuestionIntent]) -> str | None:
        if not self.settings.guidance_enabled or len(sub_intents) != 1:
            return None

        scores = self.intent_service.kb_scores(sub_intents[0].node_scores, min_score=self.settings.intent_min_score)
        if len(scores) < 2:
            return None

        grouped = defaultdict(list)
        for score in scores:
            grouped[self._normalize(score.node.name)].append(score)

        best_group = None
        best_top = -1.0
        for _, group in grouped.items():
            group.sort(key=lambda x: x.score, reverse=True)
            if len(group) < 2:
                continue
            ratio = group[1].score / group[0].score if group[0].score > 0 else 0
            if ratio < self.settings.guidance_ambiguity_score_ratio:
                continue
            systems = {self._resolve_system_node_id(each) for each in group}
            systems.discard("")
            if len(systems) < 2:
                continue
            if group[0].score > best_top:
                best_group = group
                best_top = group[0].score

        if not best_group:
            return None

        options = []
        for score in best_group:
            system_id = self._resolve_system_node_id(score)
            if not system_id or system_id in options:
                continue
            options.append(system_id)

        if len(options) < 2:
            return None

        names = []
        for node_id in options[: self.settings.guidance_max_options]:
            node = self.intent_service.get_node_by_id(node_id)
            names.append(node.name if node else node_id)

        normalized_question = self._normalize(question)
        for name in names:
            if len(name) > 1 and self._normalize(name) in normalized_question:
                return None

        option_text = "\n".join([f"{i + 1}) {name}" for i, name in enumerate(names)])
        topic_name = best_group[0].node.name
        return (
            f"关于{topic_name}，在知识库中检索到了以下内容：\n"
            f"{option_text}\n\n"
            "请问你具体想了解哪个？请回复数字选择（可多选，如 1,2），或回复‘都/全部’"
        )

    def _resolve_system_node_id(self, score: NodeScore) -> str:
        current = score.node
        parent = self.intent_service.get_node_by_id(current.parent_id) if current.parent_id else None

        while True:
            if current.level == IntentLevel.CATEGORY and (parent is None or parent.level == IntentLevel.DOMAIN):
                return current.id
            if parent is None:
                return current.id
            current = parent
            parent = self.intent_service.get_node_by_id(current.parent_id) if current.parent_id else None

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(ch.lower() for ch in text if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"))
