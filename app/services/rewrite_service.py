from __future__ import annotations

import re
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import Settings
from app.models import RewriteResult
from app.prompts import PromptRegistry
from app.services.llm_service import LLMService



class QueryRewriteService:
    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self.settings = settings
        self.llm_service = llm_service
        self.term_mapping: Dict[str, str] = {
            "oa": "OA系统",
            "redis集群": "Redis 集群",
        }

    def rewrite_with_split(self, user_question: str, history: List[dict]) -> RewriteResult:
        normalized = self._normalize(user_question)
        if not self.settings.query_rewrite_enabled:
            subs = self._rule_based_split(normalized)
            return RewriteResult(rewritten_question=normalized, sub_questions=subs)

        system_prompt = PromptRegistry.get_template_text("user-question-rewrite")
        messages = [SystemMessage(content=system_prompt)]
        if history:
            recent = [h for h in history if h.get("role") in {"user", "assistant"}][-4:]
            messages.extend(self.llm_service.to_messages(recent))
        messages.append(HumanMessage(content=normalized))

        raw = self.llm_service.invoke(messages, temperature=0.1, top_p=0.3)
        parsed = self.llm_service.try_parse_json(raw)

        if isinstance(parsed, dict):
            rewrite = str(parsed.get("rewrite", "")).strip()
            sub_questions = parsed.get("sub_questions") or []
            subs = [str(x).strip() for x in sub_questions if str(x).strip()]
            if rewrite:
                return RewriteResult(
                    rewritten_question=rewrite,
                    sub_questions=subs or [rewrite],
                )

        fallback_subs = self._rule_based_split(normalized)
        return RewriteResult(rewritten_question=normalized, sub_questions=fallback_subs)

    def _normalize(self, question: str) -> str:
        q = question.strip()
        for src, dst in self.term_mapping.items():
            q = re.sub(src, dst, q, flags=re.IGNORECASE)
        return q

    @staticmethod
    def _rule_based_split(question: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"[?？。；;\n]+", question) if p.strip()]
        if not parts:
            return [question]
        return [p if p.endswith(("?", "？")) else f"{p}？" for p in parts]
