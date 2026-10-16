from __future__ import annotations

import json
import re
from typing import Iterable, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.service_logging import service_logger


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @service_logger()
    def build_chat_model(self, temperature: float = 0.1, top_p: float = 0.3) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=self.settings.chat_base_url,
            api_key=self.settings.chat_api_key,
            model=self.settings.chat_model,
            temperature=temperature,
            model_kwargs={"top_p": top_p},
        )

    @service_logger()
    def invoke(self, messages: Sequence[BaseMessage], temperature: float = 0.1, top_p: float = 0.3) -> str:
        model = self.build_chat_model(temperature=temperature, top_p=top_p)
        result = model.invoke(list(messages))
        return str(result.content)

    @staticmethod
    def to_messages(history: Iterable[dict]) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for item in history:
            role = (item.get("role") or "user").lower()
            content = item.get("content", "")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    @staticmethod
    def extract_json_block(raw: str) -> str:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
        return cleaned.strip()

    @staticmethod
    def try_parse_json(raw: str):
        text = LLMService.extract_json_block(raw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
