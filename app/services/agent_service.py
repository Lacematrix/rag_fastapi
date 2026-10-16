from __future__ import annotations

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from app.core.config import Settings
from app.models import RetrievalContext
from app.prompts import PromptRegistry
from app.services.llm_service import LLMService
from app.core.service_logging import service_logger


class AgentService:
    def __init__(self, settings: Settings, llm_service: LLMService) -> None:
        self.settings = settings
        self.llm_service = llm_service

    @service_logger()
    def answer(self, question: str, retrieval_context: RetrievalContext, history: list[dict] | None = None) -> str:
        context_blob = retrieval_context.kb_context or "未检索到与问题相关的文档内容。"
        history = history or []
        history_messages = self._build_history_messages(history)

        @tool("get_retrieval_context")
        def get_retrieval_context(_: str) -> str:
            """Return retrieval context for current question."""
            return context_blob

        tools = [get_retrieval_context]
        llm = self.llm_service.build_chat_model(temperature=0.2, top_p=0.8)

        system_prompt = PromptRegistry.get_template_text("answer-chat-kb")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("history"),
                MessagesPlaceholder("agent_scratchpad"),
                ("human", "{input}"),
            ]
        )

        try:
            agent = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=3)
            result = executor.invoke({"input": question, "history": history_messages})
            output = result.get("output", "")
            if output:
                return output
        except Exception:
            pass

        fallback_messages = [SystemMessage(content=system_prompt)]
        fallback_messages.extend(history_messages)
        fallback_messages.append(
            HumanMessage(content=f"问题: {question}\n\n文档内容:\n{context_blob}")
        )
        return self.llm_service.invoke(fallback_messages, temperature=0.1, top_p=0.5)

    def _build_history_messages(self, history: list[dict]) -> list:
        if not history:
            return []
        recent = history[-6:]
        return self.llm_service.to_messages(recent)
