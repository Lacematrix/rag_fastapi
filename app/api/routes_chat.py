from __future__ import annotations

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.container import get_container
from app.core.service_logging import service_logger
from app.prompts import PromptRegistry
from app.models import RewriteResult
from app.schemas import ChatRequest, ChatResponse, IntentScoreView, RetrievedChunkView

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
@service_logger("app.api.routes_chat.chat")
def chat(request: ChatRequest) -> ChatResponse:
    container = get_container()
    settings = container.settings

    base_rewrite = _build_base_rewrite(request.question)
    intent_analysis = container.intent_service.analyze_intent(base_rewrite.rewritten_question)

    if intent_analysis.chat_type == "clarify" or intent_analysis.confidence < settings.intent_route_min_confidence:
        follow_up = intent_analysis.follow_up or "你是想闲聊，还是希望我从资料里帮你查找并回答？"
        return ChatResponse(
            answer=follow_up,
            guidance_prompt=follow_up,
            rewritten_question=base_rewrite.rewritten_question,
            sub_questions=base_rewrite.sub_questions,
            intents=_collect_intents_from_candidates(container, intent_analysis.intents),
            chunks=[],
            intent_status="clarify",
            intent_confidence=intent_analysis.confidence,
            need_rag=intent_analysis.need_rag,
            chat_type=intent_analysis.chat_type,
        )

    if intent_analysis.chat_type == "chitchat" or not intent_analysis.need_rag:
        system_prompt = PromptRegistry.get_template_text("answer-chat-system")
        answer = container.llm_service.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=base_rewrite.rewritten_question),
            ],
            temperature=0.5,
            top_p=0.8,
        )
        return ChatResponse(
            answer=answer,
            rewritten_question=base_rewrite.rewritten_question,
            sub_questions=base_rewrite.sub_questions,
            intents=_collect_intents_from_candidates(container, intent_analysis.intents),
            chunks=[],
            intent_status="chitchat",
            intent_confidence=intent_analysis.confidence,
            need_rag=intent_analysis.need_rag,
            chat_type=intent_analysis.chat_type,
        )

    history = [item.model_dump() for item in request.history]
    rewrite_result = container.rewrite_service.rewrite_with_split(request.question, history)
    sub_intents = container.intent_service.resolve(rewrite_result)
    guidance_prompt = container.guidance_service.detect_ambiguity(
        rewrite_result.rewritten_question,
        sub_intents,
    )
    if guidance_prompt:
        return ChatResponse(
            answer=guidance_prompt,
            guidance_prompt=guidance_prompt,
            rewritten_question=rewrite_result.rewritten_question,
            sub_questions=rewrite_result.sub_questions,
            intents=_collect_intents(sub_intents),
            chunks=[],
            intent_status="rag",
            intent_confidence=intent_analysis.confidence,
            need_rag=intent_analysis.need_rag,
            chat_type=intent_analysis.chat_type,
        )

    top_k = request.top_k or settings.top_k
    retrieval_ctx = container.retrieval_service.retrieve(sub_intents, top_k, rewrite_result.rewritten_question, request.question)

    if retrieval_ctx.is_empty():
        answer = "未检索到与问题相关的文档内容。"
    else:
        answer = container.agent_service.answer(
            rewrite_result.rewritten_question,
            retrieval_ctx,
            history,
        )

    return ChatResponse(
        answer=answer,
        rewritten_question=rewrite_result.rewritten_question,
        sub_questions=rewrite_result.sub_questions,
        intents=_collect_intents(sub_intents),
        chunks=_collect_chunks(retrieval_ctx.intent_chunks),
        intent_status="rag",
        intent_confidence=intent_analysis.confidence,
        need_rag=intent_analysis.need_rag,
        chat_type=intent_analysis.chat_type,
    )


def _build_base_rewrite(question: str) -> RewriteResult:
    normalized = question.strip()
    return RewriteResult(rewritten_question=normalized, sub_questions=[normalized])


def _collect_intents_from_candidates(container, candidates):
    rows = []
    for item in candidates:
        node = container.intent_service.get_node_by_id(item.id)
        if not node:
            continue
        rows.append(
            IntentScoreView(
                sub_question="",
                intent_id=node.id,
                intent_name=node.name,
                score=float(item.score),
            )
        )
    return rows


def _collect_intents(sub_intents):
    rows = []
    for sub in sub_intents:
        for node_score in sub.node_scores:
            rows.append(
                IntentScoreView(
                    sub_question=sub.sub_question,
                    intent_id=node_score.node.id,
                    intent_name=node_score.node.name,
                    score=node_score.score,
                )
            )
    return rows


def _collect_chunks(intent_chunks):
    rows = []
    seen = set()
    for intent_id, chunks in intent_chunks.items():
        for chunk in chunks:
            key = chunk.id or f"{intent_id}:{hash(chunk.text)}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                RetrievedChunkView(
                    id=chunk.id,
                    text=chunk.text,
                    score=chunk.score,
                    intent_id=intent_id,
                    retrieval_score=chunk.retrieval_score,
                    rerank_score=chunk.rerank_score,
                )
            )
    return rows
