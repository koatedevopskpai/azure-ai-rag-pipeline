"""
RAG orchestration — coordinates retrieval and generation for question answering.

Architecture decision: Explicit RAG pipeline over LangChain or LlamaIndex.

Why: The Orbis role requires demonstrating understanding of RAG internals —
chunking strategies, retrieval optimisation, embedding model selection —
not just ability to call LangChain's RetrievalQA chain. Building the RAG
pipeline with explicit steps (retrieve → re-rank → format → generate) proves
we understand each component. LangChain abstracts these steps into black-box
chains that hide the architectural decisions we want to showcase.

That said, the pipeline is designed so any step can be replaced by a LangChain
equivalent without changing the API. If we later need LangChain's document
loaders or chain-of-thought prompting, swapping a service module is trivial.
"""
from __future__ import annotations

import logging
import time
from typing import AsyncGenerator

from app.core import retrieval as retrieval_svc
from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    RetrievalMode,
    SearchResult,
)
from app.services import chat as chat_svc

logger = logging.getLogger(__name__)


def answer_question(request: ChatRequest) -> ChatResponse:
    """
    Full RAG pipeline: retrieve → generate → respond.

    Args:
        request: The chat request with query, messages, and parameters.

    Returns:
        ChatResponse with answer and source citations.
    """
    overall_start = time.perf_counter()

    # 1. Retrieve relevant context
    retrieval_result = retrieval_svc.retrieve(
        query=request.query,
        top_k=request.top_k,
        mode=request.retrieval_mode,
    )

    retrieval_time = retrieval_result.retrieval_time_ms

    # 2. Generate answer with RAG context
    answer, generation_time = chat_svc.generate_answer(
        query=request.query,
        messages=request.messages if request.use_rag else None,
        sources=retrieval_result.results if request.use_rag else None,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )

    total_time = (time.perf_counter() - overall_start) * 1000

    return ChatResponse(
        answer=answer,
        sources=retrieval_result.results,
        retrieval_time_ms=retrieval_time,
        generation_time_ms=generation_time,
        total_time_ms=total_time,
        model="gpt-4o",  # Or whichever deployment is configured
    )


async def answer_question_stream(request: ChatRequest) -> AsyncGenerator[dict, None]:
    """
    Streaming RAG pipeline: retrieve → stream generate tokens via SSE.

    Architecture decision: Retrieve-then-generate over retrieve-while-generate.

    Why: In retrieve-while-generate, the LLM starts generating before
    retrieval completes, and retrieved results are injected mid-stream.
    This reduces perceived latency but adds significant complexity (stream
    interleaving, context injection timing). For the portfolio project,
    retrieve-then-generate is simpler and more predictable. The retrieval
    phase is typically 200-500ms, which is fast enough that the user does
    not perceive a delay.

    Yields:
        SSE event dicts — initially a "sources" event with retrieved
        documents, then "token" events for each generated token, then
        a "done" event with timing metadata.
    """
    # 1. Retrieve sources first
    retrieval_result = retrieval_svc.retrieve(
        query=request.query,
        top_k=request.top_k,
        mode=request.retrieval_mode,
    )

    yield {
        "event": "sources",
        "data": {
            "count": len(retrieval_result.results),
            "retrieval_time_ms": retrieval_result.retrieval_time_ms,
            "results": [r.model_dump() for r in retrieval_result.results],
        },
    }

    # 2. Stream generated tokens
    async for event in chat_svc.generate_answer_stream(
        query=request.query,
        messages=request.messages if request.use_rag else None,
        sources=retrieval_result.results if request.use_rag else None,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    ):
        yield event
