"""
Azure OpenAI Chat service — LLM-powered answer generation with RAG context.

Architecture decision: Azure OpenAI chat completions over Azure AI Foundry
model deployment.

Why: For a FastAPI RAG backend, the chat completions API is the simplest
integration path. AI Foundry's model deployment adds a management layer
(monitoring, A/B testing, content safety) that is valuable in production
but adds unnecessary complexity for the RAG API itself. Foundry integration
is added as a separate module when the project needs prompt flow management.

Architecture decision: Structured system prompt with clear sections over a
free-form instruction.

Why: A structured prompt (RULES, CONTEXT, FORMAT) makes the LLM's behaviour
predictable and testable. Each section is independently auditable for prompt
injection vulnerabilities. A free-form prompt is harder to debug and
maintain.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionChunk
from openai import NOT_GIVEN

from app.config import settings
from app.models.schemas import ChatMessage, MessageRole, SearchResult

logger = logging.getLogger(__name__)

_client: AzureOpenAI | None = None

# ------------------------------------------------------------------
# System prompt — grounds the LLM in the RAG context with strict
# behavioural guardrails.
# ------------------------------------------------------------------
# Architecture decision: Dynamic system prompt over hardcoded string.
# Why: The system prompt can be extended to include the current date,
# user-specific instructions, or compliance disclaimers without changing
# the calling code. The base prompt is defined here and build_system_prompt()
# injects dynamic context.

_BASE_SYSTEM_PROMPT = """You are a financial research assistant powered by a RAG pipeline. You help investment professionals analyse documents, extract insights, and answer questions with evidence.

## RULES
1. Answer based ONLY on the provided context. If the context does not contain the answer, say "I cannot find information about that in the provided documents."
2. Cite your sources using the document ID and chunk index: [Doc: {document_id}, Chunk: {chunk_index}].
3. Do not make up financial data, metrics, or projections. If the context contains conflicting information, present both sides with their sources.
4. Use precise financial terminology when appropriate (EBITDA, CAGR, DCF, etc.) but explain terms that a general investment professional might not know.
5. For numerical claims, always include the unit (£, %, $) and time period.
6. If the user asks a question outside the scope of the provided documents, politely decline to answer.
7. Keep responses concise but thorough. Investment professionals prefer bullet points over prose for data-heavy answers.

## FORMAT
- Use markdown for structure: **bold** for key terms, bullet lists for data points, tables for comparisons.
- Maximum 500 words unless the user explicitly asks for detail.
- Start with a direct answer, then provide supporting evidence, then note any caveats."""


def _get_client() -> AzureOpenAI:
    """Lazy-initialised Azure OpenAI client."""
    global _client
    if _client is not None:
        return _client
    if not settings.AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
    _client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY or None,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )
    return _client


def build_system_prompt(sources: list[SearchResult] | None = None) -> str:
    """
    Build the system prompt with optional RAG context.

    Args:
        sources: Search results to inject as context. If None, the prompt
            instructs the model to answer without document context.

    Returns:
        The complete system prompt string.
    """
    prompt = _BASE_SYSTEM_PROMPT

    if sources:
        context_parts = []
        for i, src in enumerate(sources, 1):
            context_parts.append(
                f"[SOURCE {i}] Document: {src.document_id}\n"
                f"[SOURCE {i}] Chunk: {src.chunk_id}\n"
                f"[SOURCE {i}] Content: {src.content}\n"
            )
        context = "\n".join(context_parts)
        prompt += f"\n\n## CONTEXT\nThe following documents are relevant to the user's question. Use them to answer:\n\n{context}"
    else:
        prompt += "\n\n## CONTEXT\nNo documents are available. Answer based on your general knowledge, but clearly state that you do not have access to the document store."

    return prompt


def generate_answer(
    query: str,
    messages: list[ChatMessage] | None = None,
    sources: list[SearchResult] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    model: str | None = None,
) -> tuple[str, float]:
    """
    Generate an answer using Azure OpenAI chat completion with RAG context.

    Architecture decision: Chat completions API over completion API.

    Why: The chat completions API supports system messages, conversation
    history, and the structured prompt format required for RAG. The older
    completion API treats the entire input as a single text prompt with
    no role distinction, making it impossible to separate system instructions
    from user queries and RAG context.

    Args:
        query: The user's question.
        messages: Optional conversation history for multi-turn chat.
        sources: Search results to use as RAG context.
        temperature: LLM temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens: Maximum response tokens.
        model: Optional override for the deployment name.

    Returns:
        Tuple of (answer_text, generation_time_ms).
    """
    client = _get_client()
    deployment = model or settings.AZURE_OPENAI_CHAT_DEPLOYMENT
    start_time = time.perf_counter()

    system_prompt = build_system_prompt(sources)

    # Build the message list: system + history + user query
    chat_messages = [{"role": "system", "content": system_prompt}]

    if messages:
        for msg in messages:
            chat_messages.append({"role": msg.role.value, "content": msg.content})

    chat_messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=deployment,
        messages=chat_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    answer = response.choices[0].message.content or ""
    logger.info("Generated %d-char answer in %.1fms", len(answer), elapsed_ms)
    return answer, elapsed_ms


async def generate_answer_stream(
    query: str,
    messages: list[ChatMessage] | None = None,
    sources: list[SearchResult] | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    model: str | None = None,
) -> Any:
    """
    Generate a streaming answer using Azure OpenAI with SSE support.

    Architecture decision: Async generator over callback-based streaming.

    Why: FastAPI + SSE works naturally with async generators. Each yielded
    token is sent as an SSE event to the client. The generator cleans up
    the Azure OpenAI response iterator when the client disconnects.

    Yields:
        SSE event dicts with token and metadata fields.
    """
    client = _get_client()
    deployment = model or settings.AZURE_OPENAI_CHAT_DEPLOYMENT

    system_prompt = build_system_prompt(sources)

    chat_messages = [{"role": "system", "content": system_prompt}]
    if messages:
        for msg in messages:
            chat_messages.append({"role": msg.role.value, "content": msg.content})
    chat_messages.append({"role": "user", "content": query})

    stream = client.chat.completions.create(
        model=deployment,
        messages=chat_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            yield {
                "event": "token",
                "data": token,
            }

    yield {
        "event": "done",
        "data": "[DONE]",
    }
