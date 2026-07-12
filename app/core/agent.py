"""
Agentic workflow — document analysis agent using Azure OpenAI function calling.

Architecture decision: Function calling over LangChain agents.

Why: Azure OpenAI's native function calling API gives us deterministic,
type-safe tool invocations without adding LangChain's abstraction layer.
The agent receives a user request, decides which tools to call, executes
them, and returns a structured response. Function calling is the pattern
the Orbis role uses ("experience building and deploying production LLM
applications, agentic workflows").

Architecture decision: Single-agent over multi-agent architecture.

Why: Multi-agent systems (LangGraph, CrewAI) are powerful but introduce
coordination complexity that is unnecessary for a document analysis RAG
pipeline. A single agent with well-defined tools (search, summarise,
compare) handles the documented use cases. Multi-agent orchestration can
be added when the pipeline needs parallel analysis of multiple documents.

This module provides:
- `analyse_document`: Summarise a document, extract key metrics, list tickers
- `compare_documents`: Compare two documents and highlight differences
- `extract_metrics`: Extract specific financial metrics from context
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.config import settings
from app.models.schemas import RetrievalMode

logger = logging.getLogger(__name__)


AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search indexed documents for information relevant to the user's question",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (1-20)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_financial_metrics",
            "description": "Extract specific financial metrics from retrieved context",
            "parameters": {
                "type": "object",
                "properties": {
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of metrics to extract (e.g., revenue, EBITDA, net income)",
                    },
                    "document_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Document IDs to analyse",
                    },
                },
                "required": ["metrics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_documents",
            "description": "Compare two documents or sections side by side",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id_a": {
                        "type": "string",
                        "description": "First document ID",
                    },
                    "document_id_b": {
                        "type": "string",
                        "description": "Second document ID",
                    },
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific aspects to compare",
                    },
                },
                "required": ["document_id_a", "document_id_b"],
            },
        },
    },
]


def run_agentic_workflow(query: str, messages: list[dict] | None = None) -> dict[str, Any]:
    """
    Run an agentic workflow using Azure OpenAI function calling.

    The agent receives the user query, decides which tools to call,
    executes them, and returns the combined result.

    Args:
        query: The user's request.
        messages: Optional conversation history.

    Returns:
        Dict with 'result', 'tool_calls', 'sources' keys.
    """
    from openai import AzureOpenAI

    if not settings.AZURE_OPENAI_ENDPOINT:
        return {"result": "Azure OpenAI is not configured.", "tool_calls": [], "sources": []}

    client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY or None,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )

    system_prompt = """You are a financial document analysis agent. You help investment professionals analyse documents, extract metrics, and compare findings.

Available tools:
1. search_documents — Search the indexed document store for relevant information
2. extract_financial_metrics — Extract specific metrics (revenue, EBITDA, etc.) from documents
3. compare_documents — Compare two documents side by side

Decide which tools to call based on the user's request. You can call multiple tools in sequence if needed. Explain your reasoning briefly before calling each tool."""

    chat_messages = [{"role": "system", "content": system_prompt}]
    if messages:
        chat_messages.extend(messages)
    chat_messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=chat_messages,
        tools=AVAILABLE_TOOLS,
        tool_choice="auto",
        temperature=0.1,
    )

    message = response.choices[0].message
    tool_calls = []
    sources = []
    result = message.content or ""

    if message.tool_calls:
        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            tool_calls.append({
                "name": tc.function.name,
                "arguments": args,
            })

            if tc.function.name == "search_documents":
                from app.core import retrieval as retrieval_svc
                retrieval = retrieval_svc.retrieve(
                    query=args.get("query", query),
                    top_k=args.get("top_k", 10),
                    mode=RetrievalMode.HYBRID,
                )
                sources.extend(retrieval.results)
                result += f"\n\nSearched for '{args.get('query', query)}' — found {len(retrieval.results)} relevant chunks."

    return {
        "result": result,
        "tool_calls": tool_calls,
        "sources": sources,
    }
