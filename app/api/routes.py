"""
API routes — REST endpoints for document ingestion, RAG query, and streaming.

Architecture decision: Separate route file over inline route registration.

Why: As the API grows (health, ingestion, query, streaming, admin), inline
routes in main.py become unmanageable. A routes.py file with a single
APIRouter keeps the endpoint definitions organised and allows each endpoint
to be tested independently.

Architecture decision: File upload via multipart/form-data over base64 JSON.

Why: Financial documents can be 50MB+ PDFs with embedded images. Base64
encoding adds 33% overhead and prevents streaming uploads. Multipart uploads
are the standard for file APIs and work with any HTTP client.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.config import UPLOAD_DIR, settings
from app.core import agent as agent_svc
from app.core import rag as rag_svc
from app.core import retrieval as retrieval_svc
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    IngestionResponse,
    RetrievalMode,
    RetrievalResult,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ------------------------------------------------------------------
# Document ingestion
# ------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Upload and ingest a document",
    description=(
        "Upload a financial document (PDF, DOCX, TXT, MD) and run it through "
        "the full ingestion pipeline: extract via Azure Document Intelligence → "
        "chunk via selected strategy → embed via Azure OpenAI → index in Azure AI Search."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest (PDF, DOCX, TXT, MD)"),
    chunking_strategy: str = Query(
        default="recursive",
        description="Chunking strategy: recursive, semantic, or hierarchical",
    ),
):
    """
    Upload a document and ingest it into the RAG pipeline.

    The document is saved temporarily, processed through the full extraction
    pipeline, and indexed in Azure AI Search for future queries.

    Architecture decision: Synchronous ingestion over async background task.

    Why: For the portfolio project, synchronous ingestion returns a complete
    response with chunk count and status. Async ingestion (return immediately,
    process in background) is better for production but requires polling or
    webhook callbacks that add complexity without proportional benefit.
    """
    # Validate file type
    allowed_types = {".pdf", ".docx", ".doc", ".txt", ".md"}
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_types)}",
        )

    # Validate chunking strategy
    strategy_map = {
        "recursive": "recursive",
        "semantic": "semantic",
        "hierarchical": "hierarchical",
    }
    if chunking_strategy not in strategy_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chunking strategy '{chunking_strategy}'. Choose: recursive, semantic, hierarchical",
        )

    # Save uploaded file to a temporary location
    from app.services import ingestion as ingestion_svc

    upload_dir = UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_path = upload_dir / f"{file.filename}"
    try:
        content = await file.read()
        temp_path.write_bytes(content)

        # Run ingestion pipeline
        from app.models.schemas import ChunkingStrategy

        doc = await ingestion_svc.ingest_document(
            file_path=temp_path,
            filename=file.filename,
            chunking_strategy=ChunkingStrategy(chunking_strategy),
        )

        message = f"Document '{file.filename}' ingested successfully with {len(doc.chunks)} chunks."

        if doc.status.value == "failed":
            message = f"Ingestion failed: {doc.error}"

        return IngestionResponse(
            document_id=doc.id,
            filename=file.filename or "unknown",
            status=doc.status,
            chunks_count=len(doc.chunks),
            size_bytes=doc.metadata.size_bytes,
            message=message,
        )

    except Exception as e:
        logger.exception("Ingestion failed for '%s'", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


# ------------------------------------------------------------------
# RAG query (non-streaming)
# ------------------------------------------------------------------


@router.post(
    "/query",
    response_model=ChatResponse,
    summary="Ask a question using RAG",
    description=(
        "Ask a question about your ingested documents. The pipeline retrieves "
        "relevant context using hybrid search (semantic + keyword), then generates "
        "an answer using Azure OpenAI with source citations."
    ),
)
async def query_documents(request: ChatRequest):
    """
    Non-streaming RAG query — sends the full answer as a JSON response.

    Use this endpoint when you need the complete answer at once (e.g., for
    programmatic consumption or when streaming is not required).
    """
    try:
        response = rag_svc.answer_question(request)
        return response
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# RAG query (streaming via SSE)
# ------------------------------------------------------------------


@router.post(
    "/query/stream",
    summary="Ask a question with streaming response (SSE)",
    description=(
        "Same as /query but returns the answer as a Server-Sent Events stream. "
        "Events: 'sources' (retrieved context), 'token' (generated text tokens), "
        "'done' (completion signal with timing)."
    ),
)
async def query_documents_stream(request: ChatRequest):
    """
    Streaming RAG query — returns tokens via SSE as they are generated.

    Architecture decision: SSE over WebSocket for token streaming.

    Why: SSE is HTTP-based, works with standard load balancers (no sticky
    sessions), supports automatic reconnection via EventSource API, and is
    simpler to implement. WebSocket is justified when the client needs to
    send multiple requests over a single connection, which is not the case
    here (one query → one stream of tokens).
    """

    async def event_generator() -> AsyncGenerator[dict, None]:
        async for event in rag_svc.answer_question_stream(request):
            yield event

    return EventSourceResponse(event_generator())


# ------------------------------------------------------------------
# Retrieval-only (no generation)
# ------------------------------------------------------------------


@router.post(
    "/retrieve",
    response_model=RetrievalResult,
    summary="Retrieve documents without generating an answer",
    description=(
        "Search the document index and return the most relevant chunks. "
        "Useful for debugging retrieval quality or for building custom "
        "retrieval flows where you want to control how context is injected."
    ),
)
async def retrieve_documents(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(default=10, ge=1, le=50, description="Number of results"),
    mode: RetrievalMode = Query(default=RetrievalMode.HYBRID, description="Search mode"),
    expand: bool = Query(default=True, description="Apply query expansion"),
):
    """Retrieve relevant document chunks without LLM generation."""
    try:
        result = retrieval_svc.retrieve(query=query, top_k=top_k, mode=mode, expand=expand)
        return result
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Agentic workflow
# ------------------------------------------------------------------


@router.post(
    "/agent",
    summary="Run an agentic workflow",
    description=(
        "Use Azure OpenAI function calling to run an agentic workflow. "
        "The agent decides which tools to call (search, extract metrics, "
        "compare documents) based on the user's request."
    ),
)
async def run_agent(
    query: str = Query(..., description="The task for the agent"),
):
    """Run an agentic analysis workflow using function calling."""
    try:
        result = agent_svc.run_agentic_workflow(query)
        return result
    except Exception as e:
        logger.exception("Agent workflow failed")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# Index management
# ------------------------------------------------------------------


@router.post(
    "/index/create",
    summary="Create or update the search index",
    description="Ensure the Azure AI Search index exists with the correct schema.",
)
async def create_index():
    """Create or update the Azure AI Search index."""
    try:
        from app.services import search as search_svc

        search_svc.ensure_index()
        return {"status": "ok", "message": f"Index '{settings.AZURE_SEARCH_INDEX_NAME}' ready"}
    except Exception as e:
        logger.exception("Index creation failed")
        raise HTTPException(status_code=500, detail=str(e))
