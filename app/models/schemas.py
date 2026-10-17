"""
Pydantic models — schema enforcement for every data type in the RAG pipeline.

Architecture decision: Pydantic v2 models over dataclasses or TypedDicts.

Why: Pydantic v2 (pydantic-core, written in Rust) is 5-10x faster than
v1 for validation. It gives us runtime type checking, JSON Schema generation
for OpenAPI docs, and serialisation/deserialisation out of the box.
Dataclasses lack validation. TypedDicts are type-check-time only.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Enums — explicit state machines for the RAG pipeline
# ------------------------------------------------------------------
# Architecture decision: StrEnum over string constants.
# Why: StrEnum gives you type-safe enum values that serialise to strings
# in API responses. String constants are just strings — any function that
# accepts a str will accept "invalid_chunking_strategy_value" silently.


class ChunkingStrategy(str, Enum):
    """Supported text chunking strategies."""

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


class DocumentStatus(str, Enum):
    """Lifecycle state of a document in the pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class RetrievalMode(str, Enum):
    """Search mode for document retrieval."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class MessageRole(str, Enum):
    """Role of a message in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ------------------------------------------------------------------
# Document models
# ------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    """
    Metadata extracted from or assigned to a document.

    For financial research documents, this includes tickers, sectors,
    document type, and date of publication — enabling filtered retrieval
    (e.g., "only Q3 earnings reports for MSFT").
    """

    filename: str = Field(..., description="Original filename")
    source: str = Field(default="upload", description="Source of the document")
    content_type: str = Field(default="application/octet-stream", description="MIME type")
    size_bytes: int = Field(default=0, description="File size in bytes")
    page_count: int | None = Field(default=None, description="Number of pages")
    author: str | None = Field(default=None, description="Document author")
    created_at: datetime | None = Field(default=None, description="Document creation date")
    tickers: list[str] = Field(default_factory=list, description="Stock tickers mentioned")
    sectors: list[str] = Field(default_factory=list, description="Industry sectors")
    doc_type: str | None = Field(
        default=None, description="Document type (report, filing, article)"
    )


class Document(BaseModel):
    """
    A document after ingestion and parsing.
    """

    id: str = Field(..., description="Unique document identifier")
    metadata: DocumentMetadata
    raw_text: str = Field(..., description="Full extracted text")
    chunks: list[DocumentChunk] = Field(default_factory=list, description="Split chunks")
    status: DocumentStatus = DocumentStatus.PENDING
    error: str | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """
    A single chunk of a document after splitting.
    """

    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in the document (0-based)")
    strategy: ChunkingStrategy = Field(..., description="Strategy used to create this chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk-level metadata")

    # For hierarchical chunking: parent-child relationships
    parent_chunk_id: str | None = Field(default=None, description="Parent chunk ID (hierarchical)")
    level: int = Field(
        default=0, description="Hierarchical level (0=section, 1=subsection, 2=paragraph)"
    )


# ------------------------------------------------------------------
# Embedding models
# ------------------------------------------------------------------


class EmbeddingVector(BaseModel):
    """
    A chunk with its embedding vector.
    """

    chunk_id: str = Field(..., description="Chunk identifier")
    vector: list[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Embedding model name")
    dimensions: int = Field(..., description="Vector dimensionality")


class SearchResult(BaseModel):
    """
    A single search result from Azure AI Search.
    """

    chunk_id: str
    document_id: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlights: list[str] = Field(default_factory=list, description="Matched text snippets")


class RetrievalResult(BaseModel):
    """
    Complete retrieval result from a RAG query.
    """

    query: str
    results: list[SearchResult]
    total_count: int
    retrieval_time_ms: float
    mode: RetrievalMode
    query_expansion: str | None = None


# ------------------------------------------------------------------
# Chat / RAG models
# ------------------------------------------------------------------


class ChatMessage(BaseModel):
    """
    A message in the RAG chat conversation.
    """

    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    """
    Request body for the RAG chat endpoint.
    """

    messages: list[ChatMessage] = Field(..., min_length=1, description="Conversation history")
    query: str = Field(..., description="The user's question")
    use_rag: bool = Field(default=True, description="Whether to use RAG retrieval")
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=50, description="Number of documents to retrieve")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=4096, ge=64, le=16384, description="Max response tokens")
    stream: bool = Field(default=False, description="Whether to stream the response via SSE")


class ChatResponse(BaseModel):
    """
    Response from the RAG chat endpoint (non-streaming).
    """

    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    model: str = ""


class IngestionResponse(BaseModel):
    """
    Response from the document ingestion endpoint.
    """

    document_id: str
    filename: str
    status: DocumentStatus
    chunks_count: int
    size_bytes: int
    message: str


class HealthCheck(BaseModel):
    """
    Health check response.
    """

    status: str = "healthy"
    version: str = "1.0.0"
    services: dict[str, str] = Field(default_factory=dict)
