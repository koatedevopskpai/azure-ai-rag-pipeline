"""
Document ingestion pipeline — orchestrates the full extract → chunk → embed → index flow.

Architecture decision: Orchestrated pipeline over event-driven processing.

Why: For a FastAPI backend with explicit user-facing endpoints, synchronous
orchestration gives the caller a clear response (document_id, status, chunk
count) after every ingestion. Event-driven processing (Azure Service Bus +
Functions) is better for high-throughput, fire-and-forget ingestion at scale.
The orchestrated approach is simpler, testable, and sufficient for the
portfolio project. An event-driven layer can be added later without changing
the core pipeline logic.

Architecture decision: Unique document IDs via UUID over sequential IDs.

Why: UUIDs are globally unique without a central counter, enabling future
distributed processing where multiple ingestion workers run concurrently.
Sequential IDs require synchronisation.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

from app.models.schemas import (
    ChunkingStrategy,
    Document,
    DocumentMetadata,
    DocumentStatus,
)
from app.services import chunking as chunking_svc
from app.services import document_intelligence as di_svc
from app.services import embedding as embedding_svc
from app.services import search as search_svc

logger = logging.getLogger(__name__)


async def ingest_document(
    file_path: str | Path,
    filename: str | None = None,
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> Document:
    """
    Full ingestion pipeline: extract → chunk → embed → index.

    Args:
        file_path: Path to the uploaded document.
        filename: Original filename (for metadata).
        chunking_strategy: Chunking strategy to use.
        chunk_size: Override default chunk size.
        chunk_overlap: Override default chunk overlap.

    Returns:
        The ingested Document with all chunks and status.
    """
    file_path = Path(file_path)
    filename = filename or file_path.name

    # 1. Generate a unique document ID from file content hash + timestamp
    content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:12]
    document_id = f"doc-{content_hash}-{uuid.uuid4().hex[:8]}"

    doc = Document(
        id=document_id,
        metadata=DocumentMetadata(
            filename=filename,
            size_bytes=file_path.stat().st_size,
            content_type=_guess_content_type(file_path.suffix),
        ),
        status=DocumentStatus.PROCESSING,
    )

    try:
        # 2. Extract text using Azure Document Intelligence
        logger.info("Extracting text from '%s' (doc: %s)", filename, document_id)
        raw_text = di_svc.extract_text(file_path)
        doc.raw_text = raw_text

        if not raw_text:
            doc.status = DocumentStatus.FAILED
            doc.error = "Document Intelligence returned empty content"
            return doc

        # 3. Chunk the extracted text
        logger.info("Chunking text (%d chars) using %s", len(raw_text), chunking_strategy)
        chunk_dicts = chunking_svc.chunk_document(
            text=raw_text,
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )

        # 4. Create DocumentChunk objects
        from app.models.schemas import DocumentChunk

        doc_chunks = []
        for cd in chunk_dicts:
            chunk_id = f"{document_id}-chunk-{cd['chunk_index']:04d}"
            doc_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    content=cd["content"],
                    chunk_index=cd["chunk_index"],
                    strategy=cd["strategy"],
                    parent_chunk_id=cd.get("parent_chunk_id"),
                    level=cd.get("level", 0),
                )
            )
        doc.chunks = doc_chunks

        # 5. Generate embeddings for all chunks
        logger.info("Generating embeddings for %d chunks", len(doc_chunks))
        texts = [c.content for c in doc_chunks]
        vectors = embedding_svc.embed_texts(texts)

        # 6. Prepare search documents with vectors
        search_docs = []
        for chunk, vector in zip(doc_chunks, vectors):
            search_docs.append(
                {
                    "id": chunk.id,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "content_vector": vector,
                    "chunk_index": chunk.chunk_index,
                    "strategy": chunk.strategy.value,
                    "metadata": chunk.metadata,
                }
            )

        # 7. Index in Azure AI Search
        indexed_count = search_svc.index_document_chunks(search_docs)
        doc.status = DocumentStatus.INDEXED
        logger.info(
            "Document '%s' ingested: %d chars, %d chunks, %d indexed",
            document_id,
            len(raw_text),
            len(doc_chunks),
            indexed_count,
        )

    except Exception as e:
        doc.status = DocumentStatus.FAILED
        doc.error = str(e)
        logger.error("Ingestion failed for '%s': %s", document_id, e)

    return doc


def _guess_content_type(suffix: str) -> str:
    """Map file extensions to MIME types."""
    mapping = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".bmp": "image/bmp",
    }
    return mapping.get(suffix.lower(), "application/octet-stream")
