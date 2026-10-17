"""
Azure AI Search service — vector + keyword hybrid search for RAG retrieval.

Architecture decision: Azure AI Search over Pinecone, Weaviate, or Qdrant.

Why: The Orbis role requires Azure AI Foundry / AI Search experience.
Beyond portfolio alignment, Azure AI Search offers:
1. Hybrid search (semantic + keyword) out of the box — no separate setup
2. Integrated vector storage + keyword indexing in a single service
3. Semantic ranking with Microsoft's cross-encoder models
4. Managed service — no infrastructure to operate on AKS
5. Role-based access control for enterprise compliance

Architecture decision: Index schema with separate content and metadata fields.

Why: Storing content and metadata as separate fields allows the search
service to return only the metadata (for building source citations) without
fetching the full content, reducing response size and latency. The content
field is configured for semantic search (max 32K chars) and the metadata
is stored as a searchable collection.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizableTextQuery,
    VectorizedQuery,
)

from app.config import settings
from app.models.schemas import RetrievalMode, SearchResult

logger = logging.getLogger(__name__)

_clients: dict[str, SearchClient | SearchIndexClient] = {}


def _get_index_client() -> SearchIndexClient:
    """Lazy-initialised Search Index client for creating/managing indexes."""
    key = "index_client"
    if key not in _clients:
        _clients[key] = SearchIndexClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
        )
    return _clients[key]


def _get_search_client(index_name: str | None = None) -> SearchClient:
    """Lazy-initialised Search client for querying an index."""
    name = index_name or settings.AZURE_SEARCH_INDEX_NAME
    cache_key = f"search_client:{name}"
    if cache_key not in _clients:
        _clients[cache_key] = SearchClient(
            endpoint=settings.AZURE_SEARCH_ENDPOINT,
            index_name=name,
            credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
        )
    return _clients[cache_key]


def ensure_index(index_name: str | None = None) -> None:
    """
    Create or update the search index if it does not exist.

    Architecture decision: Schema-on-write over schema-on-read.

    Why: Azure AI Search requires a predefined schema. This forces explicit
    decisions about which fields are searchable, filterable, facetable, and
    sortable — rather than discovering these at query time via schema-on-read.
    For a production RAG pipeline, schema-on-write prevents the class of bugs
    where a field is stored but cannot be queried.

    This function is idempotent — safe to call on every startup.
    """
    index_client = _get_index_client()
    name = index_name or settings.AZURE_SEARCH_INDEX_NAME

    try:
        index_client.get_index(name)
        logger.info("Search index '%s' already exists", name)
        return
    except ResourceNotFoundError:
        logger.info("Creating search index '%s'", name)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
        SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=False,
        ),
        SimpleField(
            name="strategy",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,  # text-embedding-3-large
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name="rag-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=None,
            prioritized_content_fields=[SemanticField(field_name="content")],
        ),
    )

    index = SearchIndex(
        name=name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )

    index_client.create_or_update_index(index)
    logger.info("Search index '%s' created successfully", name)


def index_document_chunks(chunks: list[dict[str, Any]], index_name: str | None = None) -> int:
    """
    Index a batch of document chunks into Azure AI Search.

    Args:
        chunks: List of documents to index. Each document should have:
            - id: str
            - chunk_id: str
            - document_id: str
            - content: str
            - content_vector: list[float] (embedding)
            - chunk_index: int
            - metadata: dict (will be JSON-serialised)
            - strategy: str
        index_name: Optional override for the index name.

    Returns:
        Number of chunks successfully indexed.
    """
    ensure_index(index_name)
    client = _get_search_client(index_name)

    # Serialise metadata to JSON for storage
    for chunk in chunks:
        if isinstance(chunk.get("metadata"), dict):
            chunk["metadata"] = json.dumps(chunk["metadata"])

    result = client.upload_documents(documents=chunks)
    succeeded = sum(1 for r in result if r.succeeded)
    logger.info("Indexed %d/%d chunks", succeeded, len(chunks))
    return succeeded


def search(
    query: str,
    query_vector: list[float] | None = None,
    top_k: int = 10,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    index_name: str | None = None,
    filters: str | None = None,
) -> list[SearchResult]:
    """
    Search the index using the specified retrieval mode.

    Architecture decision: Three retrieval modes in one function over
    separate functions for each mode.

    Why: The retrieval mode selection is a user-facing parameter (the UI
    lets users choose keyword, semantic, or hybrid). A single function
    with a mode parameter keeps the API surface small and the routing
    logic in one place. Each mode path is separated by clear conditionals.

    Args:
        query: The search query text.
        query_vector: Pre-computed query embedding (required for vector/hybrid).
        top_k: Number of results to return.
        mode: Retrieval mode (keyword, semantic, hybrid).
        index_name: Optional override for the index name.
        filters: OData filter string for result filtering.

    Returns:
        List of SearchResult objects, sorted by relevance.
    """
    client = _get_search_client(index_name)
    start_time = time.perf_counter()

    if mode == RetrievalMode.KEYWORD:
        results = client.search(
            search_text=query,
            query_type=QueryType.SIMPLE,
            top=top_k,
            filter=filters,
            include_total_count=True,
        )

    elif mode == RetrievalMode.SEMANTIC:
        if not query_vector:
            raise ValueError("query_vector is required for semantic search mode")
        # Pure vector search using the query vector
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector",
        )
        results = client.search(
            search_text="",
            vector_queries=[vector_query],
            top=top_k,
            filter=filters,
            include_total_count=True,
        )

    elif mode == RetrievalMode.HYBRID:
        if not query_vector:
            raise ValueError("query_vector is required for hybrid search mode")
        # Architecture decision: Hybrid search over pure vector search.
        # Why: Hybrid search combines keyword BM25 scoring with vector
        # similarity, then uses Microsoft's semantic reranker to produce
        # a single relevance score. In financial RAG, keyword matching is
        # critical for ticker symbols ("MSFT", "AAPL") and exact phrases
        # ("Q3 2024 earnings") where pure vector search often fails.
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=top_k,
            fields="content_vector",
        )
        results = client.search(
            search_text=query,
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="rag-semantic-config",
            top=top_k,
            filter=filters,
            include_total_count=True,
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
        )
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    search_results = []
    for r in results:
        metadata = {}
        raw_metadata = r.get("metadata")
        if raw_metadata:
            try:
                metadata = (
                    json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
                )
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        search_results.append(
            SearchResult(
                chunk_id=r.get("chunk_id", ""),
                document_id=r.get("document_id", ""),
                content=r.get("content", ""),
                score=float(r.get("@search.score", 0.0)),
                metadata=metadata,
                highlights=list(r.get("@search.captions", [])),
            )
        )

    logger.debug(
        "Search (%s) returned %d results in %.1fms",
        mode,
        len(search_results),
        elapsed_ms,
    )
    return search_results


def delete_document_chunks(document_id: str, index_name: str | None = None) -> int:
    """
    Delete all chunks belonging to a document.

    Used when re-ingesting a document or cleaning up test data.

    Args:
        document_id: The document ID to delete.
        index_name: Optional override for the index name.

    Returns:
        Number of chunks deleted.
    """
    client = _get_search_client(index_name)

    # First, find all chunks for this document
    results = client.search(
        search_text="*",
        filter=f"document_id eq '{document_id}'",
        select=["id"],
        top=1000,
    )
    ids_to_delete = [{"id": r["id"]} for r in results]

    if not ids_to_delete:
        return 0

    result = client.delete_documents(documents=ids_to_delete)
    deleted = sum(1 for r in result if r.succeeded)
    logger.info("Deleted %d chunks for document '%s'", deleted, document_id)
    return deleted
