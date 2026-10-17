"""
Azure OpenAI embedding service — converts text chunks to vector embeddings.

Architecture decision: Azure OpenAI embeddings over OpenAI.com API.

Why: The Orbis role explicitly requires Azure AI services. Azure OpenAI
provides the same models (text-embedding-3-large) with enterprise security:
private networking via VNET, managed identity authentication, data residency
controls, and no data sent to OpenAI for training. For financial services
compliance, Azure OpenAI is the only viable option.

Architecture decision: text-embedding-3-large over text-embedding-3-small.

Why: RAG quality is highly sensitive to embedding quality. The large model
(3072 dimensions) captures more semantic nuance than the small model (1536
dimensions). For a financial research RAG pipeline where retrieval accuracy
directly impacts the quality of investment insights, the additional cost
per token is justified.
"""

from __future__ import annotations

import logging

import numpy as np
from openai import AzureOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    """Lazy-initialised Azure OpenAI client."""
    global _client
    if _client is not None:
        return _client

    if not settings.AZURE_OPENAI_ENDPOINT:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT is not set. "
            "Add it to your .env file or set the environment variable."
        )

    _client = AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY or None,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )
    logger.info("Azure OpenAI client initialised (endpoint: %s)", settings.AZURE_OPENAI_ENDPOINT)
    return _client


def embed_text(text: str, model: str | None = None) -> list[float]:
    """
    Generate an embedding vector for a single text string.

    Architecture decision: reduce_dimensions=False (full 3072 dimensions).

    Why: text-embedding-3-large supports dimension reduction via
    MATRYoshka, but reducing dimensions loses information. For a
    production RAG pipeline, the full vector captures maximum semantic
    information. Dimension reduction only makes sense when you have
    millions of vectors and need to optimise storage costs.

    Args:
        text: The text to embed.
        model: Optional override for the deployment name.

    Returns:
        A list of floats (the embedding vector).

    Raises:
        ValueError: If Azure OpenAI is not configured.
    """
    client = _get_client()
    deployment = model or settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    response = client.embeddings.create(
        input=text,
        model=deployment,
    )

    vector = response.data[0].embedding
    logger.debug(
        "Generated embedding with %d dimensions using %s",
        len(vector),
        deployment,
    )
    return vector


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """
    Generate embedding vectors for multiple texts in a single API call.

    Architecture decision: Batched API call over sequential calls.

    Why: Azure OpenAI supports embedding multiple inputs in one request.
    This is significantly faster than calling embed_text() in a loop
    because it reduces HTTP round-trips and the server processes batches
    in parallel internally.

    Args:
        texts: List of texts to embed.
        model: Optional override for the deployment name.

    Returns:
        List of embedding vectors, one per input text.
    """
    if not texts:
        return []

    client = _get_client()
    deployment = model or settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    response = client.embeddings.create(
        input=texts,
        model=deployment,
    )

    # Sort by index to maintain input order
    sorted_data = sorted(response.data, key=lambda x: x.index)
    vectors = [item.embedding for item in sorted_data]

    logger.debug("Generated %d embeddings using %s", len(vectors), deployment)
    return vectors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Used for re-ranking and for the semantic chunking strategy.
    """
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
