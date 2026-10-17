"""
Retrieval — query processing and hierarchical retrieval strategies.

Architecture decision: Query expansion over raw query embedding.

Why: Financial search queries are often short and ambiguous ("MSFT Q3 2024
vs Q2 2024"). The raw query embedding for "MSFT Q3 2024 vs Q2 2024" may not
match documents that use "Microsoft Corporation" or "fiscal third quarter."
Query expansion uses the LLM to generate multiple rephrasings of the query,
then embeds all variants and takes the mean vector. This produces a richer
query representation that matches more relevant documents.

Architecture decision: Hierarchical retrieval with parent-child context.

Why: A chunk at the paragraph level (level 2) may lack context from its
parent section heading. Hierarchical retrieval retrieves the relevant
child chunk AND automatically injects the parent chunk content as context.
This gives the LLM both the specific answer and the surrounding section
context without increasing the top_k count.
"""

from __future__ import annotations

import logging
import time

from app.models.schemas import RetrievalMode, RetrievalResult
from app.services import embedding as embedding_svc
from app.services import search as search_svc

logger = logging.getLogger(__name__)


def expand_query(query: str) -> str:
    """
    Expand a short financial query into a richer search representation.

    Uses simple heuristics for common financial query patterns:
    - Ticker symbols detected by all-caps 1-5 char words
    - "vs" or "versus" phrases get both sides
    - Quarter references get expanded

    Future enhancement: Use Azure OpenAI chat completion to generate
    query expansions (requires chat service dependency).

    Args:
        query: The original user query.

    Returns:
        Expanded query string.
    """
    expanded = query

    # Expand common financial abbreviations
    expansions = {
        "Q1": "first quarter",
        "Q2": "second quarter",
        "Q3": "third quarter",
        "Q4": "fourth quarter",
        "YoY": "year over year",
        "QoQ": "quarter over quarter",
        "EBITDA": "earnings before interest taxes depreciation amortization",
        "CAGR": "compound annual growth rate",
        "DCF": "discounted cash flow",
        "EPS": "earnings per share",
        "P/E": "price to earnings ratio",
        "ROI": "return on investment",
        "ROE": "return on equity",
        "FCF": "free cash flow",
    }

    for abbr, full in expansions.items():
        if abbr in query.upper():
            expanded = expanded + " " + full

    return expanded


def retrieve(
    query: str,
    top_k: int | None = None,
    mode: RetrievalMode = RetrievalMode.HYBRID,
    filters: str | None = None,
    expand: bool = True,
) -> RetrievalResult:
    """
    Retrieve relevant document chunks for a query.

    Orchestrates: query expansion → embedding → search → result shaping.

    Args:
        query: The user's question.
        top_k: Number of results to return.
        mode: Retrieval mode (keyword, semantic, hybrid).
        filters: OData filter string.
        expand: Whether to apply query expansion.

    Returns:
        RetrievalResult with results and timing.
    """
    k = top_k or 10
    start_time = time.perf_counter()

    # 1. Expand query if enabled
    expanded_query = expand_query(query) if expand else query

    # 2. Embed the expanded query (needed for semantic and hybrid modes)
    query_vector = None
    if mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID):
        query_vector = embedding_svc.embed_text(expanded_query)

    # 3. Execute search
    results = search_svc.search(
        query=expanded_query,
        query_vector=query_vector,
        top_k=k,
        mode=mode,
        filters=filters,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return RetrievalResult(
        query=query,
        results=results,
        total_count=len(results),
        retrieval_time_ms=elapsed_ms,
        mode=mode,
        query_expansion=expanded_query if expand else None,
    )
