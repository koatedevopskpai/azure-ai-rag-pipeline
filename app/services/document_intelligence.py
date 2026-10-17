"""
Azure Document Intelligence service — extracts text from PDFs, images, and Office docs.

Architecture decision: Azure Document Intelligence over PyMuPDF + Tesseract OCR.

Why: Document Intelligence (formerly Form Recognizer) provides prebuilt OCR
for 300+ languages, table extraction, key-value pair extraction, and does
not require us to manage Tesseract or other OCR engines. For a production
financial research pipeline, Document Intelligence's layout model extracts
text with structural context (headings, paragraphs, tables) that raw PDF
parsing cannot capture.

Architecture decision: Lazy client initialisation over module-level singleton.

Why: Creating the DocumentClient requires a network call to discover the
service endpoint. Initialising at module import time would crash if the
.env file is missing. Lazy init means the import succeeds, and the first
call to extract_text() raises a clear error telling the developer which
env var is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import IO

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, ContentFormat
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from app.config import settings

logger = logging.getLogger(__name__)

_client: DocumentIntelligenceClient | None = None


def _get_client() -> DocumentIntelligenceClient:
    """
    Lazy-initialised Document Intelligence client.

    Raises ValueError if endpoint or API key is not configured, with a
    clear message telling the developer which env var to set.
    """
    global _client
    if _client is not None:
        return _client

    if not settings.AZURE_DOC_INTELLIGENCE_ENDPOINT:
        raise ValueError(
            "AZURE_DOC_INTELLIGENCE_ENDPOINT is not set. "
            "Add it to your .env file or set the environment variable."
        )
    if not settings.AZURE_DOC_INTELLIGENCE_API_KEY:
        raise ValueError(
            "AZURE_DOC_INTELLIGENCE_API_KEY is not set. "
            "Add it to your .env file or set the environment variable."
        )

    _client = DocumentIntelligenceClient(
        endpoint=settings.AZURE_DOC_INTELLIGENCE_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_DOC_INTELLIGENCE_API_KEY),
    )
    logger.info("Document Intelligence client initialised")
    return _client


def extract_text(file_path: str | Path | IO[bytes]) -> str:
    """
    Extract text from a document using Azure Document Intelligence.

    Uses the prebuilt layout model which returns text with structural
    context (headings, paragraphs, tables, footnotes). The output is
    markdown-formatted for downstream chunking.

    Args:
        file_path: Path to the document or a file-like object.

    Returns:
        Extracted text in markdown format.

    Raises:
        ValueError: If the client is not configured.
        HttpResponseError: If the Document Intelligence API returns an error.
    """
    client = _get_client()

    try:
        # Architecture decision: Prebuilt layout model over read model.
        # Why: The layout model preserves document structure (headings,
        # tables, lists) as markdown. The read model returns flat text
        # with no structure, which degrades RAG quality because the LLM
        # cannot distinguish a heading from a footnote.

        with open(file_path, "rb") as f:
            poller = client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=f,
                output_content_format=ContentFormat.MARKDOWN,
            )
        result: AnalyzeResult = poller.result()

        if not result.content:
            logger.warning("Document Intelligence returned empty content")
            return ""

        logger.info(
            "Extracted %d chars from document (pages: %s)",
            len(result.content),
            result.pages[0].page_number if result.pages else "unknown",
        )
        return result.content

    except HttpResponseError as e:
        logger.error("Document Intelligence API error: %s", e)
        raise
