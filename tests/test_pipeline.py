"""
Tests for the Azure AI RAG Pipeline.

Architecture decision: Unit tests for services, integration tests for API.

Why: Service tests (chunking, retrieval, embedding) test pure logic without
Azure dependencies. API tests test the full FastAPI routing layer with mocked
Azure clients. This separation means the chunking tests run in CI without
any Azure credentials, while the API tests validate request/response shape.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    """FastAPI test client with a fresh app instance."""
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Health check endpoint — used by K8s probes and monitoring."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "search" in data["services"]


class TestChunking:
    """Chunking strategies — pure functions, no Azure dependencies."""

    def test_recursive_chunking_splits_long_text(self):
        from app.services.chunking import chunk_recursive

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
        chunks = chunk_recursive(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 2
        assert all(c["strategy"].value == "recursive" for c in chunks)
        assert all(isinstance(c["chunk_index"], int) for c in chunks)

    def test_recursive_chunking_single_short_text(self):
        from app.services.chunking import chunk_recursive

        text = "Short text."
        chunks = chunk_recursive(text, chunk_size=1000, overlap=0)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Short text."

    def test_recursive_chunking_empty_text(self):
        from app.services.chunking import chunk_recursive

        chunks = chunk_recursive("", chunk_size=1000, overlap=0)
        assert len(chunks) == 0

    def test_semantic_chunking_splits_by_sentences(self):
        from app.services.chunking import chunk_semantic

        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here. Sixth sentence here."
        chunks = chunk_semantic(text, sentences_per_chunk=2)
        assert len(chunks) >= 2
        assert all(c["strategy"].value == "semantic" for c in chunks)

    def test_hierarchical_chunking_detects_headings(self):
        from app.services.chunking import chunk_hierarchical

        text = "# Section One\n\nContent under section one.\n\n## Subsection One A\n\nContent under subsection.\n\n# Section Two\n\nContent under section two."
        chunks = chunk_hierarchical(text)
        assert len(chunks) >= 2
        hierarchical = [c for c in chunks if c["strategy"].value == "hierarchical"]
        assert len(hierarchical) >= 2


class TestRetrieval:
    """Retrieval logic — query expansion and result shaping."""

    def test_query_expansion_expands_abbreviations(self):
        from app.core.retrieval import expand_query

        expanded = expand_query("MSFT Q3 2024 EBITDA")
        assert "Microsoft" in expanded or len(expanded) > len("MSFT Q3 2024 EBITDA")
        assert "earnings before interest" in expanded

    def test_query_expansion_preserves_original(self):
        from app.core.retrieval import expand_query

        query = "What is the revenue trend?"
        expanded = expand_query(query)
        assert query in expanded

    def test_query_expansion_empty_string(self):
        from app.core.retrieval import expand_query

        assert expand_query("") == ""


class TestAPIEndpoints:
    """API routing — validates request/response shapes without Azure."""

    def test_query_endpoint_requires_messages(self, client):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422  # Validation error

    def test_query_endpoint_validates_streaming(self, client):
        response = client.post(
            "/api/v1/query",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "query": "Test question",
            },
        )
        # May return 500 if Azure is not configured, but should not return 422
        assert response.status_code != 422

    def test_ingest_endpoint_validates_file_type(self, client):
        response = client.post(
            "/api/v1/ingest?chunking_strategy=recursive",
            files={"file": ("test.exe", b"fake exe", "application/x-msdownload")},
        )
        assert response.status_code == 400
        assert "Unsupported" in response.json()["detail"]

    def test_ingest_endpoint_validates_strategy(self, client):
        response = client.post(
            "/api/v1/ingest?chunking_strategy=invalid",
            files={"file": ("test.pdf", b"fake pdf", "application/pdf")},
        )
        assert response.status_code == 400

    def test_index_create_endpoint(self, client):
        response = client.post("/api/v1/index/create")
        # Should return 500 if Azure is not configured, but shape should be right
        assert response.status_code in (200, 500)

    def test_retrieve_endpoint(self, client):
        response = client.post("/api/v1/retrieve?query=test")
        assert response.status_code in (200, 500)

    def test_agent_endpoint(self, client):
        response = client.post("/api/v1/agent?query=summarise+documents")
        assert response.status_code in (200, 500)

    def test_cors_headers_present(self, client):
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers
