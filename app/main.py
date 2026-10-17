"""
FastAPI application entrypoint — RAG pipeline API with SSE streaming.

Architecture decision: Application factory pattern over module-level app.

Why: The factory pattern (create_app()) allows us to create a fresh app
instance for testing with different configuration, run multiple instances
with different settings, and avoid circular imports when registering
startup/shutdown handlers.

Architecture decision: SSE (Server-Sent Events) over WebSocket for streaming.

Why: SSE is simpler — it is HTTP-based, works with standard load balancers
and reverse proxies (no sticky sessions required), auto-reconnects on
connection loss, and is supported by the EventSource API in browsers.
WebSocket is bidirectional, which the RAG pipeline does not need — the
client sends one request and receives a stream of tokens. SSE is the
correct tool for unidirectional server-to-client streaming.

Architecture decision: sse-starlette over raw StreamingResponse.

Why: sse-starlette handles the SSE protocol formatting (event types, IDs,
retry intervals, data framing) that raw StreamingResponse would require
us to implement manually. It also manages connection lifecycle (clean
disconnect when the client closes).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Application lifecycle — startup and shutdown hooks
# ------------------------------------------------------------------
# Architecture decision: Lifespan context manager over @app.on_event.
# Why: The @app.on_event("startup") and @app.on_event("shutdown")
# decorators are deprecated in FastAPI 0.115+. The lifespan context
# manager is the modern pattern and guarantees cleanup even if the
# startup handler raises an exception.


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — runs once at startup and shutdown."""
    # Startup: log configuration summary
    logger.info(
        "Starting %s in %s mode",
        settings.APP_NAME,
        settings.APP_ENV,
    )
    logger.info(
        "Azure OpenAI: %s (chat: %s, embeddings: %s)",
        settings.AZURE_OPENAI_ENDPOINT or "not configured",
        settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
    logger.info(
        "Azure AI Search: %s (index: %s)",
        settings.AZURE_SEARCH_ENDPOINT or "not configured",
        settings.AZURE_SEARCH_INDEX_NAME,
    )

    yield

    # Shutdown: log termination
    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title="Azure AI RAG Pipeline",
        description=(
            "Production-grade RAG pipeline using Azure AI services. "
            "Ingest financial documents, chunk with multiple strategies, "
            "embed with Azure OpenAI, index in Azure AI Search, and query "
            "with hybrid retrieval + LLM generation. Supports SSE streaming "
            "for real-time token-by-token responses."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    # CORS — permissive in development, strict in production
    origins = settings.APP_CORS_ORIGINS.split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add X-Process-Time header to all responses for observability."""
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        return response

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    app.include_router(api_router, prefix="/api/v1")

    # Root health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint — used by K8s liveness and readiness probes."""
        from app.models.schemas import HealthCheck

        # Check Azure AI Search connectivity
        search_status = "unknown"
        try:
            if settings.AZURE_SEARCH_ENDPOINT:
                from azure.core.credentials import AzureKeyCredential
                from azure.search.documents import SearchClient

                client = SearchClient(
                    endpoint=settings.AZURE_SEARCH_ENDPOINT,
                    index_name=settings.AZURE_SEARCH_INDEX_NAME,
                    credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
                )
                # A simple get_document_count validates connectivity
                client.get_document_count()
                search_status = "connected"
        except Exception as e:
            search_status = f"error: {str(e)[:60]}"

        return HealthCheck(
            status="healthy",
            services={
                "search": search_status,
                "openai": "configured" if settings.AZURE_OPENAI_ENDPOINT else "not configured",
            },
        )

    return app


app = create_app()
