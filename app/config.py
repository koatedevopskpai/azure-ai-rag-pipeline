"""
Application configuration — Azure AI services and runtime settings.

Architecture decision: pydantic-settings over os.environ or django-environ.

Why: pydantic-settings validates types at import time, provides IDE
autocompletion via the Settings dataclass, supports .env files out of
the box, and raises clear ValidationError messages when a required
variable is missing. os.environ.get() silently returns None for missing
keys, which causes confusing AttributeError deep in the call stack.

Architecture decision: Lazy validation over strict validation on import.

Why: This module should not crash at import time when .env is missing.
Instead, services validate their own config on first use and raise a
clear error message that tells the developer exactly which env var is
missing. This makes local development (without Azure) possible for
testing non-Azure code paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application configuration, loaded from environment variables.

    Azure SDKs use DefaultAzureCredential for authentication in production
    (managed identity on AKS) and .env file values for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = "azure-ai-rag-pipeline"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_LOG_LEVEL: str = "INFO"
    APP_MAX_UPLOAD_SIZE_MB: int = 50
    APP_CORS_ORIGINS: str = "*"

    # ------------------------------------------------------------------
    # Azure OpenAI
    # ------------------------------------------------------------------
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-large"

    # ------------------------------------------------------------------
    # Azure AI Search
    # ------------------------------------------------------------------
    AZURE_SEARCH_ENDPOINT: str = ""
    AZURE_SEARCH_API_KEY: str = ""
    AZURE_SEARCH_INDEX_NAME: str = "rag-documents"

    # ------------------------------------------------------------------
    # Azure Document Intelligence
    # ------------------------------------------------------------------
    AZURE_DOC_INTELLIGENCE_ENDPOINT: str = ""
    AZURE_DOC_INTELLIGENCE_API_KEY: str = ""

    # ------------------------------------------------------------------
    # Chunking defaults
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNKING_STRATEGY: Literal["recursive", "semantic", "hierarchical"] = "recursive"

    # ------------------------------------------------------------------
    # Retrieval defaults
    # ------------------------------------------------------------------
    RETRIEVAL_TOP_K: int = 10
    RETRIEVAL_USE_SEMANTIC: bool = True
    RETRIEVAL_MIN_SCORE: float = 0.0

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    OTEL_SERVICE_NAME: str = "azure-ai-rag-pipeline"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"


settings = Settings()

# ------------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
