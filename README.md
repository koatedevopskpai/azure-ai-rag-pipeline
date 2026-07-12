# Azure AI RAG Pipeline

> Production-grade financial research RAG (Retrieval-Augmented Generation) pipeline using Azure AI services. Ingest financial documents, chunk with multiple strategies, embed with Azure OpenAI, index in Azure AI Search with hybrid retrieval, and query with streaming SSE responses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client (UI / API Consumer)                   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTPS / SSE
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (app/)                          │
│                                                                      │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────────────┐  │
│  │  /api/v1     │  │  /api/v1/query  │  │  /api/v1/query/stream  │  │
│  │  /ingest     │  │  (non-stream)   │  │  (SSE stream)          │  │
│  └──────┬───────┘  └───────┬─────────┘  └───────────┬────────────┘  │
│         │                  │                         │               │
│         ▼                  ▼                         ▼               │
│  ┌──────────┐     ┌────────────────┐      ┌──────────────────────┐  │
│  │Ingestion │     │  RAG Pipeline  │      │    SSE EventSource   │  │
│  │ Pipeline │     │  (core/rag.py) │      │  (core/rag.py stream)│  │
│  └────┬─────┘     └───────┬────────┘      └──────────┬───────────┘  │
│       │                   │                           │              │
│       ▼                   ▼                           ▼              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Service Layer (app/services/)              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │   │
│  │  │Document  │  │Chunking  │  │Embedding │  │Azure AI     │  │   │
│  │  │Intelligence│  │(3 strats)│  │(Azure    │  │Search       │  │   │
│  │  │          │  │          │  │OpenAI)   │  │(hybrid)     │  │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘  │   │
│  └───────┼──────────────┼────────────┼─────────────────┼────────┘   │
└──────────┼──────────────┼────────────┼─────────────────┼────────────┘
           │              │            │                 │
           ▼              ▼            ▼                 ▼
    ┌──────────┐   ┌────────────┐  ┌──────────┐  ┌──────────────┐
    │Document  │   │  Chunk     │  │ Azure    │  │  Azure AI    │
    │Intelligence│   │  Store     │  │ OpenAI   │  │  Search      │
    │(OCR+     │   │  (memory)  │  │ text-    │  │  (vector +   │
    │ Layout)  │   │            │  │embedding │  │  keyword)    │
    └──────────┘   └────────────┘  │ 3-large  │  └──────────────┘
                                   └──────────┘
```

---

## Key Features

| Feature | Implementation | Why It Matters |
|---------|---------------|----------------|
| **Document ingestion** | Azure Document Intelligence (prebuilt-layout model) extracts text with structural context (headings, tables, lists) as markdown | Financial documents (PDF reports, filings) preserve their structure for better RAG quality |
| **3 chunking strategies** | Recursive (boundary-aware), Semantic (sentence-grouping), Hierarchical (section/subsection/paragraph) | Different document types need different chunk boundaries — one size does not fit all |
| **Hybrid search** | Azure AI Search with vector (semantic) + keyword (BM25) + semantic reranker | Ticker symbols ("MSFT") and exact phrases ("Q3 2024 EBITDA") require keyword matching that pure vector search misses |
| **Query expansion** | Automatic expansion of financial abbreviations (YoY, EBITDA, CAGR) + ticker detection | Short financial queries become richer search representations without user effort |
| **Agentic workflow** | Azure OpenAI function calling with search, extract_metrics, and compare_documents tools | Multi-step analysis tasks (compare two earnings reports) without hardcoding workflows |
| **SSE streaming** | Server-Sent Events for real-time token-by-token response generation | Users see the answer forming in real time instead of waiting for the full response |
| **Distroless Docker** | Multi-stage build with `gcr.io/distroless/python3-debian12` runtime | ~120MB image, zero shell, minimal attack surface — enterprise security standard |
| **K8s deployment** | Deployment + Service + Ingress manifests with readiness/liveness probes, resource limits | One-command deploy to any AKS cluster |

---

## Prerequisites

- Python 3.12+
- An Azure subscription with the following services enabled:
  - Azure OpenAI (gpt-4o + text-embedding-3-large deployments)
  - Azure AI Search (Basic tier or higher with semantic search)
  - Azure Document Intelligence (S0 tier)
- Terraform 1.6+ (for infrastructure deployment)
- Docker (for local container build)
- kubectl + AKS cluster (for K8s deployment)

---

## Quick Start (Local Development)

```bash
# 1. Clone
git clone https://github.com/koate-kpai/azure-ai-rag-pipeline.git
cd azure-ai-rag-pipeline

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# 3. Install
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your Azure service endpoints and keys

# 5. Create the search index
curl -X POST http://localhost:8000/api/v1/index/create

# 6. Run
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API documentation (Swagger UI).

---

## API Reference

### `POST /api/v1/ingest` — Ingest a document

Upload a financial document for extraction, chunking, embedding, and indexing.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@annual-report-2024.pdf" \
  -F "chunking_strategy=hierarchical"
```

### `POST /api/v1/query` — Ask a question (non-streaming)

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Analyse the document"}],
    "query": "What was Microsofts Q3 2024 revenue?",
    "retrieval_mode": "hybrid",
    "top_k": 10
  }'
```

### `POST /api/v1/query/stream` — Ask a question (SSE streaming)

```bash
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Analyse the document"}],
    "query": "Summarise the key financial metrics",
    "stream": true
  }'
```

### `POST /api/v1/agent` — Run agentic workflow

```bash
curl -X POST "http://localhost:8000/api/v1/agent?query=Compare+Q2+and+Q3+2024+revenue"
```

### Full Endpoint Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (K8s probes) |
| `/docs` | GET | Swagger UI |
| `/api/v1/ingest` | POST | Upload and ingest a document |
| `/api/v1/query` | POST | RAG query (non-streaming) |
| `/api/v1/query/stream` | POST | RAG query (SSE streaming) |
| `/api/v1/retrieve` | POST | Retrieval only (no generation) |
| `/api/v1/agent` | POST | Agentic workflow (function calling) |
| `/api/v1/index/create` | POST | Create or update the search index |

---

## Chunking Strategies

| Strategy | Best For | How It Works |
|----------|----------|-------------|
| **Recursive** | Structured text with clear paragraph boundaries | Tries double-newline → single-newline → sentence-end → character count |
| **Semantic** | Dense analytical content | Splits into sentences, groups by sentence count (default: 5 sentences per chunk) |
| **Hierarchical** | Long structured documents (annual reports, filings) | Three levels: section (h1/h2) → subsection (h3/h4) → paragraph; stores parent-child relationships |

Choose via the `chunking_strategy` parameter on the ingest endpoint.

---

## Project Structure

```
azure-ai-rag-pipeline/
├── .github/workflows/     # CI pipeline (lint + test + Docker build)
├── app/
│   ├── api/               # FastAPI routes
│   │   └── routes.py      #   REST endpoints + SSE streaming
│   ├── core/              # RAG orchestration
│   │   ├── agent.py       #   Agentic workflow (function calling)
│   │   ├── rag.py         #   RAG pipeline (retrieve → generate)
│   │   └── retrieval.py   #   Query expansion + retrieval
│   ├── models/            # Pydantic v2 schemas
│   │   └── schemas.py     #   15+ request/response models
│   ├── services/          # Azure AI service integrations
│   │   ├── chat.py        #   Azure OpenAI chat completions
│   │   ├── chunking.py    #   3 chunking strategies
│   │   ├── document_intelligence.py  # Azure Doc Intelligence
│   │   ├── embedding.py   #   Azure OpenAI embeddings
│   │   ├── ingestion.py   #   Full extraction pipeline
│   │   └── search.py      #   Azure AI Search (hybrid)
│   ├── config.py          # pydantic-settings configuration
│   └── main.py            # FastAPI app factory + lifespan
├── docs/adr/              # Architecture Decision Records
│   ├── 001-azure-ai-over-self-hosted.md
│   ├── 002-sse-over-websocket.md
│   └── 003-explicit-rag-over-langchain.md
├── infra/                 # Terraform (Azure resources)
│   ├── main.tf            #   Resource definitions
│   ├── variables.tf       #   Configurable variables
│   ├── outputs.tf         #   Connection details
│   └── dev.tfvars         #   Dev environment values
├── k8s/                   # Kubernetes manifests
│   ├── deployment.yaml    #   Deployment + Service
│   └── ingress.yaml       #   NGINX Ingress with SSE support
├── tests/                 # Pytest test suite
├── .env.example           # Environment variable template
├── Dockerfile             # Multi-stage distroless build
├── pytest.ini
└── requirements.txt
```

---

## Architecture Decision Records

All significant architectural decisions are documented in `docs/adr/`:

| ADR | Decision | Key Rationale |
|-----|----------|---------------|
| 001 | Azure AI over self-hosted | Portfolio must demonstrate Azure; managed services are enterprise-compliant |
| 002 | SSE over WebSocket | Unidirectional streaming; simpler infra (no sticky sessions); auto-reconnect |
| 003 | Explicit RAG over LangChain | Demonstrates understanding; no framework lock-in; testable in isolation |

---

## Deployment

### Terraform (Azure Resources)

```bash
cd infra
terraform init
terraform plan -var-file=dev.tfvars
terraform apply -var-file=dev.tfvars
terraform output -json > ../.env.generated
```

### Docker

```bash
docker build --tag azure-ai-rag-pipeline .
docker run --rm -it -p 8000:8000 --env-file .env azure-ai-rag-pipeline
```

### Kubernetes (AKS)

```bash
# Create the secret from your .env file
kubectl create secret generic azure-ai-rag-secrets \
  --from-literal=openai-endpoint=$AZURE_OPENAI_ENDPOINT \
  --from-literal=openai-api-key=$AZURE_OPENAI_API_KEY \
  --from-literal=search-endpoint=$AZURE_SEARCH_ENDPOINT \
  --from-literal=search-api-key=$AZURE_SEARCH_API_KEY \
  --from-literal=doc-intelligence-endpoint=$AZURE_DOC_INTELLIGENCE_ENDPOINT \
  --from-literal=doc-intelligence-api-key=$AZURE_DOC_INTELLIGENCE_API_KEY

# Deploy
kubectl apply -f k8s/
```

---

## Gap Closure Map

| Orbis Role Requirement | Portfolio Evidence |
|------------------------|-------------------|
| Python/FastAPI backend | FastAPI with async, Pydantic v2, dependency injection, middleware |
| WebSockets/SSE | SSE streaming via sse-starlette (app/api/routes.py:80-100) |
| Production LLM apps | RAG pipeline with structured prompts, source citation, guardrails |
| Agentic workflows | Azure OpenAI function calling with 3 tools (app/core/agent.py) |
| Advanced RAG | Hierarchical retrieval + query expansion + hybrid search + semantic reranker |
| Chunking strategies | 3 strategies with architectural rationale (app/services/chunking.py) |
| Azure OpenAI | Chat + embedding deployments with managed identity support |
| Azure AI Search | Hybrid search index with HNSW + semantic configuration |
| Azure AI Foundry | Terraform-provisioned infrastructure |
| Azure Document Intelligence | Prebuilt-layout model with markdown output |
| Docker + K8s | Distroless multi-stage build + AKS manifests with probes/limits/ingress |
| CI/CD | GitHub Actions: lint → test → build → push to GHCR |
| Systems design | 3 ADRs documenting every architectural decision |

---

## Related Projects

- [gcp-data-pipeline](https://github.com/koate-kpai/gcp-data-pipeline) — Serverless GCP-native ETL pipeline (Cloud Storage → Cloud Function → BigQuery)
- [data-pipeline-engine](https://github.com/koate-kpai/data-pipeline-engine) — Modular ETL framework with FastAPI serving
- [pipeline-dashboard](https://github.com/koate-kpai/pipeline-dashboard) — Django full-stack dashboard for pipeline analytics
- [rag-pipeline-backend](https://github.com/koate-kpai/rag-pipeline-backend) — Original RAG pipeline (predecessor to this Azure-native version)
- [ai-agent-engineer](https://github.com/koate-kpai/ai-agent-engineer) — Multi-agent AI platform with LangGraph
