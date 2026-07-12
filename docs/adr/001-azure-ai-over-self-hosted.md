# ADR 001: Azure AI Services over Self-Hosted Open-Source Stack

## Status

Accepted (2026-07-13)

## Context

The RAG pipeline needs LLM inference, text embedding, vector search, and document extraction. The two broad options are:

1. **Azure AI Services** (managed): Azure OpenAI, Azure AI Search, Azure Document Intelligence
2. **Self-hosted open-source**: Ollama/vLLM (LLM), ChromaDB/Qdrant (vector store), Tesseract (OCR), PyMuPDF (PDF parsing)

The Orbis Group AI Engineer role explicitly requires "hands-on experience with Azure AI services including Azure OpenAI, AI Search, AI Foundry and Document Intelligence." Beyond the portfolio alignment, Azure AI services offer enterprise features (private networking, managed identity, RBAC, data residency) that self-hosted alternatives require significant operational effort to replicate.

## Decision

Use Azure AI Services for all three components:
- **Azure OpenAI** (gpt-4o, text-embedding-3-large) for chat and embeddings
- **Azure AI Search** (with semantic reranker) for vector + keyword hybrid search
- **Azure Document Intelligence** (prebuilt-layout model) for document extraction

## Consequences

### Positive
- **Enterprise compliance**: Azure AI services meet financial services compliance requirements (SOC 2, ISO 27001, data residency controls) that a self-hosted stack cannot guarantee
- **Zero infrastructure management**: No GPUs to provision, no vector store to scale, no OCR engine to maintain
- **Managed scaling**: Azure OpenAI handles request throttling and capacity management; AI Search handles index sharding and replication
- **Direct portfolio relevance**: This is exactly the stack the Orbis role uses

### Negative
- **Vendor lock-in**: The pipeline uses Azure-specific SDKs and service endpoints. Switching to AWS Bedrock or GCP Vertex AI would require rewriting the service layer
- **Cost at scale**: Managed services cost more than self-hosted alternatives at high throughput (though the portfolio project stays well within free tier limits)
- **Cold start latency**: Azure OpenAI and Document Intelligence have 1-3 second cold starts for infrequent requests

### Mitigations
- The service layer abstracts Azure SDK details behind `app/services/*.py` interfaces. Switching cloud providers requires implementing a new service module but does not change the API or core RAG logic
- All services are stateless and independently scalable — each can be replaced without affecting the others

## Alternatives Considered

### Self-hosted Ollama + ChromaDB + Tesseract
- **Pros**: Free, private, works offline, no API key management
- **Cons**: Requires GPU-equipped AKS nodes (expensive), no semantic reranker, poor OCR quality for complex financial documents, does not demonstrate Azure skills
- **Rejected because**: The portfolio must demonstrate Azure AI expertise. A self-hosted stack proves DevOps skills but misses the primary role requirement

### GCP Vertex AI + Cloud Search + Document AI
- **Pros**: Equally capable managed AI stack, George has GCP experience (gcp-data-pipeline)
- **Cons**: Vertex AI Search lacks the mature hybrid search + semantic reranking of Azure AI Search; Document AI has fewer prebuilt models than Document Intelligence
- **Rejected because**: The Orbis role is Azure-specific. GCP experience is transferable but not directly applicable
