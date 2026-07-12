# ADR 003: Explicit RAG Pipeline over LangChain / LlamaIndex

## Status

Accepted (2026-07-13)

## Context

The RAG pipeline needs to orchestrate document retrieval, context injection, and LLM generation. Three approaches:

1. **Explicit pipeline**: Custom code in `app/core/rag.py` that calls retrieval, formats context, and calls the LLM
2. **LangChain**: `langchain.chains.RetrievalQA` or `langchain.chains.ConversationalRetrievalChain`
3. **LlamaIndex**: `llama_index.QueryEngine` with vector store index

## Decision

Use an explicit RAG pipeline with custom orchestration code, no framework dependency.

## Consequences

### Positive
- **Demonstrates understanding**: Each RAG step (query expansion, embedding, retrieval, re-ranking, context formatting, generation) is implemented in explicit code that can be read, debugged, and optimised independently
- **No framework lock-in**: If LangChain releases a breaking change (which happens frequently), the explicit pipeline is unaffected
- **Testable in isolation**: Each step has a focused test (test_chunking.py, test_retrieval.py) that does not require mocking a framework's internal state
- **Smaller dependency footprint**: No langchain, langchain-community, langchain-azure-openai, or llama-index packages. The requirements.txt is leaner and has fewer vulnerability vectors

### Negative
- **More code**: ~200 lines of orchestration vs a 3-line LangChain chain
- **Missing framework features**: No built-in support for chain-of-thought prompting, self-querying retrievers, or query decomposition
- **No community plugins**: LangChain has 500+ integrations with document loaders, output parsers, and memory stores

### Mitigations
- The explicit pipeline is designed as a bridge to LangChain, not a rejection of it. Each service function has the same signature as its LangChain equivalent:
  - `embedding.embed_texts(texts)` ↔ `AzureOpenAIEmbeddings.embed_documents(texts)`
  - `search.search(query, top_k)` ↔ `AzureAISearchRetriever.get_relevant_documents(query)`
- If the project needs LangChain features later, the wrapper functions connect seamlessly — only the orchestration in `app/core/rag.py` changes
- The agentic workflow module (`agent.py`) uses Azure OpenAI function calling directly, which is the native pattern — LangChain's tool calling abstraction (bind_tools) adds indirection without value

## Alternatives Considered

### LangChain RetrievalQA
- **Pros**: 15 lines of code for a complete RAG chain, built-in support for Azure AI Search as a retriever, extensive community
- **Cons**: Version mismatches (langchain 0.1 vs 0.2 vs 0.3 have different APIs), black-box abstraction hides RAG internals, difficult to debug when retrieval quality is poor
- **Rejected because**: The Orbis role requires demonstrating understanding of RAG architecture, not framework proficiency. An explicit pipeline proves we understand retrieval, context formatting, and generation as separate concerns

### LlamaIndex
- **Pros**: Excellent document ingestion pipeline (loaders, parsers, node parsers), built-in support for hierarchical retrieval
- **Cons**: Heavy dependency tree (llama-index-core + 20+ plugins), steep learning curve for customisation, tightly couples the retrieval format to the indexing strategy
- **Rejected because**: LlamaIndex's document ingestion (loaders → nodes → index) would replace our explicit chunking service. For the portfolio project, chunking strategy is a key architectural decision we want to demonstrate — not hide behind a framework
