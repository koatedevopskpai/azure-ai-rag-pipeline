# ADR 002: SSE Streaming over WebSocket for Token Generation

## Status

Accepted (2026-07-13)

## Context

The RAG pipeline streams generated tokens from Azure OpenAI to the client in real time. Two protocols are available:

1. **SSE (Server-Sent Events)**: Unidirectional HTTP-based streaming from server to client
2. **WebSocket**: Bidirectional persistent connection between client and server

The streaming use case is unidirectional: the client sends one POST request and receives a stream of response tokens. The client does not need to send additional data after the initial request.

## Decision

Use SSE streaming via `sse-starlette` for all token generation endpoints.

## Consequences

### Positive
- **HTTP-native**: SSE works with standard HTTP load balancers, reverse proxies, and ingress controllers. No sticky sessions or WebSocket-specific configuration required
- **Auto-reconnect**: The browser's `EventSource` API automatically reconnects on connection loss, which is important for long-running LLM generations
- **Simpler infrastructure**: AKS ingress (NGINX) handles SSE natively with `proxy-buffering: off` — no need for a separate WebSocket ingress or service mesh
- **Smaller client surface**: The JavaScript SSE consumer is ~10 lines of code vs ~80 lines for WebSocket with reconnection logic

### Negative
- **Unidirectional only**: If the client needed to send follow-up data mid-stream (e.g., "stop generation"), SSE would require a separate control channel
- **HTTP/1.1 connection limit**: Browsers limit concurrent SSE connections to 6 per domain. This is not a problem for a single RAG query but could limit dashboard widgets displaying multiple live streams
- **No binary framing**: SSE is text-only. Binary data (e.g., embeddings during retrieval) must be base64-encoded

### Mitigations
- The streaming endpoint is used one query at a time — the user asks a question, reads the streaming answer, then asks the next. Six concurrent streams is far beyond realistic usage
- Binary data is never streamed via SSE. Embeddings are computed server-side before streaming begins (the retrieve-then-generate pattern)

## Alternatives Considered

### WebSocket
- **Pros**: Bidirectional, binary framing, no connection limit, well-suited for multi-turn conversational agents
- **Cons**: Requires sticky sessions on load balancers, more complex client code, harder to debug (the WebSocket frame is opaque vs SSE which is readable text)
- **Rejected because**: The RAG query pattern is one request → stream of responses. Bidirectional communication is not needed. SSE is the simpler, correct tool

### Long-polling
- **Pros**: Works everywhere, no protocol support required
- **Cons**: High latency (each poll is a new HTTP request), wasteful (server may poll when no new tokens are available)
- **Rejected because**: Token-level streaming requires sub-second latency that long-polling cannot provide
