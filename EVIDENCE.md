# Evidence index — Azure AI RAG Pipeline

A reviewer's route through this repository. Every claim points at code, tests, or a
dated run recorded in [`evidence/`](evidence/). No claim is asserted without a location.

**Five-minute path:** this file → [what it is](#what-it-is) →
[the strongest measured result](#the-strongest-measured-result) →
[how to reproduce](#how-to-reproduce) →
[what this does not establish](#what-this-does-not-establish).

---

## What it is

A reference RAG application on Azure AI services: document ingestion, three chunking
strategies, query expansion, hybrid retrieval, an agentic tool-calling endpoint and SSE
streaming, behind a FastAPI service with a distroless container and Kubernetes manifests.

| | |
|---|---|
| Scope | **Public reference build.** Self-funded; no Azure services provisioned, no deployment |
| Author | Koate Kpai — https://github.com/koatedevopskpai |
| Entry point | `app/main.py`, `app/api/routes.py`, `app/core/` |
| Revision described | `d853713` |
| CI | **green** on `main`, 18 October 2026 |
| Evidence task | [`docs/V1.md`](docs/V1.md), [`evidence/V1.md`](evidence/V1.md) |

## My contribution

| Area | Source |
|---|---|
| FastAPI surface — ingest, query, streaming query, agent, index, retrieve, health | [`app/api/routes.py`](app/api/routes.py), [`app/main.py`](app/main.py) |
| Three chunking strategies with rationale | [`app/services/chunking.py`](app/services/chunking.py), [`docs/adr/`](docs/adr/) |
| Query expansion for financial abbreviations and tickers | [`app/core/retrieval.py`](app/core/retrieval.py) |
| Agentic function calling with three tools | [`app/core/agent.py`](app/core/agent.py) |
| Distroless multi-stage image; AKS manifests; Terraform | [`Dockerfile`](Dockerfile), [`k8s/`](k8s/), [`infra/`](infra/) |
| Unit suite and CI (lint → test → build/push) | [`tests/test_pipeline.py`](tests/test_pipeline.py), [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

Three ADRs record the architecture decisions:
[Azure AI over self-hosted](docs/adr/001-azure-ai-over-self-hosted.md),
[SSE over WebSocket](docs/adr/002-sse-over-websocket.md),
[explicit RAG over LangChain](docs/adr/003-explicit-rag-over-langchain.md).

---

## The strongest measured result

**17 unit tests, lint and format clean, image builds — verified on Linux in CI and
re-run locally.**

| Check | Linux CI (`main` @ `d853713`) | Windows local |
|---|---|---|
| Lint + format (`ruff`) | **success** | **clean** (ruff 0.17.1, 24 files) |
| `pytest -m "not integration"` | **17 passed in 6.29s** | **17 passed in 7.87s** |
| Image build | **success** (build + push on `main`) | not run locally |

Test breakdown: `TestChunking` 5, `TestAPIEndpoints` 8, `TestRetrieval` 3,
`TestHealthEndpoint` 1.

**What that result is worth.** It establishes that the chunkers behave as specified,
query expansion produces the expected terms, and the API validates its inputs and
returns the documented shapes. It establishes **nothing** about retrieval or answer
quality, because no test in this repository calls Azure.

### A claim that was corrected

The header previously described this as a "Production-grade" pipeline. Nothing here is
production anything — no service was provisioned, no deployment exists. It now reads
"reference pipeline", and the package docstring says the same. Correcting a claim
downward is part of the evidence trail.

---

## How to reproduce

Offline and free — no Azure account, no credentials:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -v --tb=short -m "not integration"
```

Lint, exactly as CI runs it:

```bash
pip install ruff
ruff check .
ruff format --check .
```

The `-m "not integration"` marker is **load-bearing**: it excludes every test that would
need live Azure. Running the full selection would not produce a green offline run.

**Billable and not part of this evidence:** deploying `infra/` Terraform, applying
`k8s/`, or calling any Azure endpoint. None of that is recorded here.

---

## One success trace, one failure

**Success.** `TestChunking` drives the real chunkers: a long document splits on
boundaries rather than mid-sentence, a short document yields a single chunk, empty input
yields none, and the hierarchical strategy detects headings and nests sections beneath
them. That is behavioural, not structural — the code path runs.

**Failure the API blocks.** `TestAPIEndpoints` asserts the service **rejects** bad input
rather than guessing: an unsupported file type, an unknown chunking strategy, a missing
`messages` field, and an invalid streaming flag all return 4xx. Rejecting input is the
correct outcome; a silently defaulted strategy would be worse.

**Failure that is not covered.** If Azure credentials are absent or wrong, these tests
never find out — the Azure clients are mocked. That failure mode is real and
**untested here**; it is why `docs/V1.md` lists live Azure behaviour as an open gap.

---

## Decisions and trade-offs

| Decision | Where |
|---|---|
| Azure AI services over self-hosted models | [`docs/adr/001`](docs/adr/001-azure-ai-over-self-hosted.md) |
| SSE over WebSocket for streaming | [`docs/adr/002`](docs/adr/002-sse-over-websocket.md) |
| Explicit RAG wiring over a LangChain abstraction | [`docs/adr/003`](docs/adr/003-explicit-rag-over-langchain.md) |
| Three chunking strategies rather than one configurable chunker | [`app/services/chunking.py`](app/services/chunking.py) |
| Separate Azure SDK packages over the meta-package, for image size | [`requirements.txt`](requirements.txt) |
| Distroless runtime | [`Dockerfile`](Dockerfile) |

---

## What this does not establish

- **Retrieval quality.** No retrieval metric of any kind — no recall, MRR or nDCG — is
  measured. Hybrid search and the semantic reranker are implemented, not evaluated.
- **Answer quality.** No generation was scored, by a model or a human.
- **Live Azure behaviour.** No Document Intelligence extraction, no Azure OpenAI call,
  no Search query, no live SSE stream. Every Azure client in the suite is mocked.
- **A deployment.** Neither the Terraform nor the Kubernetes manifests have been
  applied; there is no AKS cluster, no service and no uptime.
- **The image size claim.** The build succeeds; "~120 MB" is a design intention taken
  from the distroless base, not a measured figure.
- **Production anything.** No customer, no scale, no reliability claim.

## Known weaknesses

- Coverage is **not measured** in CI; there is no coverage gate and no recorded figure.
- CI installs **ruff unpinned**, so a future release that adds default rules will break
  the lint job on untouched files — the failure mode already fixed in
  `pipeline-dashboard` by pinning.
- The "Gap Closure Map" in the README maps role requirements to code that exists. Those
  rows are reading pointers, **not** results.
- `tests/test_pipeline.py` is a single 17-test file; the suite is small and unit-level
  throughout.
