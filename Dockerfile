# Architecture decision: Multi-stage Docker build with distroless runtime.
# Why: Stage 1 builds with full Python + build deps. Stage 2 copies only
# the installed packages into a distroless image (~120MB vs ~1GB for full
# Python image). Smaller images reduce AKS cold start times and attack
# surface.
#
# Architecture decision: python:3.12-slim for builder, distroless for runtime.
# Why: distroless images have no shell, no package manager, no utilities.
# This is the security best practice for production deployments on AKS,
# matching the enterprise security posture the Orbis role requires.

# === Stage 1: Build ===
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# === Stage 2: Runtime (distroless) ===
FROM gcr.io/distroless/python3-debian12:nonroot

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production

WORKDIR /app

COPY --from=builder /root/.local /home/nonroot/.local
ENV PATH=/home/nonroot/.local/bin:$PATH

# Copy application code (after deps to optimise Docker layer caching)
COPY --chown=nonroot:nonroot . .

# Health check — the /health endpoint returns 200 when the app is ready
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["/home/nonroot/.local/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

EXPOSE 8000

# Architecture decision: Gunicorn with Uvicorn workers over raw Uvicorn.
# Why: Gunicorn provides process management (worker lifecycle, graceful
# shutdown, worker timeouts) that raw Uvicorn lacks. Uvicorn workers handle
# the ASGI protocol. This is the recommended production pattern for FastAPI.
CMD ["gunicorn", "app.main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--keep-alive", "30"]
