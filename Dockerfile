# Multi-stage Dockerfile for html_render_service
#
# Single-context build: w2v-common resolves from the public git source declared
# via [tool.uv.sources] referencing the internal index in pyproject.toml.
# Build with: docker buildx build -f Dockerfile .
#
# This service rasterises generated HTML with Playwright/Chromium, so the
# runtime stage installs the Chromium browser + its system libs (same
# pattern as analysis_service).

ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Resolve and install deps; w2v-common comes from its public git tag
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself
COPY src/ ./src/
COPY alembic.ini ./alembic.ini

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ca-certificates \
    fonts-liberation \
    # Playwright/Chromium runtime deps (HTML screenshot rendering)
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 htmlrenderservice && \
    mkdir -p /app /app/logs /app/data && \
    chown -R htmlrenderservice:htmlrenderservice /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=htmlrenderservice:htmlrenderservice /app/.venv /app/.venv

# Install Playwright Chromium browser via venv script.
# This ~500MB layer depends only on the .venv (above) and the htmlrenderservice
# home dir (from useradd above) — NOT on src/. It sits here, ABOVE `COPY src/`,
# so a source-only commit no longer invalidates it: with registry_build_cache
# the stable layer is a cache hit and buildkit skips re-pushing its blob. Below
# the source copy (as it was) it rebuilt + re-pushed ~500MB every commit, hitting
# the build-timeout cap on the CI runner's ~1.2 MB/s uplink (gitops docs/ci-
# runner-uplink-bottleneck-2026-07-17.md). `--with-deps` kept: the explicit apt
# list is only a partial subset of Chromium's OS deps.
RUN PLAYWRIGHT_BROWSERS_PATH=/home/htmlrenderservice/.cache/ms-playwright \
    /app/.venv/bin/playwright install --with-deps chromium && \
    chown -R htmlrenderservice:htmlrenderservice /home/htmlrenderservice/.cache

# Copy application code
COPY --chown=htmlrenderservice:htmlrenderservice src/ ./src/
COPY --chown=htmlrenderservice:htmlrenderservice alembic.ini ./alembic.ini

ENV PATH="/app/.venv/bin:$PATH"

USER 1000:1000

ENV PLAYWRIGHT_BROWSERS_PATH=/home/htmlrenderservice/.cache/ms-playwright

# Port 6319 (HTML Render Service)
EXPOSE 6319

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.
#
# Alembic migrations run in a dedicated k8s initContainer (alembic: true in
# the python-service archetype), not in this CMD.

CMD ["python", "-m", "src.asgi"]
