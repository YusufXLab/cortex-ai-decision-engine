# syntax=docker/dockerfile:1.7
# ==============================================================================
# Multi-stage Dockerfile for Cortex AI Decision Engine
# Python 3.12 + Astral uv package manager
# ==============================================================================

# --- Stage 1: Build Dependencies ---
FROM ghcr.io/astral-sh/uv:0.5.11-python3.12-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation and frozen lockfile
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies in a separate layer for optimal caching
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    uv pip install --no-cache -e .

# Copy application source
COPY src/ /app/src/

# Install the application itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache -e .

# --- Stage 2: Final Runtime ---
FROM python:3.12-slim-bookworm AS runtime

# Set security headers and non-interactive environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Create a non-root system user for security
RUN groupadd -g 10001 cortex && \
    useradd -u 10001 -g cortex -s /bin/bash -m cortex && \
    mkdir -p /app/logs && \
    chown -R cortex:cortex /app

# Copy virtual environment and source from builder
COPY --from=builder --chown=cortex:cortex /app/.venv /app/.venv
COPY --chown=cortex:cortex src/ /app/src/
COPY --chown=cortex:cortex pyproject.toml /app/pyproject.toml

USER cortex

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()" || exit 1

ENTRYPOINT ["uvicorn", "cortex.main:app", "--host", "0.0.0.0", "--port", "8000"]
