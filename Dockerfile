# ──────────────────────────────────────────────────────────────
# FriendlyFace — Multi-stage Docker build
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps required to compile scikit-learn, numpy, Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install production deps into a virtual-env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install deps from a requirements step first for layer caching
COPY pyproject.toml README.md ./
COPY friendlyface/__init__.py ./friendlyface/__init__.py
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy full source and reinstall to pick up all modules
COPY friendlyface/ ./friendlyface/
COPY migrations/ ./migrations/
RUN pip install --no-cache-dir .

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime libs only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libjpeg62-turbo \
    zlib1g \
    libfreetype6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 ffuser && \
    useradd --uid 1000 --gid ffuser --create-home ffuser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source and migrations
WORKDIR /app
COPY --from=builder /build/friendlyface ./friendlyface
COPY --from=builder /build/migrations ./migrations

# Create data directory for SQLite persistence
RUN mkdir -p /app/data && chown -R ffuser:ffuser /app

# Switch to non-root user
USER ffuser

EXPOSE 8000

# Health check against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "friendlyface.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
