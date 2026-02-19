# ──────────────────────────────────────────────────────────────
# FriendlyFace — Multi-stage Docker build with LiteFS
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Frontend build ─────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python builder ─────────────────────────────────
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

# ── Stage 3: Runtime with LiteFS ────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime libs + FUSE for LiteFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libjpeg62-turbo \
    zlib1g \
    libfreetype6 \
    curl \
    fuse3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install LiteFS binary from official image
COPY --from=flyio/litefs:0.5 /usr/local/bin/litefs /usr/local/bin/litefs

# Enable FUSE allow_other so non-root users can access the LiteFS mount
RUN echo "user_allow_other" >> /etc/fuse.conf

# Non-root user (LiteFS execs the app as this user)
RUN groupadd --gid 1000 ffuser && \
    useradd --uid 1000 --gid ffuser --create-home ffuser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source and migrations
WORKDIR /app
COPY --from=builder /build/friendlyface ./friendlyface
COPY --from=builder /build/migrations ./migrations

# Copy built frontend assets
COPY --from=frontend /frontend/dist ./frontend/dist

# Copy LiteFS configuration
COPY litefs.yml /etc/litefs.yml

# Create directories:
#   /litefs          — FUSE mount point (app reads/writes here)
#   /var/lib/litefs  — LiteFS internal data (on Fly persistent volume)
#   /app/data        — Legacy data dir (backups, etc.)
RUN mkdir -p /litefs /var/lib/litefs /app/data && \
    chown -R ffuser:ffuser /app

EXPOSE 8000

# Health check hits the LiteFS proxy (8080), not uvicorn directly.
# The proxy forwards to localhost:8000 after verifying DB health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# LiteFS starts as root (FUSE mount requirement), then execs
# the application as ffuser via the exec config in litefs.yml.
ENTRYPOINT ["litefs", "mount"]
