# FriendlyFace — Deployment Guide

## Local Development

```bash
git clone https://github.com/Dicoangelo/FriendlyFace.git
cd FriendlyFace
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m friendlyface  # http://localhost:3849
```

---

## Docker

### Build

```bash
docker build -t friendlyface .
```

### Run

```bash
# Minimal (dev mode, no auth)
docker run -p 8000:8000 friendlyface

# With authentication
docker run -p 8000:8000 -e FF_API_KEYS=mykey friendlyface

# With Supabase backend
docker run -p 8000:8000 \
  -e FF_STORAGE=supabase \
  -e SUPABASE_URL=https://your-project.supabase.co \
  -e SUPABASE_KEY=your-service-role-key \
  -e FF_API_KEYS=mykey \
  friendlyface

# With persistent SQLite volume
docker run -p 8000:8000 \
  -v friendlyface-data:/app/data \
  -e FF_DB_PATH=/app/data/friendlyface.db \
  -e FF_API_KEYS=mykey \
  friendlyface

# With deterministic DID key
docker run -p 8000:8000 \
  -e FF_DID_SEED=$(python3 -c "import secrets; print(secrets.token_hex(32))") \
  -e FF_API_KEYS=mykey \
  friendlyface
```

### Docker Compose Example

```yaml
version: "3.8"
services:
  friendlyface:
    build: .
    ports:
      - "8000:8000"
    environment:
      FF_API_KEYS: "${FF_API_KEYS}"
      FF_STORAGE: sqlite
      FF_DB_PATH: /app/data/friendlyface.db
      FF_DID_SEED: "${FF_DID_SEED}"
    volumes:
      - ff-data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  ff-data:
```

---

## Fly.io

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Launch (uses fly.toml from repo)
fly launch --config fly.toml

# Set secrets
fly secrets set FF_API_KEYS=your-production-key
fly secrets set FF_DID_SEED=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# For Supabase backend
fly secrets set FF_STORAGE=supabase
fly secrets set SUPABASE_URL=https://your-project.supabase.co
fly secrets set SUPABASE_KEY=your-service-role-key

# Deploy
fly deploy

# View logs
fly logs

# SSH into the running instance
fly ssh console
```

### Fly.io Persistent Storage

For SQLite persistence on Fly.io, create a volume:

```bash
fly volumes create ff_data --size 1 --region iad
```

Update `fly.toml`:
```toml
[mounts]
  source = "ff_data"
  destination = "/app/data"

[env]
  FF_DB_PATH = "/app/data/friendlyface.db"
```

---

## Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Initialize project
railway init

# Deploy
railway up

# Set environment variables via dashboard or CLI
railway variables set FF_API_KEYS=your-production-key
railway variables set FF_DID_SEED=$(python3 -c "import secrets; print(secrets.token_hex(32))")
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FF_STORAGE` | No | `sqlite` | Storage backend: `sqlite` or `supabase` |
| `FF_DB_PATH` | No | `friendlyface.db` | SQLite database file path |
| `FF_API_KEYS` | **Yes (prod)** | *(empty=dev)* | Comma-separated API keys |
| `FF_HOST` | No | `0.0.0.0` | Server bind host |
| `FF_PORT` | No | `8000` | Server bind port |
| `FF_DID_SEED` | Recommended | *(random)* | 64 hex char seed for deterministic platform DID |
| `SUPABASE_URL` | If supabase | — | Supabase project URL |
| `SUPABASE_KEY` | If supabase | — | Supabase service role key |

### Production Checklist

Before deploying to production:

1. **Set `FF_API_KEYS`** — Never run without authentication in production
2. **Set `FF_DID_SEED`** — Use a fixed seed so the platform DID persists across restarts. Generate once and store securely:
   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```
3. **Choose storage backend** — SQLite is fine for single-instance deployments. Use Supabase for multi-instance or production scale.
4. **Configure persistent storage** — If using SQLite, ensure the database file survives container restarts (volume mount)
5. **Enable HTTPS** — Use a reverse proxy (nginx, Caddy) or platform-provided TLS (Fly.io, Railway)
6. **Monitor health** — Poll `GET /health` for uptime monitoring
7. **Set up log aggregation** — SSE stream at `GET /events/stream` for real-time forensic monitoring

### Storage Backend Decision Guide

| Scenario | Recommendation |
|----------|---------------|
| Local development | SQLite (default, zero config) |
| Single-server production | SQLite with persistent volume |
| Multi-instance / HA | Supabase |
| Testing / CI | SQLite (in-memory via tmp_path) |

---

## CI/CD

GitHub Actions CI runs on every push:

```yaml
# .github/workflows/ci.yml
- pytest tests/ -v
- ruff check friendlyface/ tests/
- ruff format --check friendlyface/ tests/
```

The CI badge on the README reflects the current build status.

---

## Monitoring & Observability

### Health Check

```bash
curl http://your-server:8000/health
# {"status": "healthy", "uptime_seconds": 12345.67}
```

### Chain Integrity

```bash
curl -H "X-API-Key: key" http://your-server:8000/chain/integrity
# {"valid": true, "count": 1234, "errors": []}
```

Run this periodically to detect any data corruption.

### Real-Time Event Stream

```bash
curl -N -H "X-API-Key: key" http://your-server:8000/events/stream
```

SSE stream of all forensic events as they occur. Connect a monitoring dashboard to this for live forensic audit visibility.

### Fairness Health

```bash
curl -H "X-API-Key: key" http://your-server:8000/fairness/status
# {"status": "pass", "last_audit_time": "...", ...}
```

Alert if status changes to `"warning"` or `"fail"`.
