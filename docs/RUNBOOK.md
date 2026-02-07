# FriendlyFace — Operational Runbook

Operational procedures for running FriendlyFace in production.

---

## 1. Startup & Shutdown

### Starting the Server

```bash
# Direct (development)
source .venv/bin/activate
python3 -m friendlyface

# Docker
docker run -p 8000:8000 \
  -e FF_API_KEYS="key1,key2" \
  -e FF_LOG_FORMAT=json \
  -v ./data:/app/data \
  friendlyface

# With migrations on startup
FF_MIGRATIONS_ENABLED=true python3 -m friendlyface
```

### Graceful Shutdown

Send `SIGTERM` to the process. The lifespan handler will:
1. Shut down the SSE broadcaster
2. Close the database connection
3. Exit cleanly

```bash
# Docker
docker stop <container_id>

# Direct
kill -TERM <pid>
```

---

## 2. Health Monitoring

### Health Endpoints

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /health` | Public | Basic liveness check |
| `GET /health/deep` | Public | Deep check (DB connectivity) |
| `GET /metrics` | Public | Prometheus metrics |
| `GET /dashboard` | Requires auth | Forensic health summary |

### Health Check Script

```bash
# Basic health
curl -s http://localhost:8000/health | jq .

# Deep health (checks DB)
curl -s http://localhost:8000/health/deep | jq .

# Prometheus metrics
curl -s http://localhost:8000/metrics
```

### Monitoring Alerts

Set up alerts for:
- `/health` returning non-200 → **Service down**
- `/health/deep` returning `"database": "disconnected"` → **DB issue**
- Response time p95 > 500ms → **Performance degradation**
- Rate limit 429 errors spike → **Potential abuse**

---

## 3. Backup & Restore

### Creating Backups

```bash
# Via API
curl -X POST http://localhost:8000/admin/backup \
  -H "X-API-Key: $FF_API_KEY" | jq .

# With label
curl -X POST "http://localhost:8000/admin/backup?label=pre-deploy" \
  -H "X-API-Key: $FF_API_KEY"
```

Backups are stored in `FF_BACKUP_DIR` (default: `backups/`).

### Listing Backups

```bash
curl http://localhost:8000/admin/backups \
  -H "X-API-Key: $FF_API_KEY" | jq .
```

### Backup Statistics

```bash
curl http://localhost:8000/admin/backup/stats \
  -H "X-API-Key: $FF_API_KEY" | jq .
```

### Verifying a Backup

```bash
curl -X POST "http://localhost:8000/admin/backup/verify?filename=ff_backup_20260207_120000_abc12345.db" \
  -H "X-API-Key: $FF_API_KEY" | jq .
```

### Restoring from Backup

**WARNING:** This replaces the current database.

```bash
curl -X POST "http://localhost:8000/admin/backup/restore?filename=ff_backup_20260207_120000_abc12345.db" \
  -H "X-API-Key: $FF_API_KEY" | jq .
```

After restore, restart the server to reinitialize in-memory state (Merkle tree, provenance DAG).

### Retention Policy

Controlled by environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FF_BACKUP_RETENTION_COUNT` | 10 | Max backups to keep |
| `FF_BACKUP_RETENTION_DAYS` | None | Delete backups older than N days |

Retention is enforced automatically after each backup creation. Manual cleanup:

```bash
curl -X POST http://localhost:8000/admin/backup/cleanup \
  -H "X-API-Key: $FF_API_KEY" | jq .
```

---

## 4. Database Migrations

### Checking Status

```bash
curl http://localhost:8000/admin/migrations/status \
  -H "X-API-Key: $FF_API_KEY" | jq .
```

Returns applied, pending, and total migration counts.

### Applying Migrations

Migrations run automatically on startup when `FF_MIGRATIONS_ENABLED=true`.

### Rolling Back

Roll back the last applied migration:

```bash
# Dry run (see what would happen)
curl -X POST "http://localhost:8000/admin/migrations/rollback?dry_run=true" \
  -H "X-API-Key: $FF_API_KEY" | jq .

# Actual rollback
curl -X POST http://localhost:8000/admin/migrations/rollback \
  -H "X-API-Key: $FF_API_KEY" | jq .
```

Rollback requires a `_down.sql` companion file for the migration.

---

## 5. Authentication

### Auth Providers

Controlled by `FF_AUTH_PROVIDER`:

| Provider | Value | Config Required |
|----------|-------|-----------------|
| API Key | `api_key` (default) | `FF_API_KEYS` |
| Supabase JWT | `supabase` | `FF_SUPABASE_JWT_SECRET` |
| OIDC | `oidc` | `FF_OIDC_ISSUER`, `FF_OIDC_AUDIENCE` |

### Dev Mode

When `FF_API_KEYS` is empty and provider is `api_key`, auth is disabled entirely.

### API Key Rotation

1. Add the new key to `FF_API_KEYS` (comma-separated)
2. Restart the server
3. Distribute the new key to clients
4. Remove the old key from `FF_API_KEYS`
5. Restart again

### Role-Based Access Control

Map keys to roles via `FF_API_KEY_ROLES`:

```bash
FF_API_KEY_ROLES='{"admin-key-1": ["admin"], "viewer-key-1": ["viewer"]}'
```

Keys without explicit mapping default to `["admin"]`.

Protected endpoints requiring `admin` role:
- `POST /admin/backup` — Create backup
- `POST /admin/backup/restore` — Restore backup
- `POST /admin/backup/cleanup` — Cleanup backups
- `POST /admin/migrations/rollback` — Rollback migration
- `POST /governance/compliance/generate` — Generate compliance report

### Bearer Token Support

All providers accept `Authorization: Bearer <token>` header. Legacy `X-API-Key` header and `api_key` query param are still supported for API key provider.

---

## 6. Troubleshooting

### Common Issues

#### Server won't start

- Check `FF_DB_PATH` points to a writable location
- Check `FF_PORT` isn't in use: `lsof -i :8000`
- Check Python version: requires 3.11+

#### Auth failures (403/401)

- Verify `FF_API_KEYS` is set correctly
- Check the key is being sent via `X-API-Key` header or `Authorization: Bearer`
- Check `FF_AUTH_PROVIDER` matches your auth method

#### Database locked errors

- Only one server instance should access a SQLite database
- Check for zombie processes: `ps aux | grep friendlyface`

#### Rate limiting (429)

- Default: 100 requests/minute per IP
- Override: `FF_RATE_LIMIT=500/minute` or `FF_RATE_LIMIT=none`
- Sensitive endpoints have lower limits (5-20/min)

### Structured Logging

Enable JSON logging for production:

```bash
FF_LOG_FORMAT=json FF_LOG_LEVEL=INFO python3 -m friendlyface
```

Log fields: `timestamp`, `level`, `logger`, `message`, `request_id`, `event_category`.

Audit events are tagged with `event_category=audit`.

---

## 7. Incident Response

### Data Integrity Violation

1. **Detect:** `GET /chain/integrity` returns failures
2. **Assess:** Check which events have mismatched hashes
3. **Preserve:** Create a backup immediately: `POST /admin/backup?label=incident`
4. **Investigate:** Check audit logs for unauthorized modifications
5. **Recover:** Restore from last known-good backup if needed

### Unauthorized Access

1. **Detect:** Audit logs show `auth_failure` events
2. **Rotate:** Change API keys immediately
3. **Review:** Check `GET /admin/backups` for unauthorized backup/restore
4. **Block:** Add IP blocks at load balancer level

### Database Corruption

1. **Verify:** `POST /admin/backup/verify?filename=...` for recent backups
2. **Restore:** Use most recent verified backup
3. **Restart:** Restart server to reinitialize in-memory state

---

## 8. Scaling

### SQLite (Default)

- Single-writer, multiple-reader
- Suitable for up to ~100 concurrent users
- Use WAL mode (enabled by default in aiosqlite)

### Supabase (Production)

For higher scale, switch to Supabase:

```bash
FF_STORAGE=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
```

### Docker Deployment

```bash
docker build -t friendlyface .
docker run -d --name ff \
  -p 8000:8000 \
  -e FF_API_KEYS="production-key" \
  -e FF_LOG_FORMAT=json \
  -e FF_MIGRATIONS_ENABLED=true \
  -v ff-data:/app/data \
  friendlyface
```

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `FF_STORAGE` | `sqlite` | `sqlite` or `supabase` |
| `FF_DB_PATH` | `friendlyface.db` | SQLite database path |
| `FF_API_KEYS` | *(empty)* | Comma-separated API keys |
| `FF_AUTH_PROVIDER` | `api_key` | `api_key`, `supabase`, `oidc` |
| `FF_API_KEY_ROLES` | *(empty)* | JSON: `{"key": ["role"]}` |
| `FF_SUPABASE_JWT_SECRET` | — | Supabase JWT secret |
| `FF_OIDC_ISSUER` | — | OIDC issuer URL |
| `FF_OIDC_AUDIENCE` | — | OIDC audience |
| `FF_LOG_FORMAT` | `text` | `text` or `json` |
| `FF_LOG_LEVEL` | `INFO` | Python log level |
| `FF_PORT` | `8000` | Server port |
| `FF_RATE_LIMIT` | `100/minute` | Rate limit (`none` to disable) |
| `FF_MIGRATIONS_ENABLED` | `false` | Run migrations on startup |
| `FF_BACKUP_DIR` | `backups` | Backup directory |
| `FF_BACKUP_RETENTION_COUNT` | `10` | Max backups to keep |
| `FF_BACKUP_RETENTION_DAYS` | — | Max backup age in days |
| `FF_CORS_ORIGINS` | `*` | CORS origins |
| `FF_SERVE_FRONTEND` | `true` | Serve React dashboard |
