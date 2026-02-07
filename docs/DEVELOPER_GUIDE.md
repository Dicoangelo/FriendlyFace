# FriendlyFace — Developer Guide

## Prerequisites

- **Python 3.11+** (required — uses modern typing features)
- **Git** (for cloning and version control)
- **Docker** (optional, for containerized deployment)

---

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/Dicoangelo/FriendlyFace.git
cd FriendlyFace

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
python3 -m friendlyface  # Should start on http://localhost:3849
```

Open `http://localhost:3849/health` in your browser — you should see `{"status": "healthy"}`.

---

## Project Configuration

### pyproject.toml

The project uses `hatchling` as the build backend. Key sections:

```toml
[project]
name = "friendlyface"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
    "ruff>=0.8.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py311"
line-length = 100
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | ≥0.115.0 | Async web framework |
| uvicorn[standard] | ≥0.32.0 | ASGI server |
| pydantic | ≥2.10.0 | Data validation and models |
| aiosqlite | ≥0.20.0 | Async SQLite |
| scikit-learn | ≥1.4.0 | PCA + SVM ML pipeline |
| numpy | ≥1.26.0 | Numerical computing |
| Pillow | ≥10.0.0 | Image processing |
| python-multipart | ≥0.0.9 | File upload handling |
| lime | ≥0.2.0 | LIME explanations |
| supabase | ≥2.0.0 | Supabase client |
| pynacl | ≥1.5.0 | Ed25519 cryptography |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FF_STORAGE` | `sqlite` | Storage backend: `sqlite` or `supabase` |
| `FF_DB_PATH` | `friendlyface.db` | SQLite database file path |
| `FF_API_KEYS` | *(empty)* | Comma-separated API keys. Empty = dev mode (no auth) |
| `FF_HOST` | `0.0.0.0` | Server bind host |
| `FF_PORT` | `8000` | Server bind port (note: `__main__.py` uses 3849) |
| `FF_DID_SEED` | *(auto)* | Hex-encoded 32-byte seed for deterministic platform DID |
| `SUPABASE_URL` | — | Supabase project URL (required if `FF_STORAGE=supabase`) |
| `SUPABASE_KEY` | — | Supabase service role key |

### Setting Up for Development

```bash
# Minimal dev setup (no auth, SQLite)
source .venv/bin/activate
python3 -m friendlyface

# With auth enabled
FF_API_KEYS="devkey1,devkey2" python3 -m friendlyface

# With Supabase
FF_STORAGE=supabase SUPABASE_URL=https://xxx.supabase.co SUPABASE_KEY=xxx python3 -m friendlyface
```

---

## Running the Dev Server

```bash
source .venv/bin/activate
python3 -m friendlyface
```

The server starts on **port 3849** with auto-reload enabled. FastAPI auto-generates interactive API docs at:
- Swagger UI: `http://localhost:3849/docs`
- ReDoc: `http://localhost:3849/redoc`

---

## Testing

### Running Tests

```bash
# Full test suite (560+ tests, ~29 seconds)
pytest tests/ -v

# Quick run (less verbose)
pytest tests/ -q

# End-to-end pipeline test only
pytest tests/test_e2e_pipeline.py -v
```

### Testing by Layer

```bash
# Layer 1: Recognition
pytest tests/test_recognition_api.py tests/test_pca.py tests/test_svm.py -v

# Layer 2: Federated Learning
pytest tests/test_fl.py tests/test_fl_api.py tests/test_poisoning.py -v

# Layer 3: Blockchain Forensic
pytest tests/test_integrity.py tests/test_merkle.py tests/test_provenance.py tests/test_storage.py -v

# Layer 4: Fairness
pytest tests/test_fairness.py tests/test_fairness_api.py -v

# Layer 5: Explainability
pytest tests/test_explainability.py tests/test_shap_explainability.py -v

# Layer 6: Consent & Governance
pytest tests/test_consent_api.py tests/test_governance.py tests/test_compliance.py -v

# Cryptography
pytest tests/test_ed25519_did.py tests/test_schnorr.py tests/test_bundle_crypto.py -v

# Legacy compatibility
pytest tests/test_did.py -v
```

### Test Fixtures

The `conftest.py` provides shared fixtures used across all test files:

- **`db`** — Fresh SQLite database (in `tmp_path`, isolated per test)
- **`service`** — Initialized `ForensicService` with the test database
- **`client`** — `httpx.AsyncClient` wired to the FastAPI test application

All fixtures reset state between tests to ensure isolation.

### Writing New Tests

```python
import pytest
from friendlyface.core.models import EventType, ForensicEvent

class TestMyNewFeature:
    async def test_something(self, service):
        """Test using the ForensicService directly."""
        event = await service.record_event(
            event_type=EventType.INFERENCE_RESULT,
            actor="test",
            payload={"score": 0.95},
        )
        assert event.event_hash != ""
        assert event.sequence_number >= 0

    async def test_via_api(self, client):
        """Test using the HTTP client."""
        response = await client.get("/health")
        assert response.status_code == 200
```

Key patterns:
- Use `async def` for all test methods (pytest-asyncio handles the event loop)
- Use the `service` fixture for unit-testing internal logic
- Use the `client` fixture for integration-testing API endpoints
- Each test gets a fresh database — no cross-test contamination

---

## Linting and Formatting

```bash
# Check for issues
ruff check friendlyface/ tests/

# Check formatting
ruff format --check friendlyface/ tests/

# Auto-fix issues
ruff check --fix friendlyface/ tests/

# Auto-format
ruff format friendlyface/ tests/
```

Ruff is configured for Python 3.11 with a 100-character line length.

---

## Quality Gates

Before merging any code, all of these must pass:

```bash
# The canonical quality gate command
cd ~/friendlyface && source .venv/bin/activate && pytest tests/ -v && ruff check friendlyface/ tests/ && ruff format --check friendlyface/ tests/
```

CI runs this automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).

---

## Issue Tracking with `bd` (Beads)

FriendlyFace uses **bd** (beads) for issue tracking.

```bash
# Get started
bd onboard

# Find available work
bd ready

# View issue details
bd show <id>

# Claim work
bd update <id> --status in_progress

# Complete work
bd close <id>

# Sync with git
bd sync
```

### Session Completion Checklist

When ending a work session, you MUST complete ALL of these:

1. **File issues** for any remaining work
2. **Run quality gates** (tests + lint)
3. **Update issue status** (close finished, update in-progress)
4. **Push to remote:**
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # Must show "up to date with origin"
   ```
5. **Clean up** stashes and branches
6. **Verify** all changes committed AND pushed

Work is NOT complete until `git push` succeeds.

---

## Common Development Tasks

### Adding a New API Endpoint

1. Define request/response Pydantic models in `api/app.py` (or in `core/models.py` for shared models)
2. Add the endpoint function in `api/app.py`
3. Ensure the endpoint emits a `ForensicEvent` (this is the core architectural requirement)
4. Add integration tests in the appropriate test file
5. Run quality gates

### Adding a New Event Type

1. Add the new type to `EventType` enum in `core/models.py`
2. Use it in your endpoint/service code
3. Existing infrastructure (hash chaining, Merkle tree, provenance) handles it automatically

### Switching Storage Backend

```bash
# SQLite (default)
FF_STORAGE=sqlite python3 -m friendlyface

# Supabase
FF_STORAGE=supabase SUPABASE_URL=... SUPABASE_KEY=... python3 -m friendlyface
```

Both backends implement the same `Database` interface, so the rest of the code is unaware of which backend is active.

### Testing with Auth

```bash
# Start server with auth
FF_API_KEYS="test_key" python3 -m friendlyface

# Make authenticated requests
curl -H "X-API-Key: test_key" http://localhost:3849/events
curl "http://localhost:3849/events?api_key=test_key"
```

---

## Debugging Tips

### Checking Hash Chain Integrity
```bash
curl http://localhost:3849/chain/integrity
```

If this returns errors, it means events have been tampered with or the chain is corrupted.

### Viewing Merkle Root
```bash
curl http://localhost:3849/merkle/root
```

The Merkle root should change every time a new event is recorded.

### Checking Forensic Event Trail
```bash
# List all events
curl http://localhost:3849/events

# Get a specific event
curl http://localhost:3849/events/<uuid>

# Get provenance chain for a node
curl http://localhost:3849/provenance/<node_id>
```

### Database Location

By default, SQLite stores data in `friendlyface.db` in the working directory. To use a different location:

```bash
FF_DB_PATH=/tmp/test.db python3 -m friendlyface
```

The database is created automatically on first startup.
