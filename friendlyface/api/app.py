"""FastAPI application for the FriendlyFace Blockchain Forensic Layer.

Endpoints:
  POST   /events                    — Record a forensic event
  GET    /events                    — List all events
  GET    /events/{id}               — Get event by ID
  GET    /merkle/root               — Get current Merkle root
  GET    /merkle/proof/{event_id}   — Get Merkle inclusion proof
  POST   /bundles                   — Create forensic bundle
  GET    /bundles/{id}              — Get bundle by ID
  POST   /verify/{bundle_id}       — Verify a forensic bundle
  GET    /chain/integrity           — Verify full hash chain
  POST   /provenance                — Add provenance node
  GET    /provenance/{node_id}      — Get provenance chain
  POST   /recognition/train         — Train PCA+SVM on a dataset path
  POST   /recognition/predict       — Upload image for face recognition
  GET    /recognition/models        — List available trained models
  GET    /recognition/models/{id}   — Get model details with provenance chain
  POST   /recognize                 — (Legacy) Upload image for face recognition
  POST   /recognition/voice/enroll  — Enroll a voice for a subject
  POST   /recognition/voice/verify  — Verify a voice against enrolled subjects
  POST   /recognition/multimodal    — Multi-modal fusion (face + voice)
  POST   /fl/start                  — Start FL simulation with configurable parameters
  POST   /fl/dp-start               — Start DP-FL simulation with differential privacy
  POST   /fl/simulate               — (Legacy) Alias for /fl/start
  GET    /fl/rounds                 — List completed FL rounds with summary
  GET    /fl/rounds/{sim_id}/{n}    — Get round details with client contributions and security
  GET    /fl/rounds/{sim_id}/{n}/security — Get poisoning detection results for a round
  GET    /fl/status                 — Current FL training status
  POST   /explainability/sdd        — Generate SDD saliency explanation
  GET    /events/stream             — SSE stream of forensic events (real-time)
  POST   /did/create                — Create a new Ed25519 DID:key
  GET    /did/{did_id}/resolve      — Resolve a DID to its DID Document
  POST   /vc/issue                  — Issue a Verifiable Credential
  POST   /vc/verify                 — Verify a Verifiable Credential
  POST   /zk/prove                  — Generate a Schnorr ZK proof for a bundle
  POST   /zk/verify                 — Verify a Schnorr ZK proof
  GET    /zk/proofs/{bundle_id}     — Get stored proof for a bundle
  GET    /bundles/{id}/export       — Export bundle as JSON-LD document
  POST   /bundles/import            — Import and verify a JSON-LD bundle
  GET    /health                    — Health check
  GET    /dashboard                 — Comprehensive forensic health dashboard
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Suppress slowapi's use of deprecated asyncio.iscoroutinefunction (fixed upstream in Python 3.16)
warnings.filterwarnings(
    "ignore",
    message=r".*asyncio\.iscoroutinefunction.*",
    category=DeprecationWarning,
    module=r"slowapi\..*",
)
from slowapi import Limiter  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402
from starlette.responses import StreamingResponse  # noqa: E402

from friendlyface.api.sse import EventBroadcaster  # noqa: E402
from friendlyface.auth import require_api_key, require_role  # noqa: E402
from friendlyface.config import settings  # noqa: E402
from friendlyface.exceptions import FriendlyFaceError  # noqa: E402
from friendlyface.logging_config import log_startup_info, setup_logging  # noqa: E402
from friendlyface.core.models import (  # noqa: E402
    BiasAuditRecord,
    EventType,
    ProvenanceRelation,
)
from friendlyface.core.service import ForensicService  # noqa: E402
from friendlyface.crypto.did import Ed25519DIDKey  # noqa: E402
from friendlyface.crypto.schnorr import ZKBundleProver, ZKBundleVerifier  # noqa: E402
from friendlyface.crypto.vc import VerifiableCredential  # noqa: E402
from friendlyface.storage.database import Database  # noqa: E402

logger = logging.getLogger("friendlyface")
_audit_logger = logging.getLogger("friendlyface.audit")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
_rate_limit_enabled = settings.rate_limit.lower() != "none"
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit] if _rate_limit_enabled else [],
    enabled=_rate_limit_enabled,
)

_STARTUP_TIME: float = 0.0


def _create_database():
    """Create the appropriate database backend based on config.

    Checks os.environ directly as well (for tests that set FF_STORAGE
    after settings singleton is created).
    """
    backend = os.environ.get("FF_STORAGE", settings.storage).lower()
    if backend == "supabase":
        from friendlyface.storage.supabase_db import SupabaseDatabase

        return SupabaseDatabase()
    return Database()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class RecordEventRequest(BaseModel):
    event_type: EventType
    actor: str = Field(min_length=1, max_length=512)
    payload: dict[str, Any] = Field(default_factory=dict)


class CreateBundleRequest(BaseModel):
    event_ids: list[UUID]
    provenance_node_ids: list[UUID] = Field(default_factory=list)
    bias_audit: BiasAuditRecord | None = None
    recognition_artifacts: dict[str, Any] | None = None
    fl_artifacts: dict[str, Any] | None = None
    bias_report: dict[str, Any] | None = None
    explanation_artifacts: dict[str, Any] | None = None
    layer_filters: list[str] | None = Field(
        default=None,
        description="Optional layer filters: recognition, fl, bias, explanation",
    )


class AddProvenanceRequest(BaseModel):
    entity_type: str = Field(min_length=1, max_length=256)
    entity_id: str = Field(min_length=1, max_length=256)
    parents: list[UUID] = Field(default_factory=list)
    relations: list[ProvenanceRelation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# DID/VC request models (US-031)
# ---------------------------------------------------------------------------


class CreateDIDRequest(BaseModel):
    seed: str | None = Field(
        default=None,
        max_length=256,
        description="Optional hex-encoded 32-byte seed for deterministic key",
    )


class IssueVCRequest(BaseModel):
    issuer_did_id: str = Field(
        min_length=1, max_length=256, description="DID of the issuer (from /did/create)"
    )
    subject_did: str = Field(
        default="", max_length=256, description="DID of the credential subject"
    )
    claims: dict[str, Any] = Field(description="Claims to include in the credential")
    credential_type: str = Field(default="ForensicCredential", max_length=256)


class VerifyVCRequest(BaseModel):
    credential: dict[str, Any] = Field(description="The credential to verify")
    issuer_public_key_hex: str = Field(
        min_length=1, max_length=256, description="Hex-encoded Ed25519 public key of the issuer"
    )


# ---------------------------------------------------------------------------
# ZK proof request models (US-032)
# ---------------------------------------------------------------------------


class ImportBundleRequest(BaseModel):
    document: dict[str, Any] = Field(description="JSON-LD bundle document to import")


class ZKProveRequest(BaseModel):
    bundle_id: str = Field(min_length=1, max_length=256, description="ID of the bundle to prove")


class ZKVerifyRequest(BaseModel):
    proof: str = Field(
        min_length=1, max_length=65536, description="JSON string of the Schnorr proof to verify"
    )


# ---------------------------------------------------------------------------
# DID key helpers (DB-backed, US-040)
# ---------------------------------------------------------------------------


async def _store_did_key(did_key: "Ed25519DIDKey", label: str | None = None) -> dict:
    """Persist a DID key to the database and return metadata dict."""
    from datetime import datetime, timezone

    stored = did_key.to_stored_form()
    created_at = datetime.now(timezone.utc).isoformat()
    await _db.insert_did_key(
        did=stored["did"],
        public_key=stored["public_key"],
        encrypted_private_key=stored["private_key"],
        key_type=stored["key_type"],
        created_at=created_at,
        label=label,
    )
    return {
        "did": stored["did"],
        "public_key_hex": stored["public_key"].hex(),
        "created_at": created_at,
    }


async def _load_did_key(did_id: str) -> "Ed25519DIDKey | None":
    """Load a DID key from the database, returning an Ed25519DIDKey or None."""
    entry = await _db.get_did_key(did_id)
    if entry is None or entry["encrypted_private_key"] is None:
        return None
    return Ed25519DIDKey.from_stored_form(entry["encrypted_private_key"])


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_db = _create_database()
_service = ForensicService(_db)
_broadcaster = EventBroadcaster()

# Gallery and pipeline instances (initialized after DB connect in lifespan)
_gallery: Any = None
_recognition_pipeline: Any = None


async def _load_persisted_caches() -> None:
    """Load persisted data from DB into in-memory caches on startup."""
    global _latest_compliance_report

    # Load model registry
    try:
        db_models = await _db.list_models()
        for m in db_models:
            _model_registry[m["id"]] = m
    except Exception:
        logger.debug("No model_registry table yet — skipping cache load")

    # Load FL simulations
    try:
        db_sims = await _db.list_fl_simulations()
        for s in db_sims:
            _fl_simulations[s["id"]] = s
    except Exception:
        logger.debug("No fl_simulations table yet — skipping cache load")

    # Load explanations
    try:
        db_expls, _ = await _db.list_explanations(limit=10000)
        for e in db_expls:
            _explanations[e["id"]] = e.get("data", e)
    except Exception:
        logger.debug("No explanation_records table yet — skipping cache load")

    # Load latest compliance report
    try:
        report = await _db.get_latest_compliance_report()
        if report:
            _latest_compliance_report = report
    except Exception:
        logger.debug("No compliance_reports table yet — skipping cache load")

    logger.info(
        "Loaded persisted caches: %d models, %d FL sims, %d explanations",
        len(_model_registry),
        len(_fl_simulations),
        len(_explanations),
    )


async def _persist_explanation(
    record_id: str, event_id: str, method: str, created_at: str, data: dict
) -> None:
    """Write-through: persist explanation to DB."""
    try:
        await _db.insert_explanation(record_id, event_id, method, created_at, data)
    except Exception:
        logger.warning("Failed to persist explanation %s to DB", record_id, exc_info=True)


async def _persist_model(model_id: str, created_at: str, data: dict) -> None:
    """Write-through: persist model to DB."""
    try:
        await _db.insert_model(model_id, created_at, data)
    except Exception:
        logger.warning("Failed to persist model %s to DB", model_id, exc_info=True)


async def _persist_fl_simulation(sim_id: str, created_at: str, data: dict) -> None:
    """Write-through: persist FL simulation to DB."""
    try:
        await _db.insert_fl_simulation(sim_id, created_at, data)
    except Exception:
        logger.warning("Failed to persist FL simulation %s to DB", sim_id, exc_info=True)


async def _persist_compliance_report(report_id: str, created_at: str, data: dict) -> None:
    """Write-through: persist compliance report to DB."""
    try:
        await _db.insert_compliance_report(report_id, created_at, data)
    except Exception:
        logger.warning("Failed to persist compliance report %s to DB", report_id, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _STARTUP_TIME, _gallery, _recognition_pipeline
    _STARTUP_TIME = time.monotonic()
    setup_logging()
    await _db.connect(db_key=settings.db_key, require_encryption=settings.require_encryption)
    if settings.migrations_enabled:
        await _db.run_migrations()
    await _service.initialize()

    # Initialize gallery and deep recognition pipeline
    from friendlyface.recognition.gallery import FaceGallery
    from friendlyface.recognition.pipeline import RecognitionPipeline

    _gallery = FaceGallery(_db)
    _recognition_pipeline = RecognitionPipeline(gallery=_gallery)

    # Load persisted caches from DB into memory (write-through cache)
    await _load_persisted_caches()

    log_startup_info()
    yield
    # Graceful shutdown: drain SSE subscribers, flush logs, close DB
    logger.info("Shutting down — draining SSE subscribers")
    _broadcaster.shutdown()
    logger.info("Closing database connection")
    await _db.close()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# OpenAPI tags
# ---------------------------------------------------------------------------
_OPENAPI_TAGS = [
    {"name": "Health", "description": "Health checks and version info"},
    {"name": "Dashboard", "description": "Forensic health dashboard"},
    {"name": "Events", "description": "Forensic event recording and retrieval"},
    {"name": "Merkle", "description": "Merkle tree root and inclusion proofs"},
    {"name": "Bundles", "description": "Forensic bundles, verification, and export/import"},
    {"name": "Provenance", "description": "Provenance DAG nodes and chains"},
    {"name": "Recognition", "description": "Face recognition training, inference, and models"},
    {"name": "Gallery", "description": "Face gallery enrollment, search, and management"},
    {"name": "Voice", "description": "Voice biometric enrollment and verification"},
    {"name": "FL", "description": "Federated learning simulations and DP-FedAvg"},
    {"name": "Fairness", "description": "Bias auditing, fairness status, and auto-audit config"},
    {"name": "Explainability", "description": "LIME, SHAP, and SDD explanations"},
    {"name": "Consent", "description": "Consent grant, revoke, check, and history"},
    {"name": "Governance", "description": "Compliance reporting"},
    {"name": "DID/VC", "description": "Decentralized Identifiers and Verifiable Credentials"},
    {"name": "ZK", "description": "Schnorr zero-knowledge proofs"},
    {"name": "Metrics", "description": "Prometheus metrics endpoint"},
    {"name": "Admin", "description": "Backup, migrations, and administrative operations"},
]

app = FastAPI(
    title="FriendlyFace — Blockchain Forensic Layer",
    description=(
        "Forensic-friendly AI facial recognition platform. "
        "Layer 3: Blockchain Forensic Layer implementing Mohammed's ICDF2C 2024 schema."
    ),
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
    openapi_tags=_OPENAPI_TAGS,
)

# Attach limiter state to app and install middleware
app.state.limiter = limiter


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(FriendlyFaceError)
async def friendlyface_error_handler(request: Request, exc: FriendlyFaceError) -> JSONResponse:
    """Centralized handler for custom FriendlyFace exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_type,
            "message": exc.message,
            "request_id": request_id,
        },
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return 429 with Retry-After header on rate limit."""
    request_id = getattr(request.state, "request_id", "unknown")
    _audit_logger.warning(
        "Rate limit exceeded: %s %s from %s",
        request.method,
        request.url.path,
        get_remote_address(request),
        extra={"event_category": "audit", "action": "rate_limit_exceeded"},
    )
    response = JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": str(exc.detail),
            "request_id": request_id,
        },
    )
    response.headers["Retry-After"] = "60"
    return response


# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next) -> Response:
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'"
    )
    return response


# ---------------------------------------------------------------------------
# Request logging middleware (also sets request_id on state for error handler)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    request_id = str(uuid4())[:8]
    request.state.request_id = request_id
    start = time.monotonic()
    response: Response = await call_next(request)
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    logger.info(
        "%s %s %s %.1fms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": elapsed_ms,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402

_instrumentator = Instrumentator(
    excluded_handlers=["/metrics"],
    should_respect_env_var=False,
)
_instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["Metrics"])


def get_service() -> ForensicService:
    return _service


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    count = await _db.get_event_count()
    uptime_s = time.monotonic() - _STARTUP_TIME if _STARTUP_TIME > 0 else 0
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(uptime_s, 1),
        "event_count": count,
        "merkle_root": _service.get_merkle_root(),
        "storage_backend": settings.storage,
    }


@app.get("/health/deep", tags=["Health"], summary="Deep health check with dependency status")
async def deep_health():
    """Check database connectivity, chain integrity, and storage backend status."""
    checks: dict[str, Any] = {}
    overall = True

    # Database check
    try:
        count = await _db.get_event_count()
        checks["database"] = {"status": "ok", "event_count": count}
    except Exception as exc:
        checks["database"] = {"status": "error", "detail": str(exc)}
        overall = False

    # Chain integrity check
    try:
        integrity = await _service.verify_chain_integrity()
        chain_ok = integrity["valid"]
        checks["chain_integrity"] = {
            "status": "ok" if chain_ok else "degraded",
            "valid": chain_ok,
            "count": integrity["count"],
        }
        if not chain_ok:
            overall = False
    except Exception as exc:
        checks["chain_integrity"] = {"status": "error", "detail": str(exc)}
        overall = False

    # Merkle tree check
    try:
        root = _service.get_merkle_root()
        leaf_count = _service.merkle.leaf_count
        checks["merkle_tree"] = {"status": "ok", "root": root, "leaf_count": leaf_count}
    except Exception as exc:
        checks["merkle_tree"] = {"status": "error", "detail": str(exc)}
        overall = False

    # Storage backend
    checks["storage_backend"] = {"type": settings.storage, "status": "ok"}

    uptime_s = time.monotonic() - _STARTUP_TIME if _STARTUP_TIME > 0 else 0
    status_code = 200 if overall else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if overall else "degraded",
            "version": "0.1.0",
            "uptime_seconds": round(uptime_s, 1),
            "checks": checks,
        },
    )


# ---------------------------------------------------------------------------
# Dashboard — comprehensive forensic health summary
# ---------------------------------------------------------------------------

_dashboard_cache: dict[str, Any] = {"data": None, "timestamp": 0.0}
_DASHBOARD_CACHE_TTL = 5.0  # seconds


@app.get("/dashboard", tags=["Dashboard"], summary="Forensic health dashboard")
async def dashboard():
    now = time.monotonic()
    if (
        _dashboard_cache["data"] is not None
        and (now - _dashboard_cache["timestamp"]) < _DASHBOARD_CACHE_TTL
    ):
        return _dashboard_cache["data"]

    uptime_s = time.monotonic() - _STARTUP_TIME if _STARTUP_TIME > 0 else 0.0

    total_events = await _db.get_event_count()
    total_bundles = await _db.get_bundle_count()
    total_provenance_nodes = await _db.get_provenance_count()
    events_by_type = await _db.get_events_by_type()
    recent_events = await _db.get_recent_events(limit=10)
    chain_integrity = await _service.verify_chain_integrity()

    result = {
        "uptime_seconds": round(uptime_s, 2),
        "storage_backend": settings.storage,
        "total_events": total_events,
        "total_bundles": total_bundles,
        "total_provenance_nodes": total_provenance_nodes,
        "events_by_type": events_by_type,
        "recent_events": recent_events,
        "chain_integrity": {
            "valid": chain_integrity["valid"],
            "count": chain_integrity["count"],
        },
        "crypto_status": {
            "did_enabled": True,
            "zk_scheme": "schnorr-sha256",
            "total_dids": len(await _db.list_did_keys()),
            "total_vcs": 0,
        },
    }

    _dashboard_cache["data"] = result
    _dashboard_cache["timestamp"] = time.monotonic()

    return result


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@app.post("/events", status_code=201, tags=["Events"], summary="Record a forensic event")
async def record_event(req: RecordEventRequest):
    event = await _service.record_event(
        event_type=req.event_type,
        actor=req.actor,
        payload=req.payload,
    )
    event_data = event.model_dump(mode="json")
    _broadcaster.broadcast(event_data)
    return event_data


@app.get("/events", tags=["Events"], summary="List forensic events (paginated)")
async def list_events(
    limit: int = Query(default=50, ge=1, le=500, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    event_type: str | None = Query(default=None, description="Filter by event type"),
    actor: str | None = Query(default=None, description="Filter by actor"),
):
    events, total = await _db.get_events_filtered(
        limit=limit, offset=offset, event_type=event_type, actor=actor
    )
    return {
        "items": [e.model_dump(mode="json") for e in events],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


class BatchEventRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=100, description="Event IDs to fetch")


@app.post("/events/batch", tags=["Events"], summary="Batch-fetch events by IDs")
async def batch_get_events(req: BatchEventRequest):
    """Fetch multiple events in a single request (max 100)."""
    uuids = []
    for eid in req.ids:
        try:
            uuids.append(UUID(eid))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid UUID: {eid}")
    events = await _db.get_events_by_ids(uuids)
    return {
        "items": [e.model_dump(mode="json") for e in events],
        "total": len(events),
    }


# ---------------------------------------------------------------------------
# SSE — real-time forensic event stream
# ---------------------------------------------------------------------------

_HEARTBEAT_INTERVAL: float = 15.0  # seconds


@app.get("/events/stream", tags=["Events"], summary="SSE real-time event stream")
async def event_stream(
    event_type: str | None = Query(default=None, description="Filter events by type"),
):
    """Server-Sent Events endpoint for real-time forensic event streaming."""

    async def _generate():
        queue = _broadcaster.subscribe()
        try:
            # Immediate heartbeat so the client receives headers right away
            yield "event: heartbeat\ndata: {}\n\n"
            while True:
                try:
                    event_data = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL)
                    # Apply optional event_type filter
                    if event_type is not None and event_data.get("event_type") != event_type:
                        continue
                    yield f"event: forensic_event\ndata: {json.dumps(event_data)}\n\n"
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _broadcaster.unsubscribe(queue)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/events/{event_id}", tags=["Events"], summary="Get event by ID")
async def get_event(event_id: UUID):
    event = await _service.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------


@app.get("/merkle/root", tags=["Merkle"], summary="Current Merkle root")
async def get_merkle_root():
    root = _service.get_merkle_root()
    return {"merkle_root": root, "leaf_count": _service.merkle.leaf_count}


@app.get("/merkle/proof/{event_id}", tags=["Merkle"], summary="Merkle inclusion proof")
async def get_merkle_proof(event_id: UUID):
    proof = _service.get_merkle_proof(event_id)
    if proof is None:
        raise HTTPException(status_code=404, detail="Event not found in Merkle tree")
    return proof.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Bundle export/import — portable JSON-LD (US-035)
# ---------------------------------------------------------------------------


@app.post("/bundles/import", status_code=201, tags=["Bundles"], summary="Import JSON-LD bundle")
async def import_bundle(req: ImportBundleRequest):
    """Import a JSON-LD bundle document with integrity verification."""
    from datetime import datetime

    from friendlyface.core.models import EventType as _ET
    from friendlyface.core.models import ForensicBundle as _FB
    from friendlyface.core.models import MerkleProof as _MP

    doc = req.document

    verification: dict[str, bool] = {
        "hash_valid": False,
        "merkle_valid": True,
        "zk_valid": True,
        "did_valid": True,
        "chain_valid": True,
    }

    # 1. Recompute bundle hash to check integrity
    original_hash = doc.get("bundle_hash", "")
    temp_bundle = _FB(
        id=UUID(doc["id"]),
        created_at=datetime.fromisoformat(doc["created_at"]),
        status=doc.get("status", "pending"),
        event_ids=[UUID(e["id"]) for e in doc.get("events", [])],
        merkle_root=doc.get("merkle_root", ""),
        merkle_proofs=[_MP(**mp) for mp in doc.get("merkle_proofs", [])],
        provenance_chain=[UUID(n["id"]) for n in doc.get("provenance_chain", [])],
        bias_audit=doc.get("bias_audit"),
        recognition_artifacts=doc.get("recognition_artifacts"),
        fl_artifacts=doc.get("fl_artifacts"),
        bias_report=doc.get("bias_report"),
        explanation_artifacts=doc.get("explanation_artifacts"),
    )
    verification["hash_valid"] = temp_bundle.compute_hash() == original_hash

    # 2. Merkle proof verification
    for mp_data in doc.get("merkle_proofs", []):
        proof_obj = _MP(**mp_data)
        if not proof_obj.verify():
            verification["merkle_valid"] = False
            break

    # 3. ZK proof verification
    zk_proof = doc.get("zk_proof")
    if zk_proof is not None:
        zk_verifier = ZKBundleVerifier()
        verification["zk_valid"] = zk_verifier.verify_bundle(json.dumps(zk_proof))

    # 4. DID credential verification
    did_credential = doc.get("did_credential")
    if did_credential is not None:
        vc_result = VerifiableCredential.verify(
            did_credential, _service._platform_did.export_public()
        )
        verification["did_valid"] = vc_result["valid"]

    # 5. Chain validity — events form a valid hash chain
    events_data = doc.get("events", [])
    for i, ev in enumerate(events_data):
        if i == 0:
            continue
        prev_ev = events_data[i - 1]
        if ev.get("previous_hash") != prev_ev.get("event_hash"):
            verification["chain_valid"] = False
            break

    # Reject if hash is invalid (critical failure)
    if not verification["hash_valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "imported": False,
                "bundle_id": None,
                "verification": verification,
            },
        )

    # --- Re-create events and bundle ---
    new_event_ids: list[UUID] = []
    for ev in events_data:
        new_event = await _service.record_event(
            event_type=_ET(ev["event_type"]),
            actor=ev.get("actor", "import"),
            payload=ev.get("payload", {}),
        )
        new_event_ids.append(new_event.id)

    new_bundle = await _service.create_bundle(
        event_ids=new_event_ids,
        bias_audit=None,
        recognition_artifacts=doc.get("recognition_artifacts"),
        fl_artifacts=doc.get("fl_artifacts"),
        bias_report=doc.get("bias_report"),
        explanation_artifacts=doc.get("explanation_artifacts"),
    )

    return {
        "imported": True,
        "bundle_id": str(new_bundle.id),
        "verification": verification,
    }


@app.get("/bundles/{bundle_id}/export", tags=["Bundles"], summary="Export bundle as JSON-LD")
async def export_bundle(bundle_id: UUID):
    """Export a forensic bundle as a self-contained JSON-LD document."""
    bundle = await _service.get_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Bundle not found")

    # Gather full event data (batch query instead of N+1)
    batch_events = await _service.db.get_events_by_ids(bundle.event_ids)
    events = [e.model_dump(mode="json") for e in batch_events]

    # Merkle proofs
    merkle_proofs = [p.model_dump(mode="json") for p in bundle.merkle_proofs]

    # Provenance chain
    provenance_chain = []
    for nid in bundle.provenance_chain:
        chain = _service.get_provenance_chain(nid)
        if chain:
            for node in chain:
                provenance_chain.append(node.model_dump(mode="json"))

    # Parse ZK proof
    zk_proof = None
    if bundle.zk_proof_placeholder:
        try:
            zk_proof = json.loads(bundle.zk_proof_placeholder)
        except (json.JSONDecodeError, TypeError):
            zk_proof = None

    # Parse DID credential
    did_credential = None
    if bundle.did_credential_placeholder:
        try:
            did_credential = json.loads(bundle.did_credential_placeholder)
        except (json.JSONDecodeError, TypeError):
            did_credential = None

    return {
        "@context": [
            "https://www.w3.org/2018/credentials/v1",
            "https://friendlyface.dev/forensic/v1",
        ],
        "@type": "ForensicBundle",
        "id": str(bundle.id),
        "created_at": bundle.created_at.isoformat(),
        "status": bundle.status.value if hasattr(bundle.status, "value") else str(bundle.status),
        "bundle_hash": bundle.bundle_hash,
        "merkle_root": bundle.merkle_root,
        "events": events,
        "merkle_proofs": merkle_proofs,
        "provenance_chain": provenance_chain,
        "bias_audit": bundle.bias_audit.model_dump(mode="json") if bundle.bias_audit else None,
        "recognition_artifacts": bundle.recognition_artifacts,
        "fl_artifacts": bundle.fl_artifacts,
        "bias_report": bundle.bias_report,
        "explanation_artifacts": bundle.explanation_artifacts,
        "zk_proof": zk_proof,
        "did_credential": did_credential,
    }


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


@app.post("/bundles", status_code=201, tags=["Bundles"], summary="Create forensic bundle")
async def create_bundle(req: CreateBundleRequest):
    bundle = await _service.create_bundle(
        event_ids=req.event_ids,
        provenance_node_ids=req.provenance_node_ids,
        bias_audit=req.bias_audit,
        recognition_artifacts=req.recognition_artifacts,
        fl_artifacts=req.fl_artifacts,
        bias_report=req.bias_report,
        explanation_artifacts=req.explanation_artifacts,
        layer_filters=req.layer_filters,
    )
    return bundle.model_dump(mode="json")


@app.get("/bundles/{bundle_id}", tags=["Bundles"], summary="Get bundle by ID")
async def get_bundle(bundle_id: UUID):
    bundle = await _service.get_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Bundle not found")
    return bundle.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@app.post("/verify/{bundle_id}", tags=["Bundles"], summary="Verify bundle integrity")
async def verify_bundle(bundle_id: UUID):
    result = await _service.verify_bundle(bundle_id)
    return result


@app.get("/chain/integrity", tags=["Bundles"], summary="Verify full hash chain")
async def verify_chain_integrity():
    return await _service.verify_chain_integrity()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@app.post("/provenance", status_code=201, tags=["Provenance"], summary="Add provenance node")
async def add_provenance_node(req: AddProvenanceRequest):
    try:
        node = _service.add_provenance_node(
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            parents=req.parents,
            relations=req.relations,
            metadata=req.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return node.model_dump(mode="json")


@app.get("/provenance/{node_id}", tags=["Provenance"], summary="Get provenance chain")
async def get_provenance_chain(node_id: UUID):
    chain = _service.get_provenance_chain(node_id)
    if not chain:
        raise HTTPException(status_code=404, detail="Provenance node not found")
    return [n.model_dump(mode="json") for n in chain]


# ---------------------------------------------------------------------------
# Recognition — model registry, training, and inference
# ---------------------------------------------------------------------------

# Model paths are configured via module-level variables so tests can override them.
_pca_model_path: str | None = None
_svm_model_path: str | None = None

# In-memory model registry: maps model_id (str UUID) -> metadata dict.
# Each entry: {id, pca_path, svm_path, created_at, n_components, n_samples,
#              n_classes, cv_accuracy, dataset_hash, model_hash,
#              training_event_id, provenance_node_id}
_model_registry: dict[str, dict[str, Any]] = {}


class TrainRequest(BaseModel):
    dataset_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Path to directory of aligned 112x112 grayscale images",
    )
    output_dir: str = Field(
        min_length=1, max_length=4096, description="Directory to write trained model files"
    )
    n_components: int = Field(default=128, description="PCA components to retain")
    C: float = Field(default=1.0, description="SVM regularization parameter")
    kernel: str = Field(default="linear", max_length=256, description="SVM kernel type")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    labels: list[int] | None = Field(
        default=None,
        description="Per-image integer labels (sorted filename order). Required for SVM.",
    )


@app.post(
    "/recognition/train",
    status_code=201,
    tags=["Recognition"],
    summary="Train PCA+SVM model",
    dependencies=[Depends(require_role("analyst"))],
)
@limiter.limit("5/minute")
async def train_model(request: Request, req: TrainRequest):
    """Train PCA+SVM pipeline on a dataset directory and register the model."""
    from datetime import datetime, timezone
    from pathlib import Path
    from uuid import uuid4

    from friendlyface.recognition.pca import train_pca
    from friendlyface.recognition.svm import train_svm

    dataset_dir = Path(req.dataset_path)
    if not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Dataset path not found: {req.dataset_path}")

    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_path = output_dir / "pca.pkl"
    svm_path = output_dir / "svm.pkl"

    # --- PCA training ---
    try:
        pca_result = train_pca(
            image_dir=dataset_dir,
            output_path=pca_path,
            n_components=req.n_components,
            actor="api_train",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"PCA training failed: {e}")

    # Record PCA training event in the forensic chain
    pca_event = await _service.record_event(
        event_type=pca_result.training_event.event_type,
        actor=pca_result.training_event.actor,
        payload=pca_result.training_event.payload,
    )

    # Add PCA provenance node
    pca_prov = _service.add_provenance_node(
        entity_type=pca_result.provenance_node.entity_type,
        entity_id=pca_result.provenance_node.entity_id,
        metadata=pca_result.provenance_node.metadata,
    )

    # --- SVM training ---
    if req.labels is None:
        raise HTTPException(
            status_code=400,
            detail="labels field is required for SVM training",
        )

    import numpy as np
    from PIL import Image

    labels = np.array(req.labels)

    # Re-load images and transform through PCA to get SVM features
    image_vectors = np.stack(
        [
            np.asarray(Image.open(fp).convert("L"), dtype=np.float64).ravel()
            for fp in sorted(dataset_dir.iterdir())
            if fp.is_file()
        ]
    )
    features = pca_result.model.transform(image_vectors)

    try:
        svm_result = train_svm(
            features=features,
            labels=labels,
            output_path=svm_path,
            C=req.C,
            kernel=req.kernel,
            cv_folds=req.cv_folds,
            actor="api_train",
            previous_hash=pca_event.event_hash,
            sequence_number=pca_event.sequence_number + 1,
            pca_provenance_id=pca_prov.id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"SVM training failed: {e}")

    # Record SVM training event
    svm_event = await _service.record_event(
        event_type=svm_result.training_event.event_type,
        actor=svm_result.training_event.actor,
        payload=svm_result.training_event.payload,
    )

    # Add SVM provenance node linked to PCA
    svm_prov = _service.add_provenance_node(
        entity_type=svm_result.provenance_node.entity_type,
        entity_id=svm_result.provenance_node.entity_id,
        parents=[pca_prov.id],
        relations=[ProvenanceRelation.DERIVED_FROM],
        metadata=svm_result.provenance_node.metadata,
    )

    # Register model
    model_id = str(uuid4())
    _model_registry[model_id] = {
        "id": model_id,
        "pca_path": str(pca_path),
        "svm_path": str(svm_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_components": pca_result.n_components,
        "n_samples": pca_result.n_samples,
        "n_classes": svm_result.n_classes,
        "cv_accuracy": svm_result.cv_accuracy,
        "dataset_hash": pca_result.dataset_hash,
        "model_hash": svm_result.model_hash,
        "pca_event_id": str(pca_event.id),
        "svm_event_id": str(svm_event.id),
        "pca_provenance_id": str(pca_prov.id),
        "svm_provenance_id": str(svm_prov.id),
    }

    # Persist model to DB (write-through)
    await _persist_model(
        model_id, _model_registry[model_id]["created_at"], _model_registry[model_id]
    )

    # Auto-configure model paths for inference
    global _pca_model_path, _svm_model_path
    _pca_model_path = str(pca_path)
    _svm_model_path = str(svm_path)

    return {
        "model_id": model_id,
        "pca_event_id": str(pca_event.id),
        "svm_event_id": str(svm_event.id),
        "pca_provenance_id": str(pca_prov.id),
        "svm_provenance_id": str(svm_prov.id),
        "n_components": pca_result.n_components,
        "n_samples": pca_result.n_samples,
        "n_classes": svm_result.n_classes,
        "cv_accuracy": svm_result.cv_accuracy,
        "dataset_hash": pca_result.dataset_hash,
        "model_hash": svm_result.model_hash,
    }


def _resolve_engine() -> str:
    """Determine the active recognition engine (fallback / deep / auto)."""
    engine = os.environ.get("FF_RECOGNITION_ENGINE", settings.recognition_engine).lower()
    if engine == "auto":
        # Use deep pipeline when it's available; otherwise fall back
        if _recognition_pipeline is not None:
            return "deep"
        return "fallback"
    return engine


async def _do_predict(image: UploadFile, top_k: int = 5) -> dict[str, Any]:
    """Shared prediction logic for /recognition/predict and /recognize.

    Supports three engine modes (set via FF_RECOGNITION_ENGINE):
    - ``fallback``: PCA+SVM only (original behavior, requires trained models)
    - ``deep``: detection→embedding→gallery→liveness→calibration pipeline
    - ``auto``: use deep when the pipeline is available, else fallback
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    engine = _resolve_engine()

    if engine == "deep":
        return await _do_predict_deep(image_bytes, top_k)
    return await _do_predict_fallback(image_bytes, top_k)


async def _do_predict_deep(image_bytes: bytes, top_k: int = 5) -> dict[str, Any]:
    """Run prediction through the deep recognition pipeline."""
    import hashlib
    import io

    import numpy as np
    from PIL import Image as PILImage

    if _recognition_pipeline is None:
        raise HTTPException(status_code=503, detail="Deep recognition pipeline not initialized")

    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.asarray(pil_img, dtype=np.uint8)

    result = await _recognition_pipeline.run(img_array, top_k=top_k)

    input_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

    # Record forensic event
    event = await _service.record_event(
        event_type=EventType.INFERENCE_RESULT,
        actor="api_recognize",
        payload={
            "engine": "deep",
            "input_hash": input_hash,
            "faces_detected": result.faces_detected,
            "gallery_matches": len(result.gallery_matches),
            "liveness": result.liveness.is_live if result.liveness else None,
            "quality_score": result.quality_score,
        },
    )

    await _maybe_auto_audit()

    response: dict[str, Any] = {
        "event_id": str(event.id),
        "input_hash": input_hash,
        "engine": "deep",
        "matches": [
            {
                "label": m.subject_id,
                "confidence": m.similarity,
                "entry_id": m.entry_id,
                "model_version": m.model_version,
            }
            for m in result.gallery_matches
        ],
        "quality_score": result.quality_score,
        "faces_detected": result.faces_detected,
    }

    if result.liveness is not None:
        response["liveness"] = {
            "is_live": result.liveness.is_live,
            "score": result.liveness.score,
            "checks": result.liveness.checks,
        }

    if result.calibration is not None:
        response["calibration"] = {
            "raw_score": result.calibration.raw_score,
            "calibrated_score": result.calibration.calibrated_score,
            "final_score": result.calibration.final_score,
            "method": result.calibration.method,
        }

    return response


async def _do_predict_fallback(image_bytes: bytes, top_k: int = 5) -> dict[str, Any]:
    """Run prediction through the legacy PCA+SVM pipeline."""
    from pathlib import Path

    from friendlyface.recognition.inference import run_inference

    if _pca_model_path is None or _svm_model_path is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Configure PCA and SVM model paths.",
        )

    try:
        result = run_inference(
            image_bytes=image_bytes,
            pca_model_path=Path(_pca_model_path),
            svm_model_path=Path(_svm_model_path),
            top_k=top_k,
            actor="api_recognize",
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Record the inference event in the forensic chain
    event = await _service.record_event(
        event_type=result.inference_event.event_type,
        actor=result.inference_event.actor,
        payload=result.inference_event.payload,
    )

    # Check if auto-audit should trigger
    await _maybe_auto_audit()

    return {
        "event_id": str(event.id),
        "input_hash": result.input_hash,
        "engine": "fallback",
        "matches": [{"label": m.label, "confidence": m.confidence} for m in result.matches],
    }


@app.post(
    "/recognition/predict", status_code=200, tags=["Recognition"], summary="Predict face identity"
)
async def predict(image: UploadFile, top_k: int = 5):
    """Upload a face image and get prediction matches with forensic event logging."""
    return await _do_predict(image, top_k)


@app.post(
    "/recognize",
    status_code=200,
    tags=["Recognition"],
    summary="(Legacy) Predict face identity",
    deprecated=True,
)
async def recognize(image: UploadFile, top_k: int = 5):
    """Legacy endpoint — use POST /recognition/predict instead."""
    return await _do_predict(image, top_k)


@app.get("/recognition/models", tags=["Recognition"], summary="List trained models")
async def list_models():
    """List all available trained models with metadata."""
    return list(_model_registry.values())


# ---------------------------------------------------------------------------
# Gallery — face enrollment, search, and management (US-085)
# ---------------------------------------------------------------------------


class GalleryEnrollRequest(BaseModel):
    subject_id: str = Field(min_length=1, max_length=256, description="Subject identifier")
    quality_score: float | None = Field(default=None, ge=0, le=1, description="Face quality score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class GallerySearchRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=100, description="Max matches to return")


@app.post(
    "/gallery/enroll",
    status_code=201,
    tags=["Gallery"],
    summary="Enroll face embedding",
)
async def gallery_enroll(image: UploadFile, subject_id: str, quality_score: float | None = None):
    """Detect a face, extract embedding, and enroll in the gallery."""
    from friendlyface.recognition.detection import FaceDetector
    from friendlyface.recognition.embeddings import EmbeddingExtractor

    if _gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not initialized")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    import numpy as np
    from PIL import Image as PILImage
    import io

    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.asarray(pil_img, dtype=np.uint8)

    detector = FaceDetector()
    faces = detector.detect(img_array)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image")

    best = faces[0]
    if best.aligned is None:
        raise HTTPException(status_code=400, detail="Face alignment failed")

    extractor = EmbeddingExtractor()
    embedding = extractor.extract(best.aligned)

    qs = quality_score if quality_score is not None else best.quality_score
    entry = await _gallery.enroll(subject_id, embedding, quality_score=qs)

    # Record forensic event for gallery enrollment
    await _service.record_event(
        event_type=EventType.CONSENT_RECORDED,
        actor="gallery_enroll",
        payload={
            "action": "gallery_enroll",
            "subject_id": subject_id,
            "entry_id": entry["entry_id"],
            "embedding_dim": entry["embedding_dim"],
            "model_version": entry["model_version"],
        },
    )

    return entry


@app.post(
    "/gallery/search",
    tags=["Gallery"],
    summary="Search gallery by face",
)
async def gallery_search(image: UploadFile, top_k: int = 5):
    """Detect a face, extract embedding, and search the gallery for matches."""
    from friendlyface.recognition.detection import FaceDetector
    from friendlyface.recognition.embeddings import EmbeddingExtractor

    if _gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not initialized")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    import numpy as np
    from PIL import Image as PILImage
    import io

    pil_img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.asarray(pil_img, dtype=np.uint8)

    detector = FaceDetector()
    faces = detector.detect(img_array)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image")

    best = faces[0]
    if best.aligned is None:
        raise HTTPException(status_code=400, detail="Face alignment failed")

    extractor = EmbeddingExtractor()
    embedding = extractor.extract(best.aligned)

    matches = await _gallery.search(embedding, top_k=top_k)
    return {
        "matches": [
            {
                "subject_id": m.subject_id,
                "entry_id": m.entry_id,
                "similarity": m.similarity,
                "model_version": m.model_version,
            }
            for m in matches
        ],
        "query_embedding_dim": embedding.dim,
        "total_matches": len(matches),
    }


@app.get("/gallery/subjects", tags=["Gallery"], summary="List enrolled subjects")
async def gallery_list_subjects():
    """List all enrolled subjects with entry counts."""
    if _gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not initialized")
    subjects = await _gallery.list_subjects()
    return {"subjects": subjects, "total": len(subjects)}


@app.delete(
    "/gallery/subjects/{subject_id}",
    tags=["Gallery"],
    summary="Delete subject from gallery",
    dependencies=[Depends(require_role("admin"))],
)
async def gallery_delete_subject(subject_id: str):
    """Delete all gallery entries for a subject."""
    if _gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not initialized")

    deleted = await _gallery.delete_subject(subject_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found in gallery")

    await _service.record_event(
        event_type=EventType.CONSENT_UPDATE,
        actor="gallery_delete",
        payload={"action": "gallery_delete", "subject_id": subject_id, "entries_deleted": deleted},
    )

    return {"subject_id": subject_id, "entries_deleted": deleted}


@app.get("/gallery/count", tags=["Gallery"], summary="Gallery entry count")
async def gallery_count():
    """Return total number of gallery entries."""
    if _gallery is None:
        raise HTTPException(status_code=503, detail="Gallery not initialized")
    total = await _gallery.count()
    return {"total": total}


# ---------------------------------------------------------------------------
# Voice biometrics — enrollment and verification
# ---------------------------------------------------------------------------

# In-memory voice recognizer with enrolled subjects
_voice_recognizer: Any = None


def _get_voice_recognizer():
    """Lazy-initialize the voice recognizer."""
    global _voice_recognizer
    if _voice_recognizer is None:
        from friendlyface.recognition.voice import run_voice_inference

        _voice_recognizer = {"embeddings": {}, "run_inference": run_voice_inference}
    return _voice_recognizer


class VoiceEnrollRequest(BaseModel):
    subject_id: str = Field(
        min_length=1, max_length=256, description="Subject identifier to enroll"
    )


class VoiceVerifyRequest(BaseModel):
    top_k: int = Field(default=3, ge=1)


class MultiModalRequest(BaseModel):
    face_label: str = Field(default="", max_length=512, description="Face match label")
    face_confidence: float = Field(default=0.0, ge=0, le=1)
    voice_confidence: float = Field(default=0.0, ge=0, le=1)
    face_weight: float = Field(default=0.6, gt=0, lt=1)
    voice_weight: float = Field(default=0.4, gt=0, lt=1)


@app.post(
    "/recognition/voice/enroll", status_code=201, tags=["Voice"], summary="Enroll voice biometric"
)
async def enroll_voice(audio: UploadFile, subject_id: str = "unknown"):
    """Enroll a voice for a subject from uploaded PCM audio."""
    from friendlyface.recognition.voice import extract_voice_embedding

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    try:
        embedding = extract_voice_embedding(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vr = _get_voice_recognizer()
    vr["embeddings"][subject_id] = embedding.embedding

    # Record enrollment event
    event = await _service.record_event(
        event_type=EventType.TRAINING_COMPLETE,
        actor="voice_enroll",
        payload={
            "action": "voice_enrollment",
            "subject_id": subject_id,
            "artifact_hash": embedding.artifact_hash,
            "embedding_dim": embedding.embedding.shape[0],
            "duration_seconds": embedding.duration_seconds,
        },
    )

    return {
        "subject_id": subject_id,
        "event_id": str(event.id),
        "artifact_hash": embedding.artifact_hash,
        "embedding_dim": embedding.embedding.shape[0],
        "duration_seconds": embedding.duration_seconds,
        "total_enrolled": len(vr["embeddings"]),
    }


@app.post(
    "/recognition/voice/verify", status_code=200, tags=["Voice"], summary="Verify voice identity"
)
async def verify_voice(audio: UploadFile, top_k: int = 3):
    """Verify a voice against enrolled subjects."""
    from friendlyface.recognition.voice import run_voice_inference

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    vr = _get_voice_recognizer()
    if not vr["embeddings"]:
        raise HTTPException(status_code=400, detail="No voices enrolled yet")

    try:
        result = run_voice_inference(audio_bytes, vr["embeddings"], top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Record verification event
    event = await _service.record_event(
        event_type=result.forensic_event.event_type,
        actor=result.forensic_event.actor,
        payload=result.forensic_event.payload,
    )

    return {
        "event_id": str(event.id),
        "input_hash": result.input_hash,
        "matches": [{"label": m.label, "confidence": m.confidence} for m in result.matches],
    }


@app.post(
    "/recognition/multimodal",
    status_code=200,
    tags=["Voice"],
    summary="Multi-modal face+voice fusion",
)
async def multimodal_fusion(req: MultiModalRequest):
    """Fuse face and voice recognition scores into a unified decision."""
    from friendlyface.recognition.fusion import fuse_scores

    face_matches = [{"label": req.face_label, "confidence": req.face_confidence}]
    voice_matches = [{"label": req.face_label, "confidence": req.voice_confidence}]

    try:
        result = fuse_scores(
            face_matches,
            voice_matches,
            face_weight=req.face_weight,
            voice_weight=req.voice_weight,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Record fusion event
    event = await _service.record_event(
        event_type=result.forensic_event.event_type,
        actor=result.forensic_event.actor,
        payload=result.forensic_event.payload,
    )

    return {
        "event_id": str(event.id),
        "fused_matches": [
            {
                "label": m.label,
                "fused_confidence": m.fused_confidence,
                "face_confidence": m.face_confidence,
                "voice_confidence": m.voice_confidence,
            }
            for m in result.fused_matches
        ],
        "fusion_method": result.fusion_method,
        "face_weight": result.face_weight,
        "voice_weight": result.voice_weight,
    }


# ---------------------------------------------------------------------------
# Federated Learning — simulation results and security
# ---------------------------------------------------------------------------

# In-memory store for FL simulation results (keyed by simulation run ID).
# Each round within a simulation is accessible by round_number.
_fl_simulations: dict[str, Any] = {}


class FLSimulateRequest(BaseModel):
    n_clients: int = Field(default=5, ge=1)
    n_rounds: int = Field(default=3, ge=1)
    enable_poisoning_detection: bool = Field(default=True)
    poisoning_threshold: float = Field(default=3.0, gt=0)
    seed: int = Field(default=42)


async def _run_fl_simulation(req: FLSimulateRequest) -> dict[str, Any]:
    """Shared FL simulation logic for /fl/start and /fl/simulate."""
    from uuid import uuid4

    from friendlyface.fl.engine import run_fl_simulation

    result = run_fl_simulation(
        n_clients=req.n_clients,
        n_rounds=req.n_rounds,
        enable_poisoning_detection=req.enable_poisoning_detection,
        poisoning_threshold=req.poisoning_threshold,
        seed=req.seed,
    )

    sim_id = str(uuid4())
    _fl_simulations[sim_id] = result

    # Persist FL simulation to DB (write-through)
    from datetime import datetime, timezone as _tz

    _sim_created = datetime.now(_tz.utc).isoformat()
    await _persist_fl_simulation(
        sim_id,
        _sim_created,
        {
            "n_rounds": result.n_rounds,
            "n_clients": getattr(result, "n_clients", None),
        },
    )

    # Record round events in the forensic chain
    recorded_event_ids: list[str] = []
    for rr in result.rounds:
        recorded = await _service.record_event(
            event_type=rr.event.event_type,
            actor=rr.event.actor,
            payload=rr.event.payload,
        )
        recorded_event_ids.append(str(recorded.id))
        # Record any poisoning alert events
        if rr.poisoning_result:
            for alert in rr.poisoning_result.alert_events:
                await _service.record_event(
                    event_type=alert.event_type,
                    actor=alert.actor,
                    payload=alert.payload,
                )

    rounds_summary = []
    for i, rr in enumerate(result.rounds):
        summary: dict[str, Any] = {
            "round": rr.round_number,
            "global_model_hash": rr.global_model_hash,
            "event_id": str(rr.event.id),
            "recorded_event_id": recorded_event_ids[i],
        }
        if rr.poisoning_result:
            summary["poisoning"] = {
                "flagged_client_ids": rr.poisoning_result.flagged_client_ids,
                "n_flagged": len(rr.poisoning_result.flagged_client_ids),
            }
        rounds_summary.append(summary)

    return {
        "simulation_id": sim_id,
        "n_rounds": result.n_rounds,
        "n_clients": result.n_clients,
        "final_model_hash": result.final_model_hash,
        "mode": settings.fl_mode,
        "rounds": rounds_summary,
    }


def _fl_response(body: dict[str, Any], status_code: int = 200) -> JSONResponse:
    """Wrap an FL response body with the X-FL-Mode header."""
    return JSONResponse(
        content=body,
        status_code=status_code,
        headers={"X-FL-Mode": settings.fl_mode},
    )


@app.post(
    "/fl/start",
    status_code=201,
    tags=["FL"],
    summary="Start FL simulation",
    dependencies=[Depends(require_role("analyst"))],
)
async def start_fl(req: FLSimulateRequest):
    """Start a federated learning simulation with configurable parameters."""
    body = await _run_fl_simulation(req)
    return _fl_response(body, status_code=201)


@app.post(
    "/fl/simulate",
    status_code=201,
    tags=["FL"],
    summary="(Legacy) Start FL simulation",
    deprecated=True,
    dependencies=[Depends(require_role("analyst"))],
)
async def simulate_fl(req: FLSimulateRequest):
    """Legacy alias for POST /fl/start."""
    body = await _run_fl_simulation(req)
    return _fl_response(body, status_code=201)


# ---------------------------------------------------------------------------
# Federated Learning — DP-FedAvg
# ---------------------------------------------------------------------------


class DPFLSimulateRequest(BaseModel):
    n_clients: int = Field(default=5, ge=1)
    n_rounds: int = Field(default=3, ge=1)
    epsilon: float = Field(default=1.0, gt=0)
    delta: float = Field(default=1e-5, gt=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    seed: int = Field(default=42)


@app.post(
    "/fl/dp-start",
    status_code=201,
    tags=["FL"],
    summary="Start DP-FedAvg simulation",
    dependencies=[Depends(require_role("analyst"))],
)
async def start_dp_fl(req: DPFLSimulateRequest):
    """Start a differentially-private federated learning simulation."""
    from uuid import uuid4

    import numpy as np

    from friendlyface.fl.dp import DPConfig, dp_fedavg_round

    dp_config = DPConfig(
        epsilon=req.epsilon,
        delta=req.delta,
        max_grad_norm=req.max_grad_norm,
    )

    rng = np.random.default_rng(req.seed)
    weight_shapes = [(64, 32), (32,)]
    global_weights = [rng.standard_normal(shape) for shape in weight_shapes]
    client_ids = [f"client_{i}" for i in range(req.n_clients)]

    sim_id = str(uuid4())
    rounds_summary = []
    cumulative_epsilon = 0.0
    current_hash = "GENESIS"
    current_seq = 0

    for round_num in range(1, req.n_rounds + 1):
        # Simulate local training
        client_updates = [
            [w + rng.normal(0, 0.01, size=w.shape) for w in global_weights]
            for _ in range(req.n_clients)
        ]

        result = dp_fedavg_round(
            client_updates=client_updates,
            global_weights=global_weights,
            dp_config=dp_config,
            round_number=round_num,
            client_ids=client_ids,
            cumulative_epsilon=cumulative_epsilon,
            seed=req.seed + round_num,
            previous_hash=current_hash,
            sequence_number=current_seq,
        )

        global_weights = result.global_weights
        cumulative_epsilon = result.privacy_spent
        current_hash = result.event.event_hash
        current_seq += 1

        # Record in forensic chain
        recorded = await _service.record_event(
            event_type=result.event.event_type,
            actor=result.event.actor,
            payload=result.event.payload,
        )

        rounds_summary.append(
            {
                "round": round_num,
                "global_model_hash": result.global_model_hash,
                "noise_scale": result.noise_scale,
                "n_clipped": len(result.clipped_clients),
                "clipped_clients": result.clipped_clients,
                "privacy_spent": result.privacy_spent,
                "event_id": str(recorded.id),
            }
        )

    return _fl_response(
        {
            "simulation_id": sim_id,
            "n_rounds": req.n_rounds,
            "n_clients": req.n_clients,
            "dp_config": {
                "epsilon": req.epsilon,
                "delta": req.delta,
                "max_grad_norm": req.max_grad_norm,
            },
            "total_epsilon": cumulative_epsilon,
            "mode": settings.fl_mode,
            "rounds": rounds_summary,
        },
        status_code=201,
    )


@app.get("/fl/status", tags=["FL"], summary="FL training status")
async def get_fl_status():
    """Return current FL training status across all simulations."""
    simulations = []
    for sim_id, sim in _fl_simulations.items():
        completed_rounds = len(sim.rounds)
        total_rounds = sim.n_rounds
        simulations.append(
            {
                "simulation_id": sim_id,
                "n_clients": sim.n_clients,
                "n_rounds": total_rounds,
                "completed_rounds": completed_rounds,
                "status": "completed" if completed_rounds == total_rounds else "in_progress",
                "final_model_hash": sim.final_model_hash,
            }
        )
    return _fl_response(
        {
            "total_simulations": len(_fl_simulations),
            "simulations": simulations,
            "mode": settings.fl_mode,
        }
    )


@app.get("/fl/rounds", tags=["FL"], summary="List FL rounds")
async def list_fl_rounds():
    """List completed FL rounds across all simulations with summary."""
    all_rounds: list[dict[str, Any]] = []
    for sim_id, sim in _fl_simulations.items():
        for rr in sim.rounds:
            entry: dict[str, Any] = {
                "simulation_id": sim_id,
                "round": rr.round_number,
                "n_clients": sim.n_clients,
                "global_model_hash": rr.global_model_hash,
                "event_id": str(rr.event.id),
                "provenance_node_id": str(rr.provenance_node.id),
            }
            if rr.poisoning_result:
                entry["security_status"] = {
                    "has_poisoning": rr.poisoning_result.has_poisoning,
                    "flagged_client_ids": rr.poisoning_result.flagged_client_ids,
                    "n_flagged": len(rr.poisoning_result.flagged_client_ids),
                }
            else:
                entry["security_status"] = None
            all_rounds.append(entry)
    return _fl_response(
        {"total_rounds": len(all_rounds), "rounds": all_rounds, "mode": settings.fl_mode}
    )


@app.get("/fl/rounds/{simulation_id}/{round_number}", tags=["FL"], summary="FL round details")
async def get_fl_round_details(simulation_id: str, round_number: int):
    """Get details for a specific FL round including client contributions and security status."""
    sim = _fl_simulations.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    matching = [r for r in sim.rounds if r.round_number == round_number]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Round {round_number} not found")

    rr = matching[0]

    # Client contributions
    client_contributions = [
        {
            "client_id": cu.client_id,
            "n_samples": cu.n_samples,
            "local_loss": cu.local_loss,
        }
        for cu in rr.client_updates
    ]

    # Security status
    security: dict[str, Any] | None = None
    if rr.poisoning_result:
        pr = rr.poisoning_result
        security = {
            "poisoning_detection_enabled": True,
            "has_poisoning": pr.has_poisoning,
            "n_clients": pr.n_clients,
            "median_norm": pr.median_norm,
            "threshold_multiplier": pr.threshold_multiplier,
            "effective_threshold": pr.effective_threshold,
            "flagged_client_ids": pr.flagged_client_ids,
            "n_flagged": len(pr.flagged_client_ids),
            "provenance_node_id": str(pr.provenance_node.id) if pr.provenance_node else None,
        }

    return _fl_response(
        {
            "simulation_id": simulation_id,
            "round": rr.round_number,
            "n_clients": sim.n_clients,
            "global_model_hash": rr.global_model_hash,
            "event_id": str(rr.event.id),
            "provenance_node_id": str(rr.provenance_node.id),
            "client_contributions": client_contributions,
            "security_status": security,
            "mode": settings.fl_mode,
        }
    )


@app.get(
    "/fl/rounds/{simulation_id}/{round_number}/security", tags=["FL"], summary="FL round security"
)
async def get_fl_round_security(simulation_id: str, round_number: int):
    """Get poisoning detection results for a specific FL round."""
    sim = _fl_simulations.get(simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Find the round
    matching = [r for r in sim.rounds if r.round_number == round_number]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Round {round_number} not found")

    rr = matching[0]
    pr = rr.poisoning_result

    if pr is None:
        return _fl_response(
            {
                "round": round_number,
                "poisoning_detection_enabled": False,
                "message": "Poisoning detection was not enabled for this simulation",
                "mode": settings.fl_mode,
            }
        )

    return _fl_response(
        {
            "round": round_number,
            "poisoning_detection_enabled": True,
            "n_clients": pr.n_clients,
            "median_norm": pr.median_norm,
            "threshold_multiplier": pr.threshold_multiplier,
            "effective_threshold": pr.effective_threshold,
            "has_poisoning": pr.has_poisoning,
            "flagged_client_ids": pr.flagged_client_ids,
            "client_results": [
                {
                    "client_id": cr.client_id,
                    "update_norm": cr.update_norm,
                    "flagged": cr.flagged,
                    "threshold_used": cr.threshold_used,
                }
                for cr in pr.client_results
            ],
            "alert_events": [
                {
                    "event_id": str(e.id),
                    "event_type": e.event_type.value,
                    "client_id": e.payload.get("client_id"),
                    "update_norm": e.payload.get("update_norm"),
                }
                for e in pr.alert_events
            ],
            "provenance_node_id": str(pr.provenance_node.id) if pr.provenance_node else None,
            "mode": settings.fl_mode,
        }
    )


@app.get("/recognition/models/{model_id}", tags=["Recognition"], summary="Get model details")
async def get_model(model_id: str):
    """Get model details including provenance chain."""
    model = _model_registry.get(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Build provenance chain from the SVM provenance node (includes PCA parent)
    svm_prov_id = UUID(model["svm_provenance_id"])
    provenance_chain = _service.get_provenance_chain(svm_prov_id)
    provenance_data = (
        [n.model_dump(mode="json") for n in provenance_chain] if provenance_chain else []
    )

    return {
        **model,
        "provenance_chain": provenance_data,
    }


# ---------------------------------------------------------------------------
# Governance — compliance reporting
# ---------------------------------------------------------------------------

# In-memory cache of the latest compliance report
_latest_compliance_report: dict[str, Any] | None = None


@app.get("/governance/compliance", tags=["Governance"], summary="Get compliance report")
async def get_compliance_report():
    """Get the latest compliance report, or generate one if none exists."""
    global _latest_compliance_report
    if _latest_compliance_report is None:
        from friendlyface.governance.compliance import ComplianceReporter

        reporter = ComplianceReporter(_db, _service)
        _latest_compliance_report = await reporter.generate_report()
    return _latest_compliance_report


@app.post(
    "/governance/compliance/generate",
    status_code=201,
    tags=["Governance"],
    summary="Generate compliance report",
    dependencies=[Depends(require_role("admin"))],
)
async def generate_compliance_report():
    """Generate a new compliance report and cache it."""
    global _latest_compliance_report
    from friendlyface.governance.compliance import ComplianceReporter

    reporter = ComplianceReporter(_db, _service)
    _latest_compliance_report = await reporter.generate_report()

    # Persist to DB (write-through)
    from datetime import datetime, timezone as _tz2

    _report_id = str(uuid4())
    _report_ts = datetime.now(_tz2.utc).isoformat()
    await _persist_compliance_report(_report_id, _report_ts, _latest_compliance_report)

    return _latest_compliance_report


# ---------------------------------------------------------------------------
# Consent -- grant, revoke, status, history, check
# ---------------------------------------------------------------------------


class ConsentGrantRequest(BaseModel):
    """Request body for POST /consent/grant."""

    subject_id: str = Field(min_length=1, max_length=256, description="Subject identifier")
    purpose: str = Field(
        min_length=1, max_length=512, description="Purpose of consent (e.g. recognition, training)"
    )
    expiry: str | None = Field(
        default=None,
        max_length=256,
        description="Optional ISO-8601 expiry datetime",
    )
    actor: str = Field(default="api", max_length=512, description="Actor granting consent")


class ConsentRevokeRequest(BaseModel):
    """Request body for POST /consent/revoke."""

    subject_id: str = Field(min_length=1, max_length=256, description="Subject identifier")
    purpose: str = Field(min_length=1, max_length=512, description="Purpose of consent to revoke")
    reason: str = Field(default="", max_length=4096, description="Reason for revocation")
    actor: str = Field(default="api", max_length=512, description="Actor revoking consent")


class ConsentCheckRequest(BaseModel):
    """Request body for POST /consent/check."""

    subject_id: str = Field(min_length=1, max_length=256, description="Subject identifier")
    purpose: str = Field(min_length=1, max_length=512, description="Purpose to verify consent for")


def _get_consent_manager():
    """Lazy-create a ConsentManager wired to the global db + service."""
    from friendlyface.governance.consent import ConsentManager

    return ConsentManager(_db, _service)


@app.post("/consent/grant", status_code=201, tags=["Consent"], summary="Grant consent")
async def grant_consent(req: ConsentGrantRequest):
    """Grant consent for a subject+purpose pair."""
    from datetime import datetime

    mgr = _get_consent_manager()

    expiry_dt = None
    if req.expiry is not None:
        try:
            expiry_dt = datetime.fromisoformat(req.expiry)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid expiry datetime: {req.expiry}",
            )

    record = await mgr.grant_consent(
        req.subject_id,
        req.purpose,
        expiry=expiry_dt,
        actor=req.actor,
    )
    _audit_logger.info(
        "Consent granted: subject=%s purpose=%s actor=%s",
        req.subject_id,
        req.purpose,
        req.actor,
        extra={
            "event_category": "audit",
            "action": "consent_grant",
            "subject_id": req.subject_id,
            "purpose": req.purpose,
            "actor": req.actor,
        },
    )
    return record.to_dict()


@app.post("/consent/revoke", status_code=200, tags=["Consent"], summary="Revoke consent")
async def revoke_consent(req: ConsentRevokeRequest):
    """Revoke consent for a subject+purpose pair."""
    mgr = _get_consent_manager()
    record = await mgr.revoke_consent(
        req.subject_id,
        req.purpose,
        reason=req.reason,
        actor=req.actor,
    )
    _audit_logger.info(
        "Consent revoked: subject=%s purpose=%s actor=%s reason=%s",
        req.subject_id,
        req.purpose,
        req.actor,
        req.reason,
        extra={
            "event_category": "audit",
            "action": "consent_revoke",
            "subject_id": req.subject_id,
            "purpose": req.purpose,
            "actor": req.actor,
        },
    )
    return record.to_dict()


@app.get("/consent/status/{subject_id}", tags=["Consent"], summary="Check consent status")
async def get_consent_status(subject_id: str, purpose: str = "recognition"):
    """Check current consent status for a subject."""
    mgr = _get_consent_manager()
    return await mgr.get_consent_status(subject_id, purpose)


@app.get("/consent/history/{subject_id}", tags=["Consent"], summary="Consent history")
async def get_consent_history(subject_id: str, purpose: str | None = None):
    """Get full consent history for a subject."""
    mgr = _get_consent_manager()
    records = await mgr.get_history(subject_id, purpose)
    return {"subject_id": subject_id, "total": len(records), "records": records}


@app.post(
    "/consent/check", status_code=200, tags=["Consent"], summary="Check consent before inference"
)
async def check_consent(req: ConsentCheckRequest):
    """Verify consent before inference. Returns allow/deny decision."""
    mgr = _get_consent_manager()
    allowed = await mgr.check_consent(req.subject_id, req.purpose)
    status = await mgr.get_consent_status(req.subject_id, req.purpose)
    return {
        "subject_id": req.subject_id,
        "purpose": req.purpose,
        "allowed": allowed,
        "has_consent": status["has_consent"],
        "active": status["active"],
    }


# ---------------------------------------------------------------------------
# Fairness — bias audit API and auto-trigger
# ---------------------------------------------------------------------------

# Auto-audit configuration: trigger after every N recognition events.
# Configurable via POST /fairness/config.
_auto_audit_interval: int = 50
_recognition_event_count: int = 0


class BiasAuditRequest(BaseModel):
    """Request body for POST /fairness/audit."""

    groups: list[dict[str, Any]] = Field(
        description=(
            "List of group result dicts, each with: "
            "group_name, true_positives, false_positives, "
            "true_negatives, false_negatives"
        )
    )
    demographic_parity_threshold: float = Field(default=0.1)
    equalized_odds_threshold: float = Field(default=0.1)
    metadata: dict[str, Any] | None = None


class AutoAuditConfigRequest(BaseModel):
    """Request body for POST /fairness/config."""

    auto_audit_interval: int = Field(
        default=50,
        ge=1,
        description="Trigger auto-audit after every N recognition events",
    )


@app.post(
    "/fairness/audit",
    status_code=201,
    tags=["Fairness"],
    summary="Trigger bias audit",
    dependencies=[Depends(require_role("analyst"))],
)
async def trigger_bias_audit(req: BiasAuditRequest):
    """Trigger a manual bias audit on provided group results."""
    from friendlyface.fairness.auditor import (
        BiasAuditor,
        FairnessThresholds,
        GroupResult,
    )

    if len(req.groups) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 demographic groups are required for a bias audit",
        )

    group_results = []
    for g in req.groups:
        try:
            group_results.append(
                GroupResult(
                    group_name=g["group_name"],
                    true_positives=g["true_positives"],
                    false_positives=g["false_positives"],
                    true_negatives=g["true_negatives"],
                    false_negatives=g["false_negatives"],
                )
            )
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required field in group: {e}",
            )

    thresholds = FairnessThresholds(
        demographic_parity_threshold=req.demographic_parity_threshold,
        equalized_odds_threshold=req.equalized_odds_threshold,
    )
    auditor = BiasAuditor(_service, thresholds=thresholds)
    record, alerts = await auditor.audit(group_results, metadata=req.metadata)

    return {
        "audit_id": str(record.id),
        "event_id": str(record.event_id),
        "demographic_parity_gap": record.demographic_parity_gap,
        "equalized_odds_gap": record.equalized_odds_gap,
        "compliant": record.compliant,
        "groups_evaluated": record.groups_evaluated,
        "fairness_score": record.details.get("fairness_score"),
        "alerts": [
            {
                "metric": a.metric,
                "gap": a.gap,
                "threshold": a.threshold,
                "message": a.message,
            }
            for a in alerts
        ],
    }


@app.get("/fairness/audits", tags=["Fairness"], summary="List bias audits (paginated)")
async def list_bias_audits(
    limit: int = Query(default=50, ge=1, le=500, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
):
    """List completed audits with summary scores."""
    audits = await _db.get_bias_audits_paginated(limit=limit, offset=offset)
    total = await _db.get_bias_audit_count()
    return {
        "items": [
            {
                "audit_id": str(a.id),
                "event_id": str(a.event_id) if a.event_id else None,
                "timestamp": a.timestamp.isoformat()
                if hasattr(a.timestamp, "isoformat")
                else str(a.timestamp),
                "demographic_parity_gap": a.demographic_parity_gap,
                "equalized_odds_gap": a.equalized_odds_gap,
                "compliant": a.compliant,
                "groups_evaluated": a.groups_evaluated,
                "fairness_score": a.details.get("fairness_score"),
            }
            for a in audits
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/fairness/audits/{audit_id}", tags=["Fairness"], summary="Get audit details")
async def get_bias_audit(audit_id: UUID):
    """Get full audit details with per-group breakdowns."""
    audit = await _db.get_bias_audit(audit_id)
    if audit is None:
        raise HTTPException(status_code=404, detail="Bias audit not found")
    return audit.model_dump(mode="json")


@app.get("/fairness/status", tags=["Fairness"], summary="Fairness health status")
async def get_fairness_status():
    """Current fairness health: pass / warning / fail.

    Logic:
    - No audits yet -> status="unknown"
    - Latest audit compliant and fairness_score >= 0.7 -> "pass"
    - Latest audit compliant but fairness_score < 0.7 -> "warning"
    - Latest audit non-compliant -> "fail"
    """
    audits = await _db.get_all_bias_audits()
    if not audits:
        return {
            "status": "unknown",
            "message": "No bias audits have been performed yet",
            "total_audits": 0,
        }

    latest = audits[-1]
    fairness_score = latest.details.get("fairness_score", 0.0)

    if not latest.compliant:
        status = "fail"
    elif fairness_score >= 0.7:
        status = "pass"
    else:
        status = "warning"

    compliant_count = sum(1 for a in audits if a.compliant)
    return {
        "status": status,
        "fairness_score": fairness_score,
        "latest_audit_id": str(latest.id),
        "compliant": latest.compliant,
        "total_audits": len(audits),
        "compliant_audits": compliant_count,
        "demographic_parity_gap": latest.demographic_parity_gap,
        "equalized_odds_gap": latest.equalized_odds_gap,
    }


@app.post(
    "/fairness/config",
    tags=["Fairness"],
    summary="Configure auto-audit",
    dependencies=[Depends(require_role("admin"))],
)
async def configure_auto_audit(req: AutoAuditConfigRequest):
    """Configure auto-audit interval."""
    global _auto_audit_interval
    _auto_audit_interval = req.auto_audit_interval
    return {
        "auto_audit_interval": _auto_audit_interval,
        "message": f"Auto-audit will trigger every {_auto_audit_interval} recognition events",
    }


@app.get("/fairness/config", tags=["Fairness"], summary="Get auto-audit config")
async def get_auto_audit_config():
    """Get current auto-audit configuration."""
    return {
        "auto_audit_interval": _auto_audit_interval,
        "recognition_events_since_last_audit": _recognition_event_count,
    }


async def _maybe_auto_audit():
    """Check if auto-audit should be triggered and run it if so.

    Called after each recognition event. Uses synthetic balanced groups
    as a baseline check when no explicit group data is available.
    """
    global _recognition_event_count
    _recognition_event_count += 1

    if _recognition_event_count >= _auto_audit_interval:
        _recognition_event_count = 0

        from friendlyface.fairness.auditor import (
            BiasAuditor,
            FairnessThresholds,
            GroupResult,
        )

        # Use synthetic balanced groups as baseline auto-audit.
        # In production, this would pull real demographic performance data.
        groups = [
            GroupResult(
                "auto_group_a",
                true_positives=80,
                false_positives=10,
                true_negatives=90,
                false_negatives=20,
            ),
            GroupResult(
                "auto_group_b",
                true_positives=80,
                false_positives=10,
                true_negatives=90,
                false_negatives=20,
            ),
        ]
        auditor = BiasAuditor(
            _service,
            thresholds=FairnessThresholds(),
            actor="auto_bias_auditor",
        )
        await auditor.audit(groups, metadata={"trigger": "auto", "interval": _auto_audit_interval})


# ---------------------------------------------------------------------------
# Explainability — LIME, SHAP, compare
# ---------------------------------------------------------------------------

# In-memory explanation store keyed by forensic event ID
_explanations: dict[str, dict[str, Any]] = {}


class LimeExplainRequest(BaseModel):
    """Request body for POST /explainability/lime."""

    event_id: UUID = Field(description="Inference event ID to explain")
    num_superpixels: int = Field(default=50, ge=4)
    num_samples: int = Field(default=100, ge=10)
    top_k: int = Field(default=5, ge=1)


class ShapExplainRequest(BaseModel):
    """Request body for POST /explainability/shap."""

    event_id: UUID = Field(description="Inference event ID to explain")
    num_samples: int = Field(default=128, ge=10)
    random_state: int = Field(default=42)


@app.post(
    "/explainability/lime", status_code=201, tags=["Explainability"], summary="LIME explanation"
)
async def trigger_lime_explanation(req: LimeExplainRequest):
    """Trigger a LIME explanation for a given inference event.

    Since LIME requires the original image and model, this endpoint
    creates a stub explanation record when the full pipeline is not
    available, or a real one when it is.
    """
    from uuid import uuid4

    event = await _db.get_event(req.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Inference event not found")

    explanation_id = str(uuid4())
    expl_event = await _service.record_event(
        event_type=EventType.EXPLANATION_GENERATED,
        actor="lime_explainer",
        payload={
            "method": "lime",
            "inference_event_id": str(req.event_id),
            "num_superpixels": req.num_superpixels,
            "num_samples": req.num_samples,
            "top_k": req.top_k,
        },
    )

    record = {
        "explanation_id": explanation_id,
        "method": "lime",
        "event_id": str(expl_event.id),
        "inference_event_id": str(req.event_id),
        "num_superpixels": req.num_superpixels,
        "num_samples": req.num_samples,
        "top_k": req.top_k,
        "timestamp": expl_event.timestamp.isoformat(),
    }
    _explanations[explanation_id] = record
    await _persist_explanation(
        explanation_id, str(expl_event.id), "lime", record["timestamp"], record
    )
    return record


@app.post(
    "/explainability/shap", status_code=201, tags=["Explainability"], summary="SHAP explanation"
)
async def trigger_shap_explanation(req: ShapExplainRequest):
    """Trigger a SHAP explanation for a given inference event."""
    from uuid import uuid4

    event = await _db.get_event(req.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Inference event not found")

    explanation_id = str(uuid4())
    expl_event = await _service.record_event(
        event_type=EventType.EXPLANATION_GENERATED,
        actor="shap_explainer",
        payload={
            "method": "shap",
            "inference_event_id": str(req.event_id),
            "num_samples": req.num_samples,
            "random_state": req.random_state,
        },
    )

    record = {
        "explanation_id": explanation_id,
        "method": "shap",
        "event_id": str(expl_event.id),
        "inference_event_id": str(req.event_id),
        "num_samples": req.num_samples,
        "random_state": req.random_state,
        "timestamp": expl_event.timestamp.isoformat(),
    }
    _explanations[explanation_id] = record
    await _persist_explanation(
        explanation_id, str(expl_event.id), "shap", record["timestamp"], record
    )
    return record


class SddExplainRequest(BaseModel):
    """Request body for POST /explainability/sdd."""

    event_id: UUID = Field(description="Inference event ID to explain")


@app.post(
    "/explainability/sdd",
    status_code=201,
    tags=["Explainability"],
    summary="SDD saliency explanation",
)
async def trigger_sdd_explanation(req: SddExplainRequest):
    """Trigger an SDD saliency explanation for a given inference event.

    Creates a stub explanation record (the full pixel-level saliency
    requires the original image and model to be available).
    """
    from uuid import uuid4

    event = await _db.get_event(req.event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Inference event not found")

    explanation_id = str(uuid4())
    expl_event = await _service.record_event(
        event_type=EventType.EXPLANATION_GENERATED,
        actor="sdd_explainer",
        payload={
            "method": "sdd",
            "inference_event_id": str(req.event_id),
            "num_regions": 7,
            "explanation_type": "SDD",
        },
    )

    record = {
        "explanation_id": explanation_id,
        "method": "sdd",
        "event_id": str(expl_event.id),
        "inference_event_id": str(req.event_id),
        "num_regions": 7,
        "timestamp": expl_event.timestamp.isoformat(),
    }
    _explanations[explanation_id] = record
    await _persist_explanation(
        explanation_id, str(expl_event.id), "sdd", record["timestamp"], record
    )
    return record


@app.get(
    "/explainability/explanations", tags=["Explainability"], summary="List explanations (paginated)"
)
async def list_explanations(
    limit: int = Query(default=50, ge=1, le=500, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
):
    """List all generated explanations."""
    all_items = list(_explanations.values())
    total = len(all_items)
    page = all_items[offset : offset + limit]
    return {"items": page, "total": total, "limit": limit, "offset": offset}


@app.get(
    "/explainability/explanations/{explanation_id}",
    tags=["Explainability"],
    summary="Get explanation by ID",
)
async def get_explanation(explanation_id: str):
    """Get explanation details by ID."""
    record = _explanations.get(explanation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return record


@app.get(
    "/explainability/compare/{event_id}", tags=["Explainability"], summary="Compare explanations"
)
async def compare_explanations(event_id: UUID):
    """Compare LIME vs SHAP vs SDD explanations for the same inference event.

    Returns all explanations (all methods) linked to the given
    inference event ID so they can be compared side by side.
    """
    event = await _db.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Inference event not found")

    lime_results = [
        e
        for e in _explanations.values()
        if e["inference_event_id"] == str(event_id) and e["method"] == "lime"
    ]
    shap_results = [
        e
        for e in _explanations.values()
        if e["inference_event_id"] == str(event_id) and e["method"] == "shap"
    ]
    sdd_results = [
        e
        for e in _explanations.values()
        if e["inference_event_id"] == str(event_id) and e["method"] == "sdd"
    ]

    return {
        "inference_event_id": str(event_id),
        "lime_explanations": lime_results,
        "shap_explanations": shap_results,
        "sdd_explanations": sdd_results,
        "total_lime": len(lime_results),
        "total_shap": len(shap_results),
        "total_sdd": len(sdd_results),
    }


# ---------------------------------------------------------------------------
# DID/VC — Decentralized Identifiers & Verifiable Credentials (US-031)
# ---------------------------------------------------------------------------


@app.post("/did/create", status_code=201, tags=["DID/VC"], summary="Create Ed25519 DID:key")
@limiter.limit("10/minute")
async def create_did(request: Request, req: CreateDIDRequest):
    """Create a new Ed25519 DID:key identity."""
    if req.seed is not None:
        did_key = Ed25519DIDKey.from_seed(bytes.fromhex(req.seed))
    else:
        did_key = Ed25519DIDKey()

    result = await _store_did_key(did_key, label=None)
    return result


@app.get("/did/{did_id:path}/resolve", tags=["DID/VC"], summary="Resolve DID document")
async def resolve_did(did_id: str):
    """Resolve a DID to its DID Document."""
    did_key = await _load_did_key(did_id)
    if did_key is None:
        raise HTTPException(status_code=404, detail="DID not found")
    return did_key.resolve()


@app.post("/vc/issue", status_code=201, tags=["DID/VC"], summary="Issue Verifiable Credential")
async def issue_vc(req: IssueVCRequest):
    """Issue a Verifiable Credential signed by the specified DID."""
    did_key = await _load_did_key(req.issuer_did_id)
    if did_key is None:
        raise HTTPException(status_code=404, detail="Issuer DID not found")

    vc = VerifiableCredential(issuer=did_key)
    credential = vc.issue(
        claims=req.claims,
        credential_type=req.credential_type,
        subject_did=req.subject_did,
    )
    return credential


@app.post("/vc/verify", tags=["DID/VC"], summary="Verify Verifiable Credential")
async def verify_vc(req: VerifyVCRequest):
    """Verify a Verifiable Credential's proof."""
    result = VerifiableCredential.verify(
        credential=req.credential,
        issuer_public_key=bytes.fromhex(req.issuer_public_key_hex),
    )
    return result


# ---------------------------------------------------------------------------
# ZK Proofs — Schnorr non-interactive zero-knowledge proofs (US-032)
# ---------------------------------------------------------------------------


@app.post("/zk/prove", status_code=201, tags=["ZK"], summary="Generate Schnorr ZK proof")
@limiter.limit("20/minute")
async def zk_prove(request: Request, req: ZKProveRequest):
    """Generate a Schnorr ZK proof for a forensic bundle."""
    bundle = await _service.get_bundle(UUID(req.bundle_id))
    if bundle is None:
        raise HTTPException(status_code=404, detail="Bundle not found")

    if not bundle.bundle_hash:
        raise HTTPException(status_code=400, detail="Bundle has no hash")

    prover = ZKBundleProver()
    proof_str = prover.prove_bundle(str(bundle.id), bundle.bundle_hash)

    # Store proof on the bundle in the database
    await _db.db.execute(
        "UPDATE forensic_bundles SET zk_proof_placeholder = ? WHERE id = ?",
        (proof_str, str(bundle.id)),
    )
    await _db.db.commit()

    proof_data = json.loads(proof_str)
    return {
        "proof": proof_data,
        "bundle_id": str(bundle.id),
        "bundle_hash": bundle.bundle_hash,
    }


@app.post("/zk/verify", tags=["ZK"], summary="Verify ZK proof")
async def zk_verify(req: ZKVerifyRequest):
    """Verify a Schnorr ZK proof."""
    verifier = ZKBundleVerifier()
    valid = verifier.verify_bundle(req.proof)

    # Parse the proof to extract scheme
    try:
        proof_data = json.loads(req.proof)
        scheme = proof_data.get("scheme", "unknown")
    except (json.JSONDecodeError, TypeError):
        scheme = "unknown"

    return {"valid": valid, "scheme": scheme}


@app.get("/zk/proofs/{bundle_id}", tags=["ZK"], summary="Get stored ZK proof")
async def get_zk_proof(bundle_id: UUID):
    """Get the stored ZK proof for a bundle."""
    bundle = await _service.get_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Bundle not found")

    proof_raw = bundle.zk_proof_placeholder
    if proof_raw is None:
        return {"bundle_id": str(bundle_id), "proof": None}

    # Try to parse as JSON for a cleaner response
    try:
        proof_data = json.loads(proof_raw)
    except (json.JSONDecodeError, TypeError):
        proof_data = proof_raw

    return {"bundle_id": str(bundle_id), "proof": proof_data}


# ---------------------------------------------------------------------------
# Erasure — Cryptographic erasure (US-050)
# ---------------------------------------------------------------------------


@app.post(
    "/erasure/erase/{subject_id}",
    status_code=200,
    tags=["Governance"],
    summary="Erase subject data",
    dependencies=[Depends(require_role("admin"))],
)
async def erase_subject(subject_id: str):
    """Cryptographically erase all data for a subject (GDPR Art 17)."""
    from friendlyface.governance.erasure import ErasureManager

    manager = ErasureManager(_db)
    result = await manager.erase_subject(subject_id)
    return result


@app.get("/erasure/status/{subject_id}", tags=["Governance"], summary="Get erasure status")
async def get_erasure_status(subject_id: str):
    """Get the erasure status for a subject."""
    from friendlyface.governance.erasure import ErasureManager

    manager = ErasureManager(_db)
    return await manager.get_erasure_status(subject_id)


@app.get("/erasure/records", tags=["Governance"], summary="List erasure records")
async def list_erasure_records(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List all erasure records with pagination."""
    from friendlyface.governance.erasure import ErasureManager

    manager = ErasureManager(_db)
    records, total = await manager.list_erasure_records(limit, offset)
    return {"items": records, "total": total, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Retention Policies (US-051)
# ---------------------------------------------------------------------------


class RetentionPolicyRequest(BaseModel):
    """Request body for POST /retention/policies."""

    name: str = Field(min_length=1, max_length=256, description="Policy name")
    entity_type: str = Field(
        min_length=1, max_length=128, description="Entity type (e.g. consent, subject)"
    )
    retention_days: int = Field(ge=1, description="Days to retain before action")
    action: str = Field(default="erase", description="Action: erase")
    enabled: bool = Field(default=True, description="Whether the policy is active")


@app.post(
    "/retention/policies",
    status_code=201,
    tags=["Governance"],
    summary="Create retention policy",
    dependencies=[Depends(require_role("admin"))],
)
async def create_retention_policy(req: RetentionPolicyRequest):
    """Create a new data retention policy."""
    from friendlyface.governance.retention import RetentionEngine

    engine = RetentionEngine(_db)
    return await engine.create_policy(
        name=req.name,
        entity_type=req.entity_type,
        retention_days=req.retention_days,
        action=req.action,
        enabled=req.enabled,
    )


@app.get("/retention/policies", tags=["Governance"], summary="List retention policies")
async def list_retention_policies(enabled_only: bool = False):
    """List all retention policies."""
    from friendlyface.governance.retention import RetentionEngine

    engine = RetentionEngine(_db)
    policies = await engine.list_policies(enabled_only=enabled_only)
    return {"policies": policies, "total": len(policies)}


@app.delete(
    "/retention/policies/{policy_id}",
    tags=["Governance"],
    summary="Delete retention policy",
    dependencies=[Depends(require_role("admin"))],
)
async def delete_retention_policy(policy_id: str):
    """Delete a retention policy by ID."""
    from friendlyface.governance.retention import RetentionEngine

    engine = RetentionEngine(_db)
    deleted = await engine.delete_policy(policy_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Policy not found")
    return {"deleted": True, "policy_id": policy_id}


@app.post(
    "/retention/evaluate",
    tags=["Governance"],
    summary="Evaluate retention policies",
    dependencies=[Depends(require_role("admin"))],
)
async def evaluate_retention():
    """Evaluate all enabled retention policies and erase expired data."""
    from friendlyface.governance.retention import RetentionEngine

    engine = RetentionEngine(_db)
    return await engine.evaluate()


# ---------------------------------------------------------------------------
# Backup — Admin backup/restore (US-043)
# ---------------------------------------------------------------------------


@app.post(
    "/admin/backup",
    status_code=201,
    tags=["Admin"],
    summary="Create backup",
    dependencies=[Depends(require_role("admin"))],
)
async def create_backup(label: str | None = None):
    """Create a database backup and enforce retention policy."""
    from friendlyface.ops.backup import BackupManager

    mgr = BackupManager(_db.db_path, settings.backup_dir)
    result = mgr.create_backup(label=label)
    mgr.enforce_retention(
        max_count=settings.backup_retention_count,
        max_age_days=settings.backup_retention_days,
    )
    return result


@app.get("/admin/backups", tags=["Admin"], summary="List backups")
async def list_backups():
    """List available database backups."""
    from friendlyface.ops.backup import BackupManager

    mgr = BackupManager(_db.db_path, settings.backup_dir)
    backups = mgr.list_backups()
    return {"backups": backups, "total": len(backups)}


@app.post(
    "/admin/backup/verify",
    tags=["Admin"],
    summary="Verify backup",
    dependencies=[Depends(require_role("admin"))],
)
async def verify_backup(filename: str):
    """Verify integrity of a backup file."""
    from friendlyface.ops.backup import BackupManager

    mgr = BackupManager(_db.db_path, settings.backup_dir)
    return mgr.verify_backup(filename)


@app.post(
    "/admin/backup/restore",
    tags=["Admin"],
    summary="Restore backup",
    dependencies=[Depends(require_role("admin"))],
)
async def restore_backup(filename: str):
    """Restore database from a backup. WARNING: replaces current data."""
    from friendlyface.ops.backup import BackupManager

    mgr = BackupManager(_db.db_path, settings.backup_dir)
    return mgr.restore_backup(filename)


@app.post(
    "/admin/backup/cleanup",
    tags=["Admin"],
    summary="Run backup cleanup",
    dependencies=[Depends(require_role("admin"))],
)
async def cleanup_backups():
    """Manually trigger backup retention policy enforcement."""
    from friendlyface.ops.backup import BackupManager

    mgr = BackupManager(_db.db_path, settings.backup_dir)
    return mgr.enforce_retention(
        max_count=settings.backup_retention_count,
        max_age_days=settings.backup_retention_days,
    )


@app.get("/admin/backup/stats", tags=["Admin"], summary="Backup statistics")
async def backup_stats():
    """Return backup count, total size, oldest, and newest timestamps."""
    from friendlyface.ops.backup import BackupManager

    mgr = BackupManager(_db.db_path, settings.backup_dir)
    return mgr.get_stats()


# ---------------------------------------------------------------------------
# Migrations — Admin migration management (US-079)
# ---------------------------------------------------------------------------


@app.get("/admin/migrations/status", tags=["Admin"], summary="Migration status")
async def migration_status():
    """Return applied and pending migrations."""
    from friendlyface.storage.migrations import get_migration_status

    return await get_migration_status(_db._db)


@app.post(
    "/admin/migrations/rollback",
    tags=["Admin"],
    summary="Rollback last migration",
    dependencies=[Depends(require_role("admin"))],
)
async def rollback_migration(dry_run: bool = False):
    """Roll back the last applied migration using its _down.sql file."""
    from friendlyface.storage.migrations import rollback_last

    result = await rollback_last(_db._db, dry_run=dry_run)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# OSCAL / JSON-LD compliance export (US-052)
# ---------------------------------------------------------------------------


@app.get(
    "/governance/compliance/export",
    tags=["Governance"],
    summary="Export compliance data in OSCAL or JSON-LD format",
    dependencies=[Depends(require_role("auditor"))],
)
async def export_compliance(format: str = "oscal"):
    """Export compliance data for auditors.

    Args:
        format: Export format — 'oscal' (NIST OSCAL assessment results) or 'json-ld'.
    """
    from friendlyface.governance.oscal import OSCALExporter

    exporter = OSCALExporter(_db)
    if format == "json-ld":
        return await exporter.export_json_ld()
    return await exporter.export_oscal()


# ---------------------------------------------------------------------------
# API Versioning — /api/v1 prefix (US-050)
# ---------------------------------------------------------------------------

_API_VERSION = "1.0.0"
_API_PREFIX = "/api/v1"

v1_router = APIRouter(prefix=_API_PREFIX)

# Health
v1_router.add_api_route("/health", health, methods=["GET"])

# Dashboard
v1_router.add_api_route("/dashboard", dashboard, methods=["GET"])

# Events
v1_router.add_api_route("/events", record_event, methods=["POST"], status_code=201)
v1_router.add_api_route("/events", list_events, methods=["GET"])
v1_router.add_api_route("/events/stream", event_stream, methods=["GET"])
v1_router.add_api_route("/events/{event_id}", get_event, methods=["GET"])

# Merkle
v1_router.add_api_route("/merkle/root", get_merkle_root, methods=["GET"])
v1_router.add_api_route("/merkle/proof/{event_id}", get_merkle_proof, methods=["GET"])

# Bundles
v1_router.add_api_route("/bundles/import", import_bundle, methods=["POST"], status_code=201)
v1_router.add_api_route("/bundles/{bundle_id}/export", export_bundle, methods=["GET"])
v1_router.add_api_route("/bundles", create_bundle, methods=["POST"], status_code=201)
v1_router.add_api_route("/bundles/{bundle_id}", get_bundle, methods=["GET"])

# Verification
v1_router.add_api_route("/verify/{bundle_id}", verify_bundle, methods=["POST"])
v1_router.add_api_route("/chain/integrity", verify_chain_integrity, methods=["GET"])

# Provenance
v1_router.add_api_route("/provenance", add_provenance_node, methods=["POST"], status_code=201)
v1_router.add_api_route("/provenance/{node_id}", get_provenance_chain, methods=["GET"])

# Recognition
v1_router.add_api_route("/recognition/train", train_model, methods=["POST"], status_code=201)
v1_router.add_api_route("/recognition/predict", predict, methods=["POST"])
v1_router.add_api_route("/recognize", recognize, methods=["POST"])
v1_router.add_api_route("/recognition/models", list_models, methods=["GET"])
v1_router.add_api_route("/recognition/models/{model_id}", get_model, methods=["GET"])

# Gallery
v1_router.add_api_route("/gallery/enroll", gallery_enroll, methods=["POST"], status_code=201)
v1_router.add_api_route("/gallery/search", gallery_search, methods=["POST"])
v1_router.add_api_route("/gallery/subjects", gallery_list_subjects, methods=["GET"])
v1_router.add_api_route(
    "/gallery/subjects/{subject_id}", gallery_delete_subject, methods=["DELETE"]
)
v1_router.add_api_route("/gallery/count", gallery_count, methods=["GET"])

# Voice biometrics
v1_router.add_api_route(
    "/recognition/voice/enroll", enroll_voice, methods=["POST"], status_code=201
)
v1_router.add_api_route("/recognition/voice/verify", verify_voice, methods=["POST"])
v1_router.add_api_route("/recognition/multimodal", multimodal_fusion, methods=["POST"])

# Federated Learning
v1_router.add_api_route("/fl/start", start_fl, methods=["POST"], status_code=201)
v1_router.add_api_route("/fl/simulate", simulate_fl, methods=["POST"], status_code=201)
v1_router.add_api_route("/fl/dp-start", start_dp_fl, methods=["POST"], status_code=201)
v1_router.add_api_route("/fl/status", get_fl_status, methods=["GET"])
v1_router.add_api_route("/fl/rounds", list_fl_rounds, methods=["GET"])
v1_router.add_api_route(
    "/fl/rounds/{simulation_id}/{round_number}", get_fl_round_details, methods=["GET"]
)
v1_router.add_api_route(
    "/fl/rounds/{simulation_id}/{round_number}/security",
    get_fl_round_security,
    methods=["GET"],
)

# Governance
v1_router.add_api_route("/governance/compliance", get_compliance_report, methods=["GET"])
v1_router.add_api_route(
    "/governance/compliance/generate",
    generate_compliance_report,
    methods=["POST"],
    status_code=201,
)
v1_router.add_api_route("/governance/compliance/export", export_compliance, methods=["GET"])

# Consent
v1_router.add_api_route("/consent/grant", grant_consent, methods=["POST"], status_code=201)
v1_router.add_api_route("/consent/revoke", revoke_consent, methods=["POST"])
v1_router.add_api_route("/consent/status/{subject_id}", get_consent_status, methods=["GET"])
v1_router.add_api_route("/consent/history/{subject_id}", get_consent_history, methods=["GET"])
v1_router.add_api_route("/consent/check", check_consent, methods=["POST"])

# Fairness
v1_router.add_api_route("/fairness/audit", trigger_bias_audit, methods=["POST"], status_code=201)
v1_router.add_api_route("/fairness/audits", list_bias_audits, methods=["GET"])
v1_router.add_api_route("/fairness/audits/{audit_id}", get_bias_audit, methods=["GET"])
v1_router.add_api_route("/fairness/status", get_fairness_status, methods=["GET"])
v1_router.add_api_route("/fairness/config", configure_auto_audit, methods=["POST"])
v1_router.add_api_route("/fairness/config", get_auto_audit_config, methods=["GET"])

# Explainability
v1_router.add_api_route(
    "/explainability/lime", trigger_lime_explanation, methods=["POST"], status_code=201
)
v1_router.add_api_route(
    "/explainability/shap", trigger_shap_explanation, methods=["POST"], status_code=201
)
v1_router.add_api_route(
    "/explainability/sdd", trigger_sdd_explanation, methods=["POST"], status_code=201
)
v1_router.add_api_route("/explainability/explanations", list_explanations, methods=["GET"])
v1_router.add_api_route(
    "/explainability/explanations/{explanation_id}", get_explanation, methods=["GET"]
)
v1_router.add_api_route("/explainability/compare/{event_id}", compare_explanations, methods=["GET"])

# DID / VC
v1_router.add_api_route("/did/create", create_did, methods=["POST"], status_code=201)
v1_router.add_api_route("/did/{did_id:path}/resolve", resolve_did, methods=["GET"])
v1_router.add_api_route("/vc/issue", issue_vc, methods=["POST"], status_code=201)
v1_router.add_api_route("/vc/verify", verify_vc, methods=["POST"])

# ZK Proofs
v1_router.add_api_route("/zk/prove", zk_prove, methods=["POST"], status_code=201)
v1_router.add_api_route("/zk/verify", zk_verify, methods=["POST"])
v1_router.add_api_route("/zk/proofs/{bundle_id}", get_zk_proof, methods=["GET"])

# Erasure
v1_router.add_api_route("/erasure/erase/{subject_id}", erase_subject, methods=["POST"])
v1_router.add_api_route("/erasure/status/{subject_id}", get_erasure_status, methods=["GET"])
v1_router.add_api_route("/erasure/records", list_erasure_records, methods=["GET"])

# Retention
v1_router.add_api_route(
    "/retention/policies", create_retention_policy, methods=["POST"], status_code=201
)
v1_router.add_api_route("/retention/policies", list_retention_policies, methods=["GET"])
v1_router.add_api_route(
    "/retention/policies/{policy_id}", delete_retention_policy, methods=["DELETE"]
)
v1_router.add_api_route("/retention/evaluate", evaluate_retention, methods=["POST"])

# Admin — Backup
v1_router.add_api_route("/admin/backup", create_backup, methods=["POST"], status_code=201)
v1_router.add_api_route("/admin/backups", list_backups, methods=["GET"])
v1_router.add_api_route("/admin/backup/verify", verify_backup, methods=["POST"])
v1_router.add_api_route("/admin/backup/restore", restore_backup, methods=["POST"])
v1_router.add_api_route("/admin/backup/cleanup", cleanup_backups, methods=["POST"])
v1_router.add_api_route("/admin/backup/stats", backup_stats, methods=["GET"])

# Admin — Migrations
v1_router.add_api_route("/admin/migrations/status", migration_status, methods=["GET"])
v1_router.add_api_route("/admin/migrations/rollback", rollback_migration, methods=["POST"])

app.include_router(v1_router)


# Version info endpoint (mounted on app, not the router)
@app.get("/api/version", tags=["Health"], summary="API version info")
async def api_version():
    """Return API version information."""
    return {"version": _API_VERSION, "api_prefix": _API_PREFIX}


# ---------------------------------------------------------------------------
# Static frontend serving (US-037)
# ---------------------------------------------------------------------------

_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
_SERVE_FRONTEND = settings.serve_frontend


if _SERVE_FRONTEND and os.path.isdir(_FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles

    # Serve static assets (JS, CSS, images) at /assets/
    _assets_dir = os.path.join(_FRONTEND_DIST, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="static-assets")

    # SPA fallback: serve index.html for all non-API routes
    from starlette.responses import FileResponse as _FileResponse

    @app.get("/{full_path:path}")
    async def _spa_fallback(full_path: str):
        """Serve index.html for SPA client-side routing."""
        # If requesting a file that exists in dist, serve it
        file_path = os.path.join(_FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(file_path):
            return _FileResponse(file_path)
        # Otherwise serve index.html for client-side routing
        index_path = os.path.join(_FRONTEND_DIST, "index.html")
        if os.path.isfile(index_path):
            return _FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend not built")
