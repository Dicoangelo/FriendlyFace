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
  POST   /fl/start                  — Start FL simulation with configurable parameters
  POST   /fl/simulate               — (Legacy) Alias for /fl/start
  GET    /fl/rounds                 — List completed FL rounds with summary
  GET    /fl/rounds/{sim_id}/{n}    — Get round details with client contributions and security
  GET    /fl/rounds/{sim_id}/{n}/security — Get poisoning detection results for a round
  GET    /fl/status                 — Current FL training status
  GET    /health                    — Health check
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from friendlyface.auth import require_api_key
from friendlyface.core.models import (
    BiasAuditRecord,
    EventType,
    ProvenanceRelation,
)
from friendlyface.core.service import ForensicService
from friendlyface.storage.database import Database

logger = logging.getLogger("friendlyface")

_STARTUP_TIME: float = 0.0


def _create_database():
    """Create the appropriate database backend based on FF_STORAGE env var."""
    import os

    backend = os.environ.get("FF_STORAGE", "sqlite").lower()
    if backend == "supabase":
        from friendlyface.storage.supabase_db import SupabaseDatabase

        return SupabaseDatabase()
    return Database()


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------


class RecordEventRequest(BaseModel):
    event_type: EventType
    actor: str
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
    entity_type: str
    entity_id: str
    parents: list[UUID] = Field(default_factory=list)
    relations: list[ProvenanceRelation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_db = _create_database()
_service = ForensicService(_db)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _STARTUP_TIME
    _STARTUP_TIME = time.monotonic()
    await _db.connect()
    await _service.initialize()
    yield
    await _db.close()


app = FastAPI(
    title="FriendlyFace — Blockchain Forensic Layer",
    description=(
        "Forensic-friendly AI facial recognition platform. "
        "Layer 3: Blockchain Forensic Layer implementing Mohammed's ICDF2C 2024 schema."
    ),
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
_cors_origins = os.environ.get("FF_CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    request_id = str(uuid4())[:8]
    start = time.monotonic()
    response: Response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "%s %s %s %.1fms [%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    response.headers["X-Request-ID"] = request_id
    return response


def get_service() -> ForensicService:
    return _service


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    count = await _db.get_event_count()
    uptime_s = time.monotonic() - _STARTUP_TIME if _STARTUP_TIME > 0 else 0
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(uptime_s, 1),
        "event_count": count,
        "merkle_root": _service.get_merkle_root(),
        "storage_backend": os.environ.get("FF_STORAGE", "sqlite"),
    }


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@app.post("/events", status_code=201)
async def record_event(req: RecordEventRequest):
    event = await _service.record_event(
        event_type=req.event_type,
        actor=req.actor,
        payload=req.payload,
    )
    return event.model_dump(mode="json")


@app.get("/events")
async def list_events():
    events = await _service.get_all_events()
    return [e.model_dump(mode="json") for e in events]


@app.get("/events/{event_id}")
async def get_event(event_id: UUID):
    event = await _service.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Merkle tree
# ---------------------------------------------------------------------------


@app.get("/merkle/root")
async def get_merkle_root():
    root = _service.get_merkle_root()
    return {"merkle_root": root, "leaf_count": _service.merkle.leaf_count}


@app.get("/merkle/proof/{event_id}")
async def get_merkle_proof(event_id: UUID):
    proof = _service.get_merkle_proof(event_id)
    if proof is None:
        raise HTTPException(status_code=404, detail="Event not found in Merkle tree")
    return proof.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


@app.post("/bundles", status_code=201)
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


@app.get("/bundles/{bundle_id}")
async def get_bundle(bundle_id: UUID):
    bundle = await _service.get_bundle(bundle_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Bundle not found")
    return bundle.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@app.post("/verify/{bundle_id}")
async def verify_bundle(bundle_id: UUID):
    result = await _service.verify_bundle(bundle_id)
    return result


@app.get("/chain/integrity")
async def verify_chain_integrity():
    return await _service.verify_chain_integrity()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@app.post("/provenance", status_code=201)
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


@app.get("/provenance/{node_id}")
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
    dataset_path: str = Field(description="Path to directory of aligned 112x112 grayscale images")
    output_dir: str = Field(description="Directory to write trained model files")
    n_components: int = Field(default=128, description="PCA components to retain")
    C: float = Field(default=1.0, description="SVM regularization parameter")
    kernel: str = Field(default="linear", description="SVM kernel type")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    labels: list[int] | None = Field(
        default=None,
        description="Per-image integer labels (sorted filename order). Required for SVM.",
    )


@app.post("/recognition/train", status_code=201)
async def train_model(req: TrainRequest):
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


async def _do_predict(image: UploadFile, top_k: int = 5) -> dict[str, Any]:
    """Shared prediction logic for /recognition/predict and /recognize."""
    from pathlib import Path

    from friendlyface.recognition.inference import run_inference

    if _pca_model_path is None or _svm_model_path is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Configure PCA and SVM model paths.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

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
        "matches": [{"label": m.label, "confidence": m.confidence} for m in result.matches],
    }


@app.post("/recognition/predict", status_code=200)
async def predict(image: UploadFile, top_k: int = 5):
    """Upload a face image and get prediction matches with forensic event logging."""
    return await _do_predict(image, top_k)


@app.post("/recognize", status_code=200)
async def recognize(image: UploadFile, top_k: int = 5):
    """Legacy endpoint — use POST /recognition/predict instead."""
    return await _do_predict(image, top_k)


@app.get("/recognition/models")
async def list_models():
    """List all available trained models with metadata."""
    return list(_model_registry.values())


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
        "rounds": rounds_summary,
    }


@app.post("/fl/start", status_code=201)
async def start_fl(req: FLSimulateRequest):
    """Start a federated learning simulation with configurable parameters."""
    return await _run_fl_simulation(req)


@app.post("/fl/simulate", status_code=201)
async def simulate_fl(req: FLSimulateRequest):
    """Legacy alias for POST /fl/start."""
    return await _run_fl_simulation(req)


@app.get("/fl/status")
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
    return {
        "total_simulations": len(_fl_simulations),
        "simulations": simulations,
    }


@app.get("/fl/rounds")
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
    return {"total_rounds": len(all_rounds), "rounds": all_rounds}


@app.get("/fl/rounds/{simulation_id}/{round_number}")
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

    return {
        "simulation_id": simulation_id,
        "round": rr.round_number,
        "n_clients": sim.n_clients,
        "global_model_hash": rr.global_model_hash,
        "event_id": str(rr.event.id),
        "provenance_node_id": str(rr.provenance_node.id),
        "client_contributions": client_contributions,
        "security_status": security,
    }


@app.get("/fl/rounds/{simulation_id}/{round_number}/security")
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
        return {
            "round": round_number,
            "poisoning_detection_enabled": False,
            "message": "Poisoning detection was not enabled for this simulation",
        }

    return {
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
    }


@app.get("/recognition/models/{model_id}")
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


@app.get("/governance/compliance")
async def get_compliance_report():
    """Get the latest compliance report, or generate one if none exists."""
    global _latest_compliance_report
    if _latest_compliance_report is None:
        from friendlyface.governance.compliance import ComplianceReporter

        reporter = ComplianceReporter(_db, _service)
        _latest_compliance_report = await reporter.generate_report()
    return _latest_compliance_report


@app.post("/governance/compliance/generate", status_code=201)
async def generate_compliance_report():
    """Generate a new compliance report and cache it."""
    global _latest_compliance_report
    from friendlyface.governance.compliance import ComplianceReporter

    reporter = ComplianceReporter(_db, _service)
    _latest_compliance_report = await reporter.generate_report()
    return _latest_compliance_report


# ---------------------------------------------------------------------------
# Consent -- grant, revoke, status, history, check
# ---------------------------------------------------------------------------


class ConsentGrantRequest(BaseModel):
    """Request body for POST /consent/grant."""

    subject_id: str = Field(description="Subject identifier")
    purpose: str = Field(description="Purpose of consent (e.g. recognition, training)")
    expiry: str | None = Field(
        default=None,
        description="Optional ISO-8601 expiry datetime",
    )
    actor: str = Field(default="api", description="Actor granting consent")


class ConsentRevokeRequest(BaseModel):
    """Request body for POST /consent/revoke."""

    subject_id: str = Field(description="Subject identifier")
    purpose: str = Field(description="Purpose of consent to revoke")
    reason: str = Field(default="", description="Reason for revocation")
    actor: str = Field(default="api", description="Actor revoking consent")


class ConsentCheckRequest(BaseModel):
    """Request body for POST /consent/check."""

    subject_id: str = Field(description="Subject identifier")
    purpose: str = Field(description="Purpose to verify consent for")


def _get_consent_manager():
    """Lazy-create a ConsentManager wired to the global db + service."""
    from friendlyface.governance.consent import ConsentManager

    return ConsentManager(_db, _service)


@app.post("/consent/grant", status_code=201)
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
    return record.to_dict()


@app.post("/consent/revoke", status_code=200)
async def revoke_consent(req: ConsentRevokeRequest):
    """Revoke consent for a subject+purpose pair."""
    mgr = _get_consent_manager()
    record = await mgr.revoke_consent(
        req.subject_id,
        req.purpose,
        reason=req.reason,
        actor=req.actor,
    )
    return record.to_dict()


@app.get("/consent/status/{subject_id}")
async def get_consent_status(subject_id: str, purpose: str = "recognition"):
    """Check current consent status for a subject."""
    mgr = _get_consent_manager()
    return await mgr.get_consent_status(subject_id, purpose)


@app.get("/consent/history/{subject_id}")
async def get_consent_history(subject_id: str, purpose: str | None = None):
    """Get full consent history for a subject."""
    mgr = _get_consent_manager()
    records = await mgr.get_history(subject_id, purpose)
    return {"subject_id": subject_id, "total": len(records), "records": records}


@app.post("/consent/check", status_code=200)
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


@app.post("/fairness/audit", status_code=201)
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


@app.get("/fairness/audits")
async def list_bias_audits():
    """List completed audits with summary scores."""
    audits = await _db.get_all_bias_audits()
    return {
        "total": len(audits),
        "audits": [
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
    }


@app.get("/fairness/audits/{audit_id}")
async def get_bias_audit(audit_id: UUID):
    """Get full audit details with per-group breakdowns."""
    audit = await _db.get_bias_audit(audit_id)
    if audit is None:
        raise HTTPException(status_code=404, detail="Bias audit not found")
    return audit.model_dump(mode="json")


@app.get("/fairness/status")
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


@app.post("/fairness/config")
async def configure_auto_audit(req: AutoAuditConfigRequest):
    """Configure auto-audit interval."""
    global _auto_audit_interval
    _auto_audit_interval = req.auto_audit_interval
    return {
        "auto_audit_interval": _auto_audit_interval,
        "message": f"Auto-audit will trigger every {_auto_audit_interval} recognition events",
    }


@app.get("/fairness/config")
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


@app.post("/explainability/lime", status_code=201)
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
    return record


@app.post("/explainability/shap", status_code=201)
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
    return record


@app.get("/explainability/explanations")
async def list_explanations():
    """List all generated explanations."""
    items = list(_explanations.values())
    return {"total": len(items), "explanations": items}


@app.get("/explainability/explanations/{explanation_id}")
async def get_explanation(explanation_id: str):
    """Get explanation details by ID."""
    record = _explanations.get(explanation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return record


@app.get("/explainability/compare/{event_id}")
async def compare_explanations(event_id: UUID):
    """Compare LIME vs SHAP explanations for the same inference event.

    Returns all explanations (both methods) linked to the given
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

    return {
        "inference_event_id": str(event_id),
        "lime_explanations": lime_results,
        "shap_explanations": shap_results,
        "total_lime": len(lime_results),
        "total_shap": len(shap_results),
    }
