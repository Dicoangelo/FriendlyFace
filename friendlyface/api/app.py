"""FastAPI application for the FriendlyFace Blockchain Forensic Layer.

Endpoints:
  POST   /events                  — Record a forensic event
  GET    /events                  — List all events
  GET    /events/{id}             — Get event by ID
  GET    /merkle/root             — Get current Merkle root
  GET    /merkle/proof/{event_id} — Get Merkle inclusion proof
  POST   /bundles                 — Create forensic bundle
  GET    /bundles/{id}            — Get bundle by ID
  POST   /verify/{bundle_id}     — Verify a forensic bundle
  GET    /chain/integrity         — Verify full hash chain
  POST   /provenance              — Add provenance node
  GET    /provenance/{node_id}    — Get provenance chain
  GET    /health                  — Health check
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from friendlyface.core.models import (
    BiasAuditRecord,
    EventType,
    ProvenanceRelation,
)
from friendlyface.core.service import ForensicService
from friendlyface.storage.database import Database

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


class AddProvenanceRequest(BaseModel):
    entity_type: str
    entity_id: str
    parents: list[UUID] = Field(default_factory=list)
    relations: list[ProvenanceRelation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_db = Database()
_service = ForensicService(_db)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
)


def get_service() -> ForensicService:
    return _service


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    count = await _db.get_event_count()
    return {
        "status": "ok",
        "version": "0.1.0",
        "event_count": count,
        "merkle_root": _service.get_merkle_root(),
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
