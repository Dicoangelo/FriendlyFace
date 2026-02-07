"""Tests for input validation constraints on Pydantic request models (US-053)."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_record_event_actor_too_long(client: AsyncClient):
    """Test that RecordEventRequest rejects actor strings exceeding 512 chars."""
    payload = {
        "event_type": "training_complete",
        "actor": "a" * 513,  # Exceeds max_length=512
        "payload": {},
    }
    response = await client.post("/events", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("actor" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_consent_grant_subject_id_too_long(client: AsyncClient):
    """Test that ConsentGrantRequest rejects subject_id strings exceeding 256 chars."""
    payload = {
        "subject_id": "s" * 257,  # Exceeds max_length=256
        "purpose": "recognition",
        "actor": "test",
    }
    response = await client.post("/consent/grant", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("subject_id" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_consent_grant_purpose_too_long(client: AsyncClient):
    """Test that ConsentGrantRequest rejects purpose strings exceeding 512 chars."""
    payload = {
        "subject_id": "user123",
        "purpose": "p" * 513,  # Exceeds max_length=512
        "actor": "test",
    }
    response = await client.post("/consent/grant", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("purpose" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_train_request_dataset_path_too_long(client: AsyncClient):
    """Test that TrainRequest rejects dataset_path strings exceeding 4096 chars."""
    payload = {
        "dataset_path": "/path/" + "a" * 4100,  # Exceeds max_length=4096
        "output_dir": "/output",
        "labels": [0, 1],
    }
    response = await client.post("/recognition/train", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("dataset_path" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_zk_verify_proof_too_long(client: AsyncClient):
    """Test that ZKVerifyRequest rejects proof strings exceeding 65536 chars."""
    payload = {
        "proof": '{"data": "' + "x" * 70000 + '"}',  # Exceeds max_length=65536
    }
    response = await client.post("/zk/verify", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("proof" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_consent_grant_subject_id_empty(client: AsyncClient):
    """Test that ConsentGrantRequest rejects empty subject_id (min_length=1)."""
    payload = {
        "subject_id": "",  # Violates min_length=1
        "purpose": "recognition",
        "actor": "test",
    }
    response = await client.post("/consent/grant", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("subject_id" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_consent_grant_purpose_empty(client: AsyncClient):
    """Test that ConsentGrantRequest rejects empty purpose (min_length=1)."""
    payload = {
        "subject_id": "user123",
        "purpose": "",  # Violates min_length=1
        "actor": "test",
    }
    response = await client.post("/consent/grant", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("purpose" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_record_event_actor_empty(client: AsyncClient):
    """Test that RecordEventRequest rejects empty actor (min_length=1)."""
    payload = {
        "event_type": "training_complete",
        "actor": "",  # Violates min_length=1
        "payload": {},
    }
    response = await client.post("/events", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("actor" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_provenance_entity_type_too_long(client: AsyncClient):
    """Test that AddProvenanceRequest rejects entity_type exceeding 256 chars."""
    payload = {
        "entity_type": "t" * 257,  # Exceeds max_length=256
        "entity_id": "entity123",
    }
    response = await client.post("/provenance", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("entity_type" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_provenance_entity_id_empty(client: AsyncClient):
    """Test that AddProvenanceRequest rejects empty entity_id (min_length=1)."""
    payload = {
        "entity_type": "model",
        "entity_id": "",  # Violates min_length=1
    }
    response = await client.post("/provenance", json=payload)
    assert response.status_code == 422
    error_detail = response.json()["detail"]
    assert any("entity_id" in str(err).lower() for err in error_detail)


@pytest.mark.asyncio
async def test_valid_consent_grant_request(client: AsyncClient):
    """Test that valid ConsentGrantRequest with proper lengths is accepted."""
    payload = {
        "subject_id": "user123",
        "purpose": "recognition",
        "actor": "test_actor",
    }
    response = await client.post("/consent/grant", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["subject_id"] == "user123"
    assert data["purpose"] == "recognition"


@pytest.mark.asyncio
async def test_valid_record_event_request(client: AsyncClient):
    """Test that valid RecordEventRequest with proper lengths is accepted."""
    payload = {
        "event_type": "training_complete",
        "actor": "test_actor",
        "payload": {"test": "data"},
    }
    response = await client.post("/events", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["actor"] == "test_actor"
    assert data["event_type"] == "training_complete"
