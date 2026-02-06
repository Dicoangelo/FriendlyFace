"""Tests for US-009: forensic bundle full-layer artifacts.

Covers:
- ForensicBundle model fields (recognition_artifacts, fl_artifacts, bias_report, explanation_artifacts)
- Bundle hash covers all layer artifacts
- ForensicService.create_bundle auto-collects artifacts from events
- Layer filter support
- Bundle verification checks integrity of layer artifacts
- API endpoint with layer filters
- Storage round-trip
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from friendlyface.core.models import (
    BundleStatus,
    EventType,
    ForensicBundle,
)


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------


class TestForensicBundleArtifactFields:
    """ForensicBundle model extended with optional layer artifact fields."""

    def test_default_artifact_fields_are_none(self):
        bundle = ForensicBundle(event_ids=[uuid4()])
        assert bundle.recognition_artifacts is None
        assert bundle.fl_artifacts is None
        assert bundle.bias_report is None
        assert bundle.explanation_artifacts is None

    def test_set_recognition_artifacts(self):
        arts = {"inference_events": [{"event_id": str(uuid4())}]}
        bundle = ForensicBundle(event_ids=[uuid4()], recognition_artifacts=arts)
        assert bundle.recognition_artifacts == arts

    def test_set_fl_artifacts(self):
        arts = {"rounds": [{"round": 1}], "security_alerts": []}
        bundle = ForensicBundle(event_ids=[uuid4()], fl_artifacts=arts)
        assert bundle.fl_artifacts == arts

    def test_set_bias_report(self):
        report = {"audits": [{"demographic_parity_gap": 0.05}]}
        bundle = ForensicBundle(event_ids=[uuid4()], bias_report=report)
        assert bundle.bias_report == report

    def test_set_explanation_artifacts(self):
        arts = {"explanations": [{"method": "LIME"}]}
        bundle = ForensicBundle(event_ids=[uuid4()], explanation_artifacts=arts)
        assert bundle.explanation_artifacts == arts

    def test_hash_covers_artifacts(self):
        """Bundle hash must change when any artifact field changes."""
        eid = uuid4()
        base = ForensicBundle(event_ids=[eid]).seal()

        with_recognition = ForensicBundle(
            id=base.id,
            created_at=base.created_at,
            event_ids=[eid],
            recognition_artifacts={"data": "test"},
        ).seal()
        assert base.bundle_hash != with_recognition.bundle_hash

    def test_seal_and_verify_with_artifacts(self):
        bundle = ForensicBundle(
            event_ids=[uuid4()],
            recognition_artifacts={"inference_events": []},
            fl_artifacts={"rounds": []},
            bias_report={"audits": []},
            explanation_artifacts={"explanations": []},
        ).seal()
        assert bundle.verify()
        assert bundle.status == BundleStatus.COMPLETE

    def test_tamper_artifact_detected(self):
        bundle = ForensicBundle(
            event_ids=[uuid4()],
            recognition_artifacts={"inference_events": [{"score": 0.9}]},
        ).seal()
        assert bundle.verify()
        # tamper
        bundle.recognition_artifacts["inference_events"][0]["score"] = 0.1
        assert not bundle.verify()


# ---------------------------------------------------------------------------
# Service-level tests
# ---------------------------------------------------------------------------


class TestCreateBundleWithArtifacts:
    """ForensicService.create_bundle auto-collects artifacts from events."""

    @pytest.mark.asyncio
    async def test_auto_collect_recognition(self, service):
        """Inference events are auto-collected into recognition_artifacts."""
        e = await service.record_event(
            EventType.INFERENCE_RESULT,
            actor="model_v1",
            payload={"score": 0.95, "label": "authorized"},
        )
        bundle = await service.create_bundle(event_ids=[e.id])
        assert bundle.recognition_artifacts is not None
        assert len(bundle.recognition_artifacts["inference_events"]) == 1
        assert bundle.recognition_artifacts["inference_events"][0]["event_id"] == str(e.id)

    @pytest.mark.asyncio
    async def test_auto_collect_fl(self, service):
        """FL round events are auto-collected into fl_artifacts."""
        e = await service.record_event(
            EventType.FL_ROUND,
            actor="fl_coordinator",
            payload={"round": 1, "participants": 5},
        )
        bundle = await service.create_bundle(event_ids=[e.id])
        assert bundle.fl_artifacts is not None
        assert len(bundle.fl_artifacts["rounds"]) == 1

    @pytest.mark.asyncio
    async def test_auto_collect_bias(self, service):
        """Bias audit events are auto-collected into bias_report."""
        e = await service.record_event(
            EventType.BIAS_AUDIT,
            actor="auditor",
            payload={"demographic_parity_gap": 0.02},
        )
        bundle = await service.create_bundle(event_ids=[e.id])
        assert bundle.bias_report is not None
        assert len(bundle.bias_report["audits"]) == 1

    @pytest.mark.asyncio
    async def test_auto_collect_explanation(self, service):
        """Explanation events are auto-collected into explanation_artifacts."""
        e = await service.record_event(
            EventType.EXPLANATION_GENERATED,
            actor="lime_explainer",
            payload={"method": "LIME", "top_features": ["pixel_42"]},
        )
        bundle = await service.create_bundle(event_ids=[e.id])
        assert bundle.explanation_artifacts is not None
        assert len(bundle.explanation_artifacts["explanations"]) == 1

    @pytest.mark.asyncio
    async def test_explicit_artifacts_override_auto(self, service):
        """Explicitly provided artifacts are used instead of auto-collection."""
        e = await service.record_event(
            EventType.INFERENCE_RESULT,
            actor="model_v1",
            payload={"score": 0.8},
        )
        custom = {"custom_key": "custom_value"}
        bundle = await service.create_bundle(
            event_ids=[e.id],
            recognition_artifacts=custom,
        )
        assert bundle.recognition_artifacts == custom

    @pytest.mark.asyncio
    async def test_no_matching_events_yields_none(self, service):
        """Events with no matching type leave artifact field as None."""
        e = await service.record_event(
            EventType.CONSENT_RECORDED,
            actor="consent_manager",
            payload={"subject": "user_1"},
        )
        bundle = await service.create_bundle(event_ids=[e.id])
        assert bundle.recognition_artifacts is None
        assert bundle.fl_artifacts is None
        assert bundle.bias_report is None
        assert bundle.explanation_artifacts is None


class TestCreateBundleWithLayerFilters:
    """Layer filters control which artifacts are included."""

    @pytest.mark.asyncio
    async def test_filter_recognition_only(self, service):
        """Only recognition artifacts when filter=["recognition"]."""
        e1 = await service.record_event(
            EventType.INFERENCE_RESULT, actor="model", payload={"score": 0.9}
        )
        e2 = await service.record_event(
            EventType.FL_ROUND, actor="fl", payload={"round": 1}
        )
        bundle = await service.create_bundle(
            event_ids=[e1.id, e2.id],
            layer_filters=["recognition"],
        )
        assert bundle.recognition_artifacts is not None
        assert bundle.fl_artifacts is None
        assert bundle.bias_report is None
        assert bundle.explanation_artifacts is None

    @pytest.mark.asyncio
    async def test_filter_multiple_layers(self, service):
        """Multiple filters include only specified layers."""
        e1 = await service.record_event(
            EventType.INFERENCE_RESULT, actor="model", payload={"score": 0.9}
        )
        e2 = await service.record_event(
            EventType.EXPLANATION_GENERATED, actor="lime", payload={"method": "LIME"}
        )
        bundle = await service.create_bundle(
            event_ids=[e1.id, e2.id],
            layer_filters=["recognition", "explanation"],
        )
        assert bundle.recognition_artifacts is not None
        assert bundle.explanation_artifacts is not None
        assert bundle.fl_artifacts is None
        assert bundle.bias_report is None

    @pytest.mark.asyncio
    async def test_empty_filter_excludes_all(self, service):
        """Empty filter list means no layer artifacts are collected."""
        e = await service.record_event(
            EventType.INFERENCE_RESULT, actor="model", payload={"score": 0.9}
        )
        bundle = await service.create_bundle(
            event_ids=[e.id],
            layer_filters=[],
        )
        assert bundle.recognition_artifacts is None
        assert bundle.fl_artifacts is None
        assert bundle.bias_report is None
        assert bundle.explanation_artifacts is None


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------


class TestVerifyBundleWithArtifacts:
    """Bundle verification includes layer artifact integrity checks."""

    @pytest.mark.asyncio
    async def test_verify_valid_bundle_with_artifacts(self, service):
        """A correctly created bundle with artifacts verifies successfully."""
        e = await service.record_event(
            EventType.INFERENCE_RESULT,
            actor="model_v1",
            payload={"score": 0.95},
        )
        bundle = await service.create_bundle(event_ids=[e.id])
        result = await service.verify_bundle(bundle.id)
        assert result["valid"] is True
        assert result["status"] == "verified"
        # layer artifacts should be checked
        if "layer_artifacts" in result and "recognition" in result["layer_artifacts"]:
            assert result["layer_artifacts"]["recognition"]["valid"] is True

    @pytest.mark.asyncio
    async def test_verify_detects_tampered_artifact_hash(self, service, db):
        """If an artifact references a wrong event_hash, verification fails."""
        e = await service.record_event(
            EventType.INFERENCE_RESULT,
            actor="model_v1",
            payload={"score": 0.95},
        )
        # Create bundle with tampered recognition artifact event_hash
        tampered_arts = {
            "inference_events": [
                {
                    "event_id": str(e.id),
                    "event_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                    "payload": e.payload,
                }
            ]
        }
        bundle = await service.create_bundle(
            event_ids=[e.id],
            recognition_artifacts=tampered_arts,
        )
        result = await service.verify_bundle(bundle.id)
        # The bundle hash itself should be valid (it includes the tampered data),
        # but the layer artifact check should detect the mismatch
        assert result["layer_artifacts"]["recognition"]["valid"] is False


# ---------------------------------------------------------------------------
# Full integration test
# ---------------------------------------------------------------------------


class TestFullLayerIntegration:
    """Integration: create events across layers -> bundle -> verify all artifacts."""

    @pytest.mark.asyncio
    async def test_cross_layer_event_chain_bundle_verify(self, service):
        """Record events across all layers, bundle them, and verify integrity."""
        # Recognition events
        e_train = await service.record_event(
            EventType.TRAINING_START,
            actor="trainer",
            payload={"model": "resnet50", "dataset": "lfw"},
        )
        e_inference = await service.record_event(
            EventType.INFERENCE_RESULT,
            actor="model_v1",
            payload={"score": 0.97, "label": "match"},
        )

        # FL event
        e_fl = await service.record_event(
            EventType.FL_ROUND,
            actor="fl_coordinator",
            payload={"round": 3, "participants": 10, "accuracy": 0.94},
        )

        # Bias audit event
        e_bias = await service.record_event(
            EventType.BIAS_AUDIT,
            actor="auditor",
            payload={
                "demographic_parity_gap": 0.03,
                "equalized_odds_gap": 0.02,
                "compliant": True,
            },
        )

        # Explanation event
        e_explain = await service.record_event(
            EventType.EXPLANATION_GENERATED,
            actor="lime_explainer",
            payload={
                "method": "LIME",
                "top_features": ["pixel_100", "pixel_42"],
                "model_prediction": 0.97,
            },
        )

        all_ids = [e_train.id, e_inference.id, e_fl.id, e_bias.id, e_explain.id]

        # Create bundle -- auto-collects all layer artifacts
        bundle = await service.create_bundle(event_ids=all_ids)

        # Verify all artifact fields are populated
        assert bundle.recognition_artifacts is not None
        assert len(bundle.recognition_artifacts["training_events"]) == 1
        assert len(bundle.recognition_artifacts["inference_events"]) == 1
        assert bundle.fl_artifacts is not None
        assert len(bundle.fl_artifacts["rounds"]) == 1
        assert bundle.bias_report is not None
        assert len(bundle.bias_report["audits"]) == 1
        assert bundle.explanation_artifacts is not None
        assert len(bundle.explanation_artifacts["explanations"]) == 1

        # Bundle should be sealed and verifiable
        assert bundle.status == BundleStatus.COMPLETE
        assert bundle.verify()

        # Full verification via service
        result = await service.verify_bundle(bundle.id)
        assert result["valid"] is True
        assert result["status"] == "verified"
        assert result["bundle_hash_valid"] is True

        # Layer artifact integrity checks
        layer_arts = result.get("layer_artifacts", {})
        for layer_name in ("recognition", "fl", "bias", "explanation"):
            assert layer_name in layer_arts, f"Missing layer: {layer_name}"
            assert layer_arts[layer_name]["valid"] is True


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestBundleAPI:
    """API POST /bundles with layer filter support."""

    @pytest.mark.asyncio
    async def test_create_bundle_via_api(self, client):
        """POST /bundles creates a bundle and returns artifact fields."""
        # Create events first
        resp = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model_v1",
                "payload": {"score": 0.9},
            },
        )
        assert resp.status_code == 201
        event_id = resp.json()["id"]

        # Create bundle
        resp = await client.post(
            "/bundles",
            json={"event_ids": [event_id]},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "recognition_artifacts" in data
        assert data["recognition_artifacts"] is not None

    @pytest.mark.asyncio
    async def test_create_bundle_with_layer_filter_api(self, client):
        """POST /bundles with layer_filters limits included artifacts."""
        resp = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model_v1",
                "payload": {"score": 0.9},
            },
        )
        event_id = resp.json()["id"]

        resp = await client.post(
            "/bundles",
            json={
                "event_ids": [event_id],
                "layer_filters": ["recognition"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["recognition_artifacts"] is not None
        assert data["fl_artifacts"] is None
        assert data["bias_report"] is None
        assert data["explanation_artifacts"] is None

    @pytest.mark.asyncio
    async def test_verify_bundle_via_api(self, client):
        """POST /verify/{bundle_id} returns valid result with layer checks."""
        # Create event
        resp = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model_v1",
                "payload": {"score": 0.9},
            },
        )
        event_id = resp.json()["id"]

        # Create bundle
        resp = await client.post(
            "/bundles",
            json={"event_ids": [event_id]},
        )
        bundle_id = resp.json()["id"]

        # Verify
        resp = await client.post(f"/verify/{bundle_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True

    @pytest.mark.asyncio
    async def test_get_bundle_includes_artifacts(self, client):
        """GET /bundles/{id} includes artifact fields."""
        resp = await client.post(
            "/events",
            json={
                "event_type": "explanation_generated",
                "actor": "lime",
                "payload": {"method": "LIME"},
            },
        )
        event_id = resp.json()["id"]

        resp = await client.post(
            "/bundles",
            json={"event_ids": [event_id]},
        )
        bundle_id = resp.json()["id"]

        resp = await client.get(f"/bundles/{bundle_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["explanation_artifacts"] is not None

    @pytest.mark.asyncio
    async def test_bundle_api_empty_filter(self, client):
        """POST /bundles with empty layer_filters excludes all layer artifacts."""
        resp = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model_v1",
                "payload": {"score": 0.85},
            },
        )
        event_id = resp.json()["id"]

        resp = await client.post(
            "/bundles",
            json={
                "event_ids": [event_id],
                "layer_filters": [],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["recognition_artifacts"] is None
        assert data["fl_artifacts"] is None
        assert data["bias_report"] is None
        assert data["explanation_artifacts"] is None


# ---------------------------------------------------------------------------
# Storage round-trip tests
# ---------------------------------------------------------------------------


class TestBundleStorageRoundtrip:
    """Database persistence of layer artifact fields."""

    @pytest.mark.asyncio
    async def test_artifacts_persist_and_load(self, service, db):
        """Bundle with artifacts survives insert -> get round-trip."""
        e = await service.record_event(
            EventType.INFERENCE_RESULT,
            actor="model_v1",
            payload={"score": 0.95},
        )
        bundle = await service.create_bundle(event_ids=[e.id])

        # Re-fetch from database
        loaded = await db.get_bundle(bundle.id)
        assert loaded is not None
        assert loaded.recognition_artifacts == bundle.recognition_artifacts
        assert loaded.fl_artifacts == bundle.fl_artifacts
        assert loaded.bias_report == bundle.bias_report
        assert loaded.explanation_artifacts == bundle.explanation_artifacts
        assert loaded.bundle_hash == bundle.bundle_hash

    @pytest.mark.asyncio
    async def test_none_artifacts_persist(self, service, db):
        """Bundle with None artifact fields persists correctly."""
        e = await service.record_event(
            EventType.CONSENT_RECORDED,
            actor="consent_mgr",
            payload={"subject": "user_1"},
        )
        bundle = await service.create_bundle(event_ids=[e.id])

        loaded = await db.get_bundle(bundle.id)
        assert loaded is not None
        assert loaded.recognition_artifacts is None
        assert loaded.fl_artifacts is None
        assert loaded.bias_report is None
        assert loaded.explanation_artifacts is None
        assert loaded.verify()
