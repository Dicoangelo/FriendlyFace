"""End-to-end forensic pipeline test (US-019).

This is the capstone convergence test that ties all 6 architectural layers
together into a single end-to-end pipeline:

  1. Consent        — Grant consent for a subject
  2. Recognition    — Train PCA+SVM, record inference event with forensic chain
  3. Explainability — LIME and SHAP explanations with forensic events
  4. Fairness       — Bias audit with compliance verification
  5. Governance     — Compliance report generation and scoring
  6. Forensics      — Bundle creation, hash chain, Merkle proofs, provenance DAG

The test walks through the FULL forensic lifecycle via the API endpoints,
verifying that every operation produces the expected forensic trail and that
the entire chain remains cryptographically consistent throughout.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

from friendlyface.api import app as app_module
from friendlyface.api.app import _db, _service, app
from friendlyface.recognition.pca import IMAGE_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_test_dataset(
    tmp_path: Path,
    n_images: int = 12,
    n_classes: int = 3,
) -> tuple[Path, list[int]]:
    """Create a synthetic image dataset directory and return (dir_path, labels)."""
    rng = np.random.default_rng(42)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    labels = []
    for i in range(n_images):
        pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
        img = Image.fromarray(pixels, mode="L")
        img.save(dataset_dir / f"face_{i:04d}.png")
        labels.append(i % n_classes)

    return dataset_dir, labels


def _make_test_image(seed: int = 99) -> bytes:
    """Create a synthetic grayscale image and return raw PNG bytes."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database with all global state reset."""
    _db.db_path = tmp_path / "e2e_test.db"
    await _db.connect()
    await _service.initialize()

    # Reset in-memory state
    _service.merkle = __import__(
        "friendlyface.core.merkle", fromlist=["MerkleTree"]
    ).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()

    # Reset all in-memory stores
    app_module._latest_compliance_report = None
    app_module._explanations.clear()
    app_module._model_registry.clear()
    app_module._fl_simulations.clear()
    app_module._auto_audit_interval = 50
    app_module._recognition_event_count = 0

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """Full forensic lifecycle: consent -> train -> infer -> explain ->
    audit -> compliance -> bundle -> verify.
    """

    async def test_full_forensic_lifecycle(self, client, tmp_path):
        """Walk through the FULL forensic lifecycle across all 6 layers."""

        collected_event_ids: list[str] = []
        collected_provenance_ids: list[str] = []

        # ---------------------------------------------------------------
        # Step 1: Grant consent for a subject
        # ---------------------------------------------------------------
        consent_resp = await client.post(
            "/consent/grant",
            json={
                "subject_id": "subject_alpha",
                "purpose": "recognition",
                "actor": "e2e_test",
            },
        )
        assert consent_resp.status_code == 201
        consent_data = consent_resp.json()
        assert consent_data["granted"] is True
        assert consent_data["subject_id"] == "subject_alpha"
        consent_event_id = consent_data["event_id"]
        collected_event_ids.append(consent_event_id)

        # Verify consent is active
        check_resp = await client.post(
            "/consent/check",
            json={"subject_id": "subject_alpha", "purpose": "recognition"},
        )
        assert check_resp.status_code == 200
        assert check_resp.json()["allowed"] is True
        assert check_resp.json()["active"] is True

        # ---------------------------------------------------------------
        # Step 2: Train PCA+SVM model with provenance
        # ---------------------------------------------------------------
        dataset_dir, labels = _create_test_dataset(tmp_path)
        output_dir = tmp_path / "models"

        train_resp = await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(output_dir),
                "n_components": 5,
                "labels": labels,
            },
        )
        assert train_resp.status_code == 201
        train_data = train_resp.json()

        # Verify training returned all expected fields
        assert "model_id" in train_data
        assert "pca_event_id" in train_data
        assert "svm_event_id" in train_data
        assert "pca_provenance_id" in train_data
        assert "svm_provenance_id" in train_data
        assert train_data["n_components"] == 5
        assert train_data["n_samples"] == 12
        assert train_data["n_classes"] == 3

        model_id = train_data["model_id"]
        pca_event_id = train_data["pca_event_id"]
        svm_event_id = train_data["svm_event_id"]
        pca_prov_id = train_data["pca_provenance_id"]
        svm_prov_id = train_data["svm_provenance_id"]

        collected_event_ids.extend([pca_event_id, svm_event_id])
        collected_provenance_ids.extend([pca_prov_id, svm_prov_id])

        # Verify PCA training event in forensic chain
        pca_event_resp = await client.get(f"/events/{pca_event_id}")
        assert pca_event_resp.status_code == 200
        pca_event = pca_event_resp.json()
        assert pca_event["event_type"] == "training_complete"
        assert pca_event["payload"]["model_type"] == "PCA"

        # Verify SVM training event in forensic chain
        svm_event_resp = await client.get(f"/events/{svm_event_id}")
        assert svm_event_resp.status_code == 200
        svm_event = svm_event_resp.json()
        assert svm_event["event_type"] == "training_complete"
        assert svm_event["payload"]["model_type"] == "SVM"

        # Verify provenance DAG: SVM derives from PCA
        prov_chain_resp = await client.get(f"/provenance/{svm_prov_id}")
        assert prov_chain_resp.status_code == 200
        prov_chain = prov_chain_resp.json()
        assert len(prov_chain) == 2  # PCA -> SVM
        assert prov_chain[0]["entity_type"] == "training"
        assert prov_chain[0]["metadata"]["model_type"] == "PCA"
        assert prov_chain[1]["entity_type"] == "training"
        assert prov_chain[1]["metadata"]["model_type"] == "SVM"

        # ---------------------------------------------------------------
        # Step 3: Record inference event with forensic chain
        # ---------------------------------------------------------------
        image_bytes = _make_test_image()
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", image_bytes, "image/png")},
        )
        assert predict_resp.status_code == 200
        predict_data = predict_resp.json()
        assert "event_id" in predict_data
        assert "input_hash" in predict_data
        assert "matches" in predict_data
        assert len(predict_data["matches"]) > 0

        inference_event_id = predict_data["event_id"]
        collected_event_ids.append(inference_event_id)

        # Verify inference event is in the forensic chain
        inf_event_resp = await client.get(f"/events/{inference_event_id}")
        assert inf_event_resp.status_code == 200
        inf_event = inf_event_resp.json()
        assert inf_event["event_type"] == "inference_result"

        # ---------------------------------------------------------------
        # Step 4: Trigger LIME explanation -> verify forensic event
        # ---------------------------------------------------------------
        lime_resp = await client.post(
            "/explainability/lime",
            json={"event_id": inference_event_id},
        )
        assert lime_resp.status_code == 201
        lime_data = lime_resp.json()
        assert lime_data["method"] == "lime"
        assert lime_data["inference_event_id"] == inference_event_id
        assert "explanation_id" in lime_data
        assert "event_id" in lime_data

        lime_event_id = lime_data["event_id"]
        collected_event_ids.append(lime_event_id)

        # Verify the LIME forensic event exists
        lime_event_resp = await client.get(f"/events/{lime_event_id}")
        assert lime_event_resp.status_code == 200
        lime_event = lime_event_resp.json()
        assert lime_event["event_type"] == "explanation_generated"
        assert lime_event["payload"]["method"] == "lime"

        # ---------------------------------------------------------------
        # Step 5: Trigger SHAP explanation -> verify forensic event
        # ---------------------------------------------------------------
        shap_resp = await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id},
        )
        assert shap_resp.status_code == 201
        shap_data = shap_resp.json()
        assert shap_data["method"] == "shap"
        assert shap_data["inference_event_id"] == inference_event_id
        assert "explanation_id" in shap_data
        assert "event_id" in shap_data

        shap_event_id = shap_data["event_id"]
        collected_event_ids.append(shap_event_id)

        # Verify the SHAP forensic event exists
        shap_event_resp = await client.get(f"/events/{shap_event_id}")
        assert shap_event_resp.status_code == 200
        shap_event = shap_event_resp.json()
        assert shap_event["event_type"] == "explanation_generated"
        assert shap_event["payload"]["method"] == "shap"

        # Verify both explanations are retrievable
        expl_list_resp = await client.get("/explainability/explanations")
        assert expl_list_resp.status_code == 200
        expl_list = expl_list_resp.json()
        assert expl_list["total"] >= 2
        methods = {e["method"] for e in expl_list["explanations"]}
        assert "lime" in methods
        assert "shap" in methods

        # Verify comparison endpoint works
        compare_resp = await client.get(
            f"/explainability/compare/{inference_event_id}"
        )
        assert compare_resp.status_code == 200
        compare_data = compare_resp.json()
        assert compare_data["total_lime"] >= 1
        assert compare_data["total_shap"] >= 1

        # ---------------------------------------------------------------
        # Step 6: Run bias audit -> verify compliance
        # ---------------------------------------------------------------
        audit_resp = await client.post(
            "/fairness/audit",
            json={
                "groups": [
                    {
                        "group_name": "group_a",
                        "true_positives": 85,
                        "false_positives": 8,
                        "true_negatives": 92,
                        "false_negatives": 15,
                    },
                    {
                        "group_name": "group_b",
                        "true_positives": 82,
                        "false_positives": 10,
                        "true_negatives": 90,
                        "false_negatives": 18,
                    },
                ],
                "metadata": {"trigger": "e2e_pipeline"},
            },
        )
        assert audit_resp.status_code == 201
        audit_data = audit_resp.json()
        assert audit_data["compliant"] is True
        assert audit_data["audit_id"] is not None
        assert audit_data["event_id"] is not None
        assert "group_a" in audit_data["groups_evaluated"]
        assert "group_b" in audit_data["groups_evaluated"]
        assert audit_data["fairness_score"] is not None
        assert audit_data["fairness_score"] > 0

        audit_event_id = audit_data["event_id"]
        collected_event_ids.append(audit_event_id)

        # Verify fairness status is pass
        fairness_status_resp = await client.get("/fairness/status")
        assert fairness_status_resp.status_code == 200
        fairness_status = fairness_status_resp.json()
        assert fairness_status["status"] == "pass"
        assert fairness_status["compliant"] is True

        # ---------------------------------------------------------------
        # Step 7: Generate compliance report -> verify scoring
        # ---------------------------------------------------------------
        compliance_resp = await client.post("/governance/compliance/generate")
        assert compliance_resp.status_code == 201
        compliance_data = compliance_resp.json()

        assert "report_id" in compliance_data
        assert "overall_compliance_score" in compliance_data
        assert "article_5" in compliance_data
        assert "article_14" in compliance_data
        assert "metrics" in compliance_data
        assert "event_id" in compliance_data

        # Metrics should reflect our seeded data
        metrics = compliance_data["metrics"]
        assert "consent_coverage_pct" in metrics
        assert "bias_audit_pass_rate_pct" in metrics
        assert "explanation_coverage_pct" in metrics
        assert "bundle_integrity_pct" in metrics

        # We have 1 consent subject, 1 compliant bias audit,
        # 1 inference + 2 explanations, no bundles yet
        assert metrics["consent_coverage_pct"] == 100.0
        assert metrics["bias_audit_pass_rate_pct"] == 100.0

        compliance_event_id = compliance_data["event_id"]
        collected_event_ids.append(compliance_event_id)

        # ---------------------------------------------------------------
        # Step 8: Create enhanced forensic bundle with ALL layer artifacts
        # ---------------------------------------------------------------
        bundle_resp = await client.post(
            "/bundles",
            json={
                "event_ids": collected_event_ids,
                "provenance_node_ids": collected_provenance_ids,
            },
        )
        assert bundle_resp.status_code == 201
        bundle_data = bundle_resp.json()

        assert "id" in bundle_data
        assert "bundle_hash" in bundle_data
        assert bundle_data["bundle_hash"] != ""
        assert bundle_data["status"] == "complete"

        bundle_id = bundle_data["id"]

        # Verify bundle has auto-collected artifacts from ALL layers
        assert bundle_data["recognition_artifacts"] is not None
        assert bundle_data["explanation_artifacts"] is not None
        assert bundle_data["bias_report"] is not None

        # Recognition artifacts should have inference events
        rec_arts = bundle_data["recognition_artifacts"]
        assert "inference_events" in rec_arts
        assert len(rec_arts["inference_events"]) >= 1

        # Explanation artifacts should have explanations
        expl_arts = bundle_data["explanation_artifacts"]
        assert "explanations" in expl_arts
        assert len(expl_arts["explanations"]) >= 1

        # Bias report should have audits
        bias_arts = bundle_data["bias_report"]
        assert "audits" in bias_arts
        assert len(bias_arts["audits"]) >= 1

        # ---------------------------------------------------------------
        # Step 9: Verify bundle integrity (hash covers all artifacts)
        # ---------------------------------------------------------------
        verify_resp = await client.post(f"/verify/{bundle_id}")
        assert verify_resp.status_code == 200
        verify_data = verify_resp.json()
        assert verify_data["valid"] is True
        assert verify_data["status"] == "verified"
        assert verify_data["bundle_hash_valid"] is True

        # Layer artifact integrity checks
        layer_arts = verify_data.get("layer_artifacts", {})
        for layer in ("recognition", "bias", "explanation"):
            assert layer in layer_arts, f"Missing layer: {layer}"
            assert layer_arts[layer]["valid"] is True, (
                f"Layer {layer} verification failed"
            )

        # Verify bundle is retrievable
        get_bundle_resp = await client.get(f"/bundles/{bundle_id}")
        assert get_bundle_resp.status_code == 200
        retrieved_bundle = get_bundle_resp.json()
        assert retrieved_bundle["id"] == bundle_id
        assert retrieved_bundle["bundle_hash"] == bundle_data["bundle_hash"]

        # ---------------------------------------------------------------
        # Step 10: Verify full hash chain integrity across all events
        # ---------------------------------------------------------------
        chain_resp = await client.get("/chain/integrity")
        assert chain_resp.status_code == 200
        chain_data = chain_resp.json()
        assert chain_data["valid"] is True

        # ---------------------------------------------------------------
        # Step 11: Verify Merkle proofs for all events
        # ---------------------------------------------------------------
        for event_id in collected_event_ids:
            proof_resp = await client.get(f"/merkle/proof/{event_id}")
            assert proof_resp.status_code == 200, (
                f"Merkle proof failed for event {event_id}"
            )
            proof_data = proof_resp.json()
            assert proof_data["leaf_hash"] != ""
            assert proof_data["root_hash"] != ""
            assert "proof_hashes" in proof_data

        # Verify Merkle root is non-null (events exist)
        root_resp = await client.get("/merkle/root")
        assert root_resp.status_code == 200
        root_data = root_resp.json()
        assert root_data["merkle_root"] is not None
        assert root_data["leaf_count"] >= len(collected_event_ids)

        # ---------------------------------------------------------------
        # Step 12: Verify provenance DAG connectivity
        # ---------------------------------------------------------------
        # SVM provenance should have PCA as parent
        svm_chain_resp = await client.get(f"/provenance/{svm_prov_id}")
        assert svm_chain_resp.status_code == 200
        svm_chain = svm_chain_resp.json()
        assert len(svm_chain) >= 2

        # PCA provenance should exist as standalone root
        pca_chain_resp = await client.get(f"/provenance/{pca_prov_id}")
        assert pca_chain_resp.status_code == 200
        pca_chain = pca_chain_resp.json()
        assert len(pca_chain) >= 1
        assert pca_chain[0]["entity_type"] == "training"

        # Model registry should have the trained model with provenance
        model_resp = await client.get(f"/recognition/models/{model_id}")
        assert model_resp.status_code == 200
        model_data = model_resp.json()
        assert model_data["id"] == model_id
        assert "provenance_chain" in model_data
        assert len(model_data["provenance_chain"]) == 2

    async def test_event_sequencing_integrity(self, client, tmp_path):
        """Verify that events form a proper hash chain with sequential numbering."""

        # Create a sequence of events across different types
        r1 = await client.post(
            "/consent/grant",
            json={
                "subject_id": "seq_test_subj",
                "purpose": "recognition",
            },
        )
        assert r1.status_code == 201

        r2 = await client.post(
            "/events",
            json={
                "event_type": "inference_request",
                "actor": "seq_test",
                "payload": {"image_hash": "abc"},
            },
        )
        assert r2.status_code == 201

        r3 = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "seq_test",
                "payload": {"score": 0.9},
            },
        )
        assert r3.status_code == 201

        # Verify sequential ordering
        all_events_resp = await client.get("/events")
        assert all_events_resp.status_code == 200
        all_events = all_events_resp.json()

        # Events should have incrementing sequence numbers
        for i in range(len(all_events)):
            assert all_events[i]["sequence_number"] == i

        # Each event's previous_hash should match the prior event's hash
        for i in range(1, len(all_events)):
            assert all_events[i]["previous_hash"] == all_events[i - 1]["event_hash"]

        # First event should reference GENESIS
        assert all_events[0]["previous_hash"] == "GENESIS"

        # Chain integrity check should confirm
        chain_resp = await client.get("/chain/integrity")
        assert chain_resp.json()["valid"] is True

    async def test_consent_revocation_tracked_in_forensic_chain(self, client):
        """Consent grant and revocation both create forensic events."""
        await client.post(
            "/consent/grant",
            json={
                "subject_id": "revoke_subj",
                "purpose": "recognition",
            },
        )
        await client.post(
            "/consent/revoke",
            json={
                "subject_id": "revoke_subj",
                "purpose": "recognition",
                "reason": "user_request",
            },
        )

        events_resp = await client.get("/events")
        events = events_resp.json()
        consent_events = [
            e for e in events if e["event_type"] == "consent_update"
        ]
        assert len(consent_events) == 2

        # After revocation, consent check should deny
        check_resp = await client.post(
            "/consent/check",
            json={"subject_id": "revoke_subj", "purpose": "recognition"},
        )
        assert check_resp.json()["allowed"] is False

    async def test_explanation_requires_inference_event(self, client):
        """LIME and SHAP endpoints require a valid inference event ID."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        lime_resp = await client.post(
            "/explainability/lime",
            json={"event_id": fake_id},
        )
        assert lime_resp.status_code == 404

        shap_resp = await client.post(
            "/explainability/shap",
            json={"event_id": fake_id},
        )
        assert shap_resp.status_code == 404

    async def test_bias_audit_with_biased_data(self, client):
        """Bias audit correctly detects non-compliant results."""
        resp = await client.post(
            "/fairness/audit",
            json={
                "groups": [
                    {
                        "group_name": "group_a",
                        "true_positives": 90,
                        "false_positives": 5,
                        "true_negatives": 95,
                        "false_negatives": 10,
                    },
                    {
                        "group_name": "group_b",
                        "true_positives": 40,
                        "false_positives": 30,
                        "true_negatives": 70,
                        "false_negatives": 60,
                    },
                ],
                "demographic_parity_threshold": 0.01,
                "equalized_odds_threshold": 0.01,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["compliant"] is False
        assert len(data["alerts"]) > 0

        # Verify chain is still valid after biased audit
        chain_resp = await client.get("/chain/integrity")
        assert chain_resp.json()["valid"] is True

    async def test_compliance_report_reflects_pipeline_state(self, client):
        """Compliance report accurately scores data from all layers."""
        # Grant consent for multiple subjects
        for i in range(5):
            await client.post(
                "/consent/grant",
                json={
                    "subject_id": f"compliance_subj_{i}",
                    "purpose": "recognition",
                },
            )

        # Run a compliant bias audit
        await client.post(
            "/fairness/audit",
            json={
                "groups": [
                    {
                        "group_name": "group_a",
                        "true_positives": 80,
                        "false_positives": 10,
                        "true_negatives": 90,
                        "false_negatives": 20,
                    },
                    {
                        "group_name": "group_b",
                        "true_positives": 80,
                        "false_positives": 10,
                        "true_negatives": 90,
                        "false_negatives": 20,
                    },
                ],
            },
        )

        # Generate compliance report
        resp = await client.post("/governance/compliance/generate")
        assert resp.status_code == 201
        data = resp.json()

        # All consent subjects have active consent -> 100%
        assert data["metrics"]["consent_coverage_pct"] == 100.0
        # One compliant bias audit -> 100%
        assert data["metrics"]["bias_audit_pass_rate_pct"] == 100.0

        # Overall score should be above zero
        assert data["overall_compliance_score"] > 0

        # Report should produce a forensic event
        assert data["event_id"] is not None
        event_resp = await client.get(f"/events/{data['event_id']}")
        assert event_resp.status_code == 200
        assert event_resp.json()["event_type"] == "compliance_report"

    async def test_bundle_with_all_layer_filters(self, client):
        """Bundle creation with explicit layer filters includes only specified layers."""
        # Create events across multiple layers
        e1 = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model",
                "payload": {"score": 0.9},
            },
        )
        e2 = await client.post(
            "/events",
            json={
                "event_type": "explanation_generated",
                "actor": "lime",
                "payload": {"method": "LIME"},
            },
        )
        e3 = await client.post(
            "/events",
            json={
                "event_type": "bias_audit",
                "actor": "auditor",
                "payload": {"compliant": True},
            },
        )

        eid1 = e1.json()["id"]
        eid2 = e2.json()["id"]
        eid3 = e3.json()["id"]

        # Create bundle with only recognition filter
        resp = await client.post(
            "/bundles",
            json={
                "event_ids": [eid1, eid2, eid3],
                "layer_filters": ["recognition"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["recognition_artifacts"] is not None
        assert data["explanation_artifacts"] is None
        assert data["bias_report"] is None

        # Create bundle with all filters
        resp2 = await client.post(
            "/bundles",
            json={
                "event_ids": [eid1, eid2, eid3],
                "layer_filters": ["recognition", "explanation", "bias"],
            },
        )
        assert resp2.status_code == 201
        data2 = resp2.json()
        assert data2["recognition_artifacts"] is not None
        assert data2["explanation_artifacts"] is not None
        assert data2["bias_report"] is not None

    async def test_merkle_tree_grows_with_pipeline(self, client):
        """Merkle tree leaf count grows as events are added across the pipeline."""
        # Initial state
        root_resp = await client.get("/merkle/root")
        initial_count = root_resp.json()["leaf_count"]

        # Add events from different layers
        await client.post(
            "/events",
            json={
                "event_type": "training_start",
                "actor": "trainer",
                "payload": {},
            },
        )
        await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model",
                "payload": {"score": 0.8},
            },
        )
        await client.post(
            "/events",
            json={
                "event_type": "bias_audit",
                "actor": "auditor",
                "payload": {"compliant": True},
            },
        )

        # Verify leaf count increased
        root_resp2 = await client.get("/merkle/root")
        final_count = root_resp2.json()["leaf_count"]
        assert final_count == initial_count + 3
        assert root_resp2.json()["merkle_root"] is not None

    async def test_health_reflects_event_count(self, client):
        """Health endpoint reflects the total event count in the system."""
        health_resp = await client.get("/health")
        initial_count = health_resp.json()["event_count"]

        # Add some events
        for i in range(3):
            await client.post(
                "/events",
                json={
                    "event_type": "training_start",
                    "actor": "counter_test",
                    "payload": {"step": i},
                },
            )

        health_resp2 = await client.get("/health")
        assert health_resp2.json()["event_count"] == initial_count + 3
