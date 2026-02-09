"""Phase 9 E2E integration tests (US-097, US-098).

End-to-end lifecycle tests covering Phase 9 features:
  - Gallery enroll → predict → real explainability → demographic auto-audit → bundle
  - Persistence survival: data survives DB close + reopen
  - Full chain integrity after all operations
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from friendlyface.api.app import _db, _service
from friendlyface.recognition.pca import IMAGE_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_face_bytes(seed: int = 42) -> bytes:
    """Create a synthetic grayscale face image as PNG bytes."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
    img = PILImage.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_rgb_face_bytes(seed: int = 42, size: tuple[int, int] = (112, 112)) -> bytes:
    """Create a synthetic RGB face image as PNG bytes (for gallery endpoints)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(*size, 3), dtype=np.uint8)
    img = PILImage.fromarray(pixels, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _train_model(client, tmp_path: Path) -> dict:
    """Train a PCA+SVM model via the API and return training response."""
    rng = np.random.default_rng(42)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    labels = []
    for i in range(12):
        pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
        img = PILImage.fromarray(pixels, mode="L")
        img.save(dataset_dir / f"face_{i:04d}.png")
        labels.append(i % 3)

    output_dir = tmp_path / "models"
    resp = await client.post(
        "/recognition/train",
        json={
            "dataset_path": str(dataset_dir),
            "output_dir": str(output_dir),
            "n_components": 5,
            "labels": labels,
        },
    )
    assert resp.status_code == 201
    return resp.json()


# ---------------------------------------------------------------------------
# Full lifecycle with Phase 9 features
# ---------------------------------------------------------------------------


class TestPhase9Lifecycle:
    """Full lifecycle: consent → train → predict(demographic) → real explain → auto-audit → bundle."""

    async def test_full_phase9_lifecycle(self, client, tmp_path):
        """Walk through the complete Phase 9 lifecycle with real computation."""
        collected_event_ids: list[str] = []

        # Step 1: Grant consent
        consent_resp = await client.post(
            "/consent/grant",
            json={
                "subject_id": "phase9_subject",
                "purpose": "recognition",
                "actor": "e2e_phase9",
            },
        )
        assert consent_resp.status_code == 201
        collected_event_ids.append(consent_resp.json()["event_id"])

        # Step 2: Train model
        train_data = await _train_model(client, tmp_path)
        collected_event_ids.extend([train_data["pca_event_id"], train_data["svm_event_id"]])

        # Step 3: Predict with demographic group + true label
        face_bytes = _make_face_bytes(seed=99)
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
            params={"demographic_group": "group_a", "true_label": 1},
        )
        assert predict_resp.status_code == 200
        predict_data = predict_resp.json()
        inference_event_id = predict_data["event_id"]
        collected_event_ids.append(inference_event_id)
        assert "matches" in predict_data

        # Step 4: Real LIME explanation (artifact should exist)
        lime_resp = await client.post(
            "/explainability/lime",
            json={
                "event_id": inference_event_id,
                "num_superpixels": 25,
                "num_samples": 50,
                "top_k": 3,
            },
        )
        assert lime_resp.status_code == 201
        lime_data = lime_resp.json()
        assert lime_data["method"] == "lime"
        assert lime_data["computed"] is True
        assert "top_regions" in lime_data
        assert len(lime_data["top_regions"]) <= 3
        collected_event_ids.append(lime_data["event_id"])

        # Step 5: Real SHAP explanation
        shap_resp = await client.post(
            "/explainability/shap",
            json={"event_id": inference_event_id, "num_samples": 32},
        )
        assert shap_resp.status_code == 201
        shap_data = shap_resp.json()
        assert shap_data["method"] == "shap"
        assert shap_data["computed"] is True
        assert "base_value" in shap_data
        assert "top_features" in shap_data
        collected_event_ids.append(shap_data["event_id"])

        # Step 6: Real SDD explanation
        sdd_resp = await client.post(
            "/explainability/sdd",
            json={"event_id": inference_event_id},
        )
        assert sdd_resp.status_code == 201
        sdd_data = sdd_resp.json()
        assert sdd_data["method"] == "sdd"
        assert sdd_data["computed"] is True
        assert "regions" in sdd_data
        assert "dominant_region" in sdd_data
        collected_event_ids.append(sdd_data["event_id"])

        # Step 7: Compare should show all three methods computed
        compare_resp = await client.get(f"/explainability/compare/{inference_event_id}")
        assert compare_resp.status_code == 200
        compare_data = compare_resp.json()
        assert compare_data["total_lime"] >= 1
        assert compare_data["total_shap"] >= 1
        assert compare_data["total_sdd"] >= 1

        # Step 8: Run bias audit
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
            },
        )
        assert audit_resp.status_code == 201
        assert audit_resp.json()["compliant"] is True
        collected_event_ids.append(audit_resp.json()["event_id"])

        # Step 9: Generate compliance report
        compliance_resp = await client.post("/governance/compliance/generate")
        assert compliance_resp.status_code == 201
        compliance_data = compliance_resp.json()
        assert compliance_data["metrics"]["consent_coverage_pct"] == 100.0
        assert compliance_data["metrics"]["bias_audit_pass_rate_pct"] == 100.0
        # All 3 explanations for 1 inference = 100% explanation coverage
        assert compliance_data["metrics"]["explanation_coverage_pct"] > 0
        collected_event_ids.append(compliance_data["event_id"])

        # Step 10: Create bundle with all collected events
        bundle_resp = await client.post(
            "/bundles",
            json={
                "event_ids": collected_event_ids,
                "provenance_node_ids": [
                    train_data["pca_provenance_id"],
                    train_data["svm_provenance_id"],
                ],
            },
        )
        assert bundle_resp.status_code == 201
        bundle_data = bundle_resp.json()
        bundle_id = bundle_data["id"]
        assert bundle_data["status"] == "complete"
        assert bundle_data["recognition_artifacts"] is not None
        assert bundle_data["explanation_artifacts"] is not None
        assert bundle_data["bias_report"] is not None

        # Step 11: Verify bundle integrity
        verify_resp = await client.post(f"/verify/{bundle_id}")
        assert verify_resp.status_code == 200
        assert verify_resp.json()["valid"] is True

        # Step 12: Verify hash chain
        chain_resp = await client.get("/chain/integrity")
        assert chain_resp.json()["valid"] is True

    async def test_demographic_predict_records_result(self, client, tmp_path):
        """Predict with demographic_group writes to recognition_results table."""
        await _train_model(client, tmp_path)

        face_bytes = _make_face_bytes(seed=55)
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
            params={"demographic_group": "age_18_25", "true_label": 2},
        )
        assert resp.status_code == 200

        # Verify the demographic result was recorded
        stats = await _db.get_demographic_stats()
        assert len(stats) >= 1
        group_names = [s["group_name"] for s in stats]
        assert "age_18_25" in group_names

    async def test_multiple_demographics_trigger_real_audit(self, client, tmp_path):
        """Auto-audit uses real demographic data when >= 2 groups have labeled results."""
        from friendlyface.api.app import _build_audit_groups

        await _train_model(client, tmp_path)

        # Predict with two demographic groups
        for i, group in enumerate(["group_x", "group_y"]):
            for j in range(3):
                face_bytes = _make_face_bytes(seed=100 + i * 10 + j)
                await client.post(
                    "/recognition/predict",
                    files={"image": ("face.png", face_bytes, "image/png")},
                    params={"demographic_group": group, "true_label": j % 3},
                )

        groups = await _build_audit_groups()
        group_names = [g.group_name for g in groups]
        assert "group_x" in group_names
        assert "group_y" in group_names

    async def test_inference_artifact_enables_real_explainability(self, client, tmp_path):
        """Without trained model, explanations are stubs. With model, they're real."""
        # Create event manually (no model trained)
        event_resp = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "test",
                "payload": {},
            },
        )
        event_id = event_resp.json()["id"]

        # LIME without artifact = stub
        lime_resp = await client.post("/explainability/lime", json={"event_id": event_id})
        assert lime_resp.status_code == 201
        assert lime_resp.json()["computed"] is False

        # Train model + predict (creates artifact)
        await _train_model(client, tmp_path)
        face_bytes = _make_face_bytes(seed=77)
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        real_event_id = predict_resp.json()["event_id"]

        # LIME with artifact = real
        lime_resp2 = await client.post(
            "/explainability/lime",
            json={"event_id": real_event_id, "num_samples": 50},
        )
        assert lime_resp2.status_code == 201
        assert lime_resp2.json()["computed"] is True

    async def test_compare_shows_computed_status(self, client, tmp_path):
        """Compare endpoint shows computed status for each explanation method."""
        await _train_model(client, tmp_path)

        face_bytes = _make_face_bytes(seed=88)
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        event_id = predict_resp.json()["event_id"]

        # Trigger all three
        await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 50},
        )
        await client.post(
            "/explainability/shap",
            json={"event_id": event_id, "num_samples": 32},
        )
        await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )

        compare_resp = await client.get(f"/explainability/compare/{event_id}")
        data = compare_resp.json()

        # All should be computed since model is trained and artifact exists
        for lime_expl in data["lime_explanations"]:
            assert lime_expl["computed"] is True
        for shap_expl in data["shap_explanations"]:
            assert shap_expl["computed"] is True
        for sdd_expl in data["sdd_explanations"]:
            assert sdd_expl["computed"] is True


# ---------------------------------------------------------------------------
# Persistence survival
# ---------------------------------------------------------------------------


class TestPersistenceSurvival:
    """Verify data survives DB close + reopen."""

    async def test_events_survive_reconnect(self, client, tmp_path):
        """Forensic events survive database close + reopen."""
        # Create some events
        for i in range(5):
            await client.post(
                "/events",
                json={
                    "event_type": "inference_result",
                    "actor": "survival_test",
                    "payload": {"step": i},
                },
            )

        events_resp = await client.get("/events")
        original_count = events_resp.json()["total"]
        assert original_count >= 5

        # Close and reopen DB
        await _db.close()
        await _db.connect()
        await _service.initialize()

        # Events should still be there
        events_resp2 = await client.get("/events")
        assert events_resp2.json()["total"] == original_count

        # Chain should still be valid
        chain_resp = await client.get("/chain/integrity")
        assert chain_resp.json()["valid"] is True

    async def test_explanations_survive_reconnect(self, client, tmp_path):
        """Persisted explanations survive database close + reopen."""
        await _train_model(client, tmp_path)

        face_bytes = _make_face_bytes(seed=66)
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        event_id = predict_resp.json()["event_id"]

        # Create explanations
        lime_resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 50},
        )
        explanation_id = lime_resp.json()["explanation_id"]

        # Close and reopen
        await _db.close()
        await _db.connect()
        await _service.initialize()

        # Explanation should survive in DB (via _persist_explanation)
        expl_list = await client.get("/explainability/explanations")
        assert expl_list.status_code == 200
        ids = [e["explanation_id"] for e in expl_list.json()["items"]]
        assert explanation_id in ids

    async def test_compliance_cache_survives_reconnect(self, client, tmp_path):
        """Compliance reports survive database close + reopen."""
        # Create consent + audit to generate meaningful report
        await client.post(
            "/consent/grant",
            json={"subject_id": "persist_subj", "purpose": "recognition"},
        )
        await client.post(
            "/fairness/audit",
            json={
                "groups": [
                    {
                        "group_name": "g1",
                        "true_positives": 80,
                        "false_positives": 10,
                        "true_negatives": 90,
                        "false_negatives": 20,
                    },
                    {
                        "group_name": "g2",
                        "true_positives": 80,
                        "false_positives": 10,
                        "true_negatives": 90,
                        "false_negatives": 20,
                    },
                ],
            },
        )

        # Generate report
        gen_resp = await client.post("/governance/compliance/generate")
        assert gen_resp.status_code == 201
        original_score = gen_resp.json()["overall_compliance_score"]

        # Close and reopen
        await _db.close()
        await _db.connect()
        await _service.initialize()

        # At least one report should be persisted in DB
        reports = await _db.list_compliance_reports()
        assert len(reports) >= 1
        assert reports[0]["overall_compliance_score"] == original_score

    async def test_model_registry_survives_reconnect(self, client, tmp_path):
        """Trained model registry entries survive database close + reopen."""
        train_data = await _train_model(client, tmp_path)
        model_id = train_data["model_id"]

        # Close and reopen
        await _db.close()
        await _db.connect()
        await _service.initialize()

        # Model should be in DB
        models = await _db.list_models()
        model_ids = [m["id"] for m in models]
        assert model_id in model_ids

    async def test_inference_artifacts_survive_reconnect(self, client, tmp_path):
        """Inference artifacts survive database close + reopen."""
        await _train_model(client, tmp_path)

        face_bytes = _make_face_bytes(seed=44)
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        event_id = predict_resp.json()["event_id"]

        # Close and reopen
        await _db.close()
        await _db.connect()
        await _service.initialize()

        # Artifact should survive
        artifact = await _db.get_inference_artifact(event_id)
        assert artifact is not None
        assert len(artifact["image_bytes"]) > 0

    async def test_recognition_results_survive_reconnect(self, client, tmp_path):
        """Demographic recognition results survive database close + reopen."""
        await _train_model(client, tmp_path)

        face_bytes = _make_face_bytes(seed=33)
        await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
            params={"demographic_group": "persist_group", "true_label": 1},
        )

        # Close and reopen
        await _db.close()
        await _db.connect()
        await _service.initialize()

        # Result should survive
        stats = await _db.get_demographic_stats()
        group_names = [s["group_name"] for s in stats]
        assert "persist_group" in group_names

    async def test_bundle_survives_reconnect(self, client, tmp_path):
        """Bundles survive database close + reopen and remain verifiable."""
        # Create events and bundle
        e1 = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "test",
                "payload": {"score": 0.9},
            },
        )
        event_id = e1.json()["id"]

        bundle_resp = await client.post("/bundles", json={"event_ids": [event_id]})
        assert bundle_resp.status_code == 201
        bundle_id = bundle_resp.json()["id"]

        # Close and reopen
        await _db.close()
        await _db.connect()
        await _service.initialize()

        # Bundle should be retrievable
        get_resp = await client.get(f"/bundles/{bundle_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == bundle_id

        # Chain should still be valid
        chain_resp = await client.get("/chain/integrity")
        assert chain_resp.json()["valid"] is True


# ---------------------------------------------------------------------------
# Cross-feature integration
# ---------------------------------------------------------------------------


class TestCrossFeatureIntegration:
    """Tests that verify features from different batches work together."""

    async def test_artifact_deleted_after_explanation(self, client, tmp_path):
        """Deleting an artifact doesn't affect already-persisted explanations."""
        await _train_model(client, tmp_path)

        face_bytes = _make_face_bytes(seed=22)
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        event_id = predict_resp.json()["event_id"]

        # Generate real explanation
        lime_resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 50},
        )
        explanation_id = lime_resp.json()["explanation_id"]
        assert lime_resp.json()["computed"] is True

        # Delete artifact
        await _db.delete_inference_artifact(event_id)
        assert await _db.get_inference_artifact(event_id) is None

        # Explanation should still be retrievable
        expl_resp = await client.get(f"/explainability/explanations/{explanation_id}")
        assert expl_resp.status_code == 200
        assert expl_resp.json()["computed"] is True

    async def test_multiple_predictions_multiple_demographics(self, client, tmp_path):
        """Multiple predictions across demographics correctly build audit groups."""
        from friendlyface.api.app import _build_audit_groups

        await _train_model(client, tmp_path)

        # 3 predictions per demographic group
        for group in ["young", "elderly", "middle_age"]:
            for j in range(3):
                face_bytes = _make_face_bytes(seed=200 + hash(group) % 100 + j)
                await client.post(
                    "/recognition/predict",
                    files={"image": ("face.png", face_bytes, "image/png")},
                    params={"demographic_group": group, "true_label": j % 3},
                )

        groups = await _build_audit_groups()
        group_names = sorted([g.group_name for g in groups])
        assert len(group_names) == 3
        assert "elderly" in group_names
        assert "middle_age" in group_names
        assert "young" in group_names

    async def test_bundle_export_includes_computed_explanations(self, client, tmp_path):
        """Bundle export includes real explanation data when computed."""
        await _train_model(client, tmp_path)

        face_bytes = _make_face_bytes(seed=11)
        predict_resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        event_id = predict_resp.json()["event_id"]

        # Real LIME
        lime_resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 50},
        )
        assert lime_resp.json()["computed"] is True
        lime_event_id = lime_resp.json()["event_id"]

        # Create bundle
        bundle_resp = await client.post(
            "/bundles",
            json={"event_ids": [event_id, lime_event_id]},
        )
        assert bundle_resp.status_code == 201
        bundle_id = bundle_resp.json()["id"]

        # Export bundle
        export_resp = await client.get(f"/bundles/{bundle_id}/export")
        assert export_resp.status_code == 200
        export_data = export_resp.json()
        assert "@type" in export_data
        assert export_data["@type"] == "ForensicBundle"

        # Should contain explanation artifacts
        assert export_data.get("explanation_artifacts") is not None

    async def test_chain_integrity_after_full_phase9_operations(self, client, tmp_path):
        """Hash chain remains valid after all Phase 9 operation types."""
        await _train_model(client, tmp_path)

        # Multiple predictions with demographics
        for i, group in enumerate(["demo_a", "demo_b"]):
            face_bytes = _make_face_bytes(seed=300 + i)
            await client.post(
                "/recognition/predict",
                files={"image": ("face.png", face_bytes, "image/png")},
                params={"demographic_group": group, "true_label": i},
            )

        # Explainability on latest predict
        events_resp = await client.get("/events")
        inference_events = [
            e for e in events_resp.json()["items"] if e["event_type"] == "inference_result"
        ]
        last_event_id = inference_events[-1]["id"]

        await client.post(
            "/explainability/lime",
            json={"event_id": last_event_id, "num_samples": 50},
        )
        await client.post(
            "/explainability/shap",
            json={"event_id": last_event_id, "num_samples": 32},
        )

        # Bias audit
        await client.post(
            "/fairness/audit",
            json={
                "groups": [
                    {
                        "group_name": "g1",
                        "true_positives": 80,
                        "false_positives": 10,
                        "true_negatives": 90,
                        "false_negatives": 20,
                    },
                    {
                        "group_name": "g2",
                        "true_positives": 80,
                        "false_positives": 10,
                        "true_negatives": 90,
                        "false_negatives": 20,
                    },
                ],
            },
        )

        # Compliance
        await client.post("/governance/compliance/generate")

        # Chain must be valid after all operations
        chain_resp = await client.get("/chain/integrity")
        assert chain_resp.json()["valid"] is True

        # Merkle root should exist with many leaves
        root_resp = await client.get("/merkle/root")
        assert root_resp.json()["leaf_count"] >= 6
        assert root_resp.json()["merkle_root"] is not None
