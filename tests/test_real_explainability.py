"""Tests for real explainability computation (US-092 through US-094).

These tests verify that LIME, SHAP, and SDD endpoints compute real
explanations when inference artifacts are available, and fall back
to stub records when they are not.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from friendlyface.recognition.pca import IMAGE_SIZE


@pytest.fixture(autouse=True)
def _force_fallback_engine(monkeypatch):
    """Force fallback engine for PCA+SVM pipeline tests."""
    monkeypatch.setenv("FF_RECOGNITION_ENGINE", "fallback")


def _make_face_bytes() -> bytes:
    """Generate a synthetic 112x112 grayscale face image as PNG bytes."""
    rng = np.random.default_rng(99)
    pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
    img = PILImage.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _train_and_setup(client, tmp_path: Path) -> None:
    """Train a tiny PCA+SVM model via the API and configure model paths."""
    import friendlyface.api.app as app_module

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
    # Train endpoint auto-configures _pca_model_path and _svm_model_path internally
    assert app_module._pca_model_path is not None
    assert app_module._svm_model_path is not None


async def _do_inference(client) -> str:
    """Perform an inference via the API and return the event_id."""
    face_bytes = _make_face_bytes()
    resp = await client.post(
        "/recognition/predict",
        files={"image": ("face.png", face_bytes, "image/png")},
    )
    assert resp.status_code == 200
    return resp.json()["event_id"]


# ---------------------------------------------------------------------------
# LIME (US-092)
# ---------------------------------------------------------------------------


class TestRealLIME:
    async def test_real_lime_computed(self, client, tmp_path):
        """Real LIME explanation includes top_regions and confidence_decomposition."""
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 10, "num_superpixels": 9, "top_k": 3},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["computed"] is True
        assert "top_regions" in data
        assert len(data["top_regions"]) <= 3
        assert "confidence_decomposition" in data
        assert "artifact_hash" in data
        assert data["predicted_label"] is not None

    async def test_lime_top_region_structure(self, client, tmp_path):
        """Each top_region has superpixel_id, importance, bbox."""
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 10, "num_superpixels": 9, "top_k": 3},
        )
        data = resp.json()
        for region in data["top_regions"]:
            assert "superpixel_id" in region
            assert "importance" in region
            assert "bbox" in region
            assert len(region["bbox"]) == 4

    async def test_lime_stub_when_no_artifact(self, client):
        """Without artifact, LIME returns a stub (computed=False)."""
        event_resp = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        event_id = event_resp.json()["id"]

        resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["computed"] is False
        assert "top_regions" not in data

    async def test_lime_persisted_to_db(self, client, tmp_path):
        """Real LIME explanation is persisted to DB."""
        from friendlyface.api.app import _db

        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 10, "num_superpixels": 9},
        )
        explanation_id = resp.json()["explanation_id"]

        record = await _db.get_explanation(explanation_id)
        assert record is not None
        assert record["method"] == "lime"

    async def test_lime_event_not_found(self, client):
        """LIME returns 404 for non-existent event."""
        resp = await client.post(
            "/explainability/lime",
            json={"event_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# SHAP (US-093)
# ---------------------------------------------------------------------------


class TestRealSHAP:
    async def test_real_shap_computed(self, client, tmp_path):
        """Real SHAP explanation includes base_value and top_features."""
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/shap",
            json={"event_id": event_id, "num_samples": 16},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["computed"] is True
        assert "base_value" in data
        assert "top_features" in data
        assert "artifact_hash" in data
        assert "num_features" in data
        assert data["num_features"] > 0

    async def test_shap_top_features_structure(self, client, tmp_path):
        """Each top_feature has feature_index and shap_value."""
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/shap",
            json={"event_id": event_id, "num_samples": 16},
        )
        data = resp.json()
        for feat in data["top_features"]:
            assert "feature_index" in feat
            assert "shap_value" in feat
            assert isinstance(feat["feature_index"], int)
            assert isinstance(feat["shap_value"], float)

    async def test_shap_stub_when_no_artifact(self, client):
        """Without artifact, SHAP returns a stub (computed=False)."""
        event_resp = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        event_id = event_resp.json()["id"]

        resp = await client.post(
            "/explainability/shap",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["computed"] is False
        assert "base_value" not in data

    async def test_shap_persisted_to_db(self, client, tmp_path):
        """Real SHAP explanation is persisted to DB."""
        from friendlyface.api.app import _db

        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/shap",
            json={"event_id": event_id, "num_samples": 16},
        )
        explanation_id = resp.json()["explanation_id"]

        record = await _db.get_explanation(explanation_id)
        assert record is not None
        assert record["method"] == "shap"

    async def test_shap_event_not_found(self, client):
        """SHAP returns 404 for non-existent event."""
        resp = await client.post(
            "/explainability/shap",
            json={"event_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# SDD (US-094)
# ---------------------------------------------------------------------------


class TestRealSDD:
    async def test_real_sdd_computed(self, client, tmp_path):
        """Real SDD explanation includes dominant_region and regions."""
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["computed"] is True
        assert "dominant_region" in data
        assert "regions" in data
        assert "artifact_hash" in data
        assert len(data["regions"]) == 7  # 7 canonical facial regions

    async def test_sdd_region_structure(self, client, tmp_path):
        """Each region has name, importance, bbox, pixel_count."""
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )
        data = resp.json()
        for region in data["regions"]:
            assert "name" in region
            assert "importance" in region
            assert "bbox" in region
            assert "pixel_count" in region
            assert 0.0 <= region["importance"] <= 1.0
            assert len(region["bbox"]) == 4

    async def test_sdd_dominant_region_is_canonical(self, client, tmp_path):
        """Dominant region is one of the 7 canonical facial regions."""
        canonical = {
            "forehead",
            "left_eye",
            "right_eye",
            "nose",
            "mouth",
            "left_jaw",
            "right_jaw",
        }
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )
        data = resp.json()
        assert data["dominant_region"] in canonical

    async def test_sdd_stub_when_no_artifact(self, client):
        """Without artifact, SDD returns a stub (computed=False)."""
        event_resp = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        event_id = event_resp.json()["id"]

        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["computed"] is False
        assert "regions" not in data

    async def test_sdd_persisted_to_db(self, client, tmp_path):
        """Real SDD explanation is persisted to DB."""
        from friendlyface.api.app import _db

        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )
        explanation_id = resp.json()["explanation_id"]

        record = await _db.get_explanation(explanation_id)
        assert record is not None
        assert record["method"] == "sdd"

    async def test_sdd_event_not_found(self, client):
        """SDD returns 404 for non-existent event."""
        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Inference Artifact Storage
# ---------------------------------------------------------------------------


class TestInferenceArtifactStorage:
    async def test_artifact_saved_on_predict(self, client, tmp_path):
        """Inference artifact is saved to DB after prediction."""
        import friendlyface.api.app as app_module
        from friendlyface.api.app import _db

        await _train_and_setup(client, tmp_path)
        pca_path = app_module._pca_model_path
        svm_path = app_module._svm_model_path
        event_id = await _do_inference(client)

        artifact = await _db.get_inference_artifact(event_id)
        assert artifact is not None
        assert artifact["event_id"] == event_id
        assert len(artifact["image_bytes"]) > 0
        assert artifact["pca_model_path"] == pca_path
        assert artifact["svm_model_path"] == svm_path

    async def test_artifact_not_saved_when_disabled(self, client, tmp_path):
        """No artifact saved when FF_STORE_INFERENCE_ARTIFACTS=false."""
        import friendlyface.api.app as app_module
        from friendlyface.api.app import _db

        await _train_and_setup(client, tmp_path)
        old_val = app_module.settings.store_inference_artifacts
        try:
            app_module.settings.store_inference_artifacts = False
            event_id = await _do_inference(client)
            artifact = await _db.get_inference_artifact(event_id)
            assert artifact is None
        finally:
            app_module.settings.store_inference_artifacts = old_val

    async def test_artifact_delete(self, client, tmp_path):
        """Inference artifact can be deleted."""
        from friendlyface.api.app import _db

        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        assert await _db.get_inference_artifact(event_id) is not None
        await _db.delete_inference_artifact(event_id)
        assert await _db.get_inference_artifact(event_id) is None


# ---------------------------------------------------------------------------
# Compare endpoint with real computations
# ---------------------------------------------------------------------------


class TestCompareWithRealExplanations:
    async def test_compare_all_methods(self, client, tmp_path):
        """Compare endpoint returns all three methods after real computation."""
        await _train_and_setup(client, tmp_path)
        event_id = await _do_inference(client)

        # Trigger all three
        await client.post(
            "/explainability/lime",
            json={"event_id": event_id, "num_samples": 10, "num_superpixels": 9},
        )
        await client.post(
            "/explainability/shap",
            json={"event_id": event_id, "num_samples": 16},
        )
        await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )

        resp = await client.get(f"/explainability/compare/{event_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_lime"] >= 1
        assert data["total_shap"] >= 1
        assert data["total_sdd"] >= 1
