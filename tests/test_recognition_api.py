"""Integration tests for the recognition API endpoints (US-005).

Tests:
  POST /recognition/train    — train PCA+SVM on dataset
  POST /recognition/predict  — predict from image upload
  GET  /recognition/models   — list trained models
  GET  /recognition/models/{id} — get model details + provenance
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

import friendlyface.api.app as app_module
from friendlyface.recognition.pca import IMAGE_SIZE


def _create_test_dataset(
    tmp_path: Path, n_images: int = 12, n_classes: int = 3
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


def _make_test_image(size: tuple[int, int] = IMAGE_SIZE, seed: int = 42) -> bytes:
    """Create a synthetic grayscale image and return raw PNG bytes."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=size, dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POST /recognition/train
# ---------------------------------------------------------------------------


class TestTrainEndpoint:
    async def test_train_success(self, client, tmp_path):
        """Train endpoint creates model and returns metadata + event IDs."""
        dataset_dir, labels = _create_test_dataset(tmp_path)
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
        data = resp.json()

        # Verify all expected fields
        assert "model_id" in data
        assert "pca_event_id" in data
        assert "svm_event_id" in data
        assert "pca_provenance_id" in data
        assert "svm_provenance_id" in data
        assert data["n_components"] == 5
        assert data["n_samples"] == 12
        assert data["n_classes"] == 3
        assert "cv_accuracy" in data
        assert "dataset_hash" in data
        assert "model_hash" in data

    async def test_train_creates_model_files(self, client, tmp_path):
        """Training creates PCA and SVM pkl files on disk."""
        dataset_dir, labels = _create_test_dataset(tmp_path)
        output_dir = tmp_path / "models"

        await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(output_dir),
                "n_components": 5,
                "labels": labels,
            },
        )

        assert (output_dir / "pca.pkl").exists()
        assert (output_dir / "svm.pkl").exists()

    async def test_train_records_forensic_events(self, client, tmp_path):
        """Training records PCA and SVM events in the forensic chain."""
        dataset_dir, labels = _create_test_dataset(tmp_path)
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
        data = resp.json()

        # Verify PCA event exists
        pca_event_resp = await client.get(f"/events/{data['pca_event_id']}")
        assert pca_event_resp.status_code == 200
        pca_event = pca_event_resp.json()
        assert pca_event["event_type"] == "training_complete"
        assert pca_event["payload"]["model_type"] == "PCA"

        # Verify SVM event exists
        svm_event_resp = await client.get(f"/events/{data['svm_event_id']}")
        assert svm_event_resp.status_code == 200
        svm_event = svm_event_resp.json()
        assert svm_event["event_type"] == "training_complete"
        assert svm_event["payload"]["model_type"] == "SVM"

    async def test_train_creates_provenance_nodes(self, client, tmp_path):
        """Training creates provenance chain: PCA -> SVM."""
        dataset_dir, labels = _create_test_dataset(tmp_path)
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
        data = resp.json()

        # SVM provenance chain should include PCA parent
        chain_resp = await client.get(f"/provenance/{data['svm_provenance_id']}")
        assert chain_resp.status_code == 200
        chain = chain_resp.json()
        assert len(chain) == 2
        assert chain[0]["entity_type"] == "training"  # PCA node
        assert chain[1]["entity_type"] == "training"  # SVM node

    async def test_train_invalid_dataset_path(self, client, tmp_path):
        """Returns 400 for nonexistent dataset directory."""
        resp = await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(tmp_path / "nonexistent"),
                "output_dir": str(tmp_path / "models"),
                "labels": [0, 1, 2],
            },
        )
        assert resp.status_code == 400

    async def test_train_missing_labels(self, client, tmp_path):
        """Returns 400 when labels are not provided."""
        dataset_dir, _ = _create_test_dataset(tmp_path)
        resp = await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(tmp_path / "models"),
            },
        )
        assert resp.status_code == 400

    async def test_train_custom_hyperparameters(self, client, tmp_path):
        """Training accepts custom C, kernel, and cv_folds."""
        dataset_dir, labels = _create_test_dataset(tmp_path)
        output_dir = tmp_path / "models"

        resp = await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(output_dir),
                "n_components": 3,
                "C": 10.0,
                "kernel": "rbf",
                "cv_folds": 3,
                "labels": labels,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["n_components"] == 3


# ---------------------------------------------------------------------------
# POST /recognition/predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    async def test_predict_no_models(self, client):
        """Returns 503 when no models are configured and no deep pipeline."""
        import friendlyface.api.app as app_module

        old_pca = app_module._pca_model_path
        old_svm = app_module._svm_model_path
        old_pipeline = app_module._recognition_pipeline
        app_module._pca_model_path = None
        app_module._svm_model_path = None
        app_module._recognition_pipeline = None
        try:
            image_bytes = _make_test_image()
            resp = await client.post(
                "/recognition/predict",
                files={"image": ("face.png", image_bytes, "image/png")},
            )
            assert resp.status_code == 503
        finally:
            app_module._pca_model_path = old_pca
            app_module._svm_model_path = old_svm
            app_module._recognition_pipeline = old_pipeline

    async def test_predict_after_train(self, client, tmp_path, monkeypatch):
        """After training, predict returns matches with event_id (fallback engine)."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "fallback")
        dataset_dir, labels = _create_test_dataset(tmp_path)
        output_dir = tmp_path / "models"

        # Train first
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

        # Predict
        image_bytes = _make_test_image()
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "event_id" in data
        assert "input_hash" in data
        assert "matches" in data
        assert len(data["matches"]) > 0
        assert "label" in data["matches"][0]
        assert "confidence" in data["matches"][0]

    async def test_predict_forensic_event_persisted(self, client, tmp_path, monkeypatch):
        """Prediction event is persisted in the forensic chain."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "fallback")
        dataset_dir, labels = _create_test_dataset(tmp_path)
        output_dir = tmp_path / "models"

        await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(output_dir),
                "n_components": 5,
                "labels": labels,
            },
        )

        image_bytes = _make_test_image()
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", image_bytes, "image/png")},
        )
        data = resp.json()

        event_resp = await client.get(f"/events/{data['event_id']}")
        assert event_resp.status_code == 200
        event_data = event_resp.json()
        assert event_data["event_type"] == "inference_result"
        assert event_data["payload"]["input_hash"] == data["input_hash"]

    async def test_predict_empty_upload(self, client, tmp_path, monkeypatch):
        """Returns 400 for empty image upload."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "fallback")
        dataset_dir, labels = _create_test_dataset(tmp_path)
        output_dir = tmp_path / "models"

        await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(output_dir),
                "n_components": 5,
                "labels": labels,
            },
        )

        resp = await client.post(
            "/recognition/predict",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    async def test_predict_with_top_k(self, client, tmp_path, monkeypatch):
        """Predict respects the top_k query parameter (fallback engine)."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "fallback")
        dataset_dir, labels = _create_test_dataset(tmp_path)
        output_dir = tmp_path / "models"

        await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(output_dir),
                "n_components": 5,
                "labels": labels,
            },
        )

        image_bytes = _make_test_image()
        resp = await client.post(
            "/recognition/predict?top_k=2",
            files={"image": ("face.png", image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["matches"]) == 2


# ---------------------------------------------------------------------------
# GET /recognition/models
# ---------------------------------------------------------------------------


class TestListModelsEndpoint:
    async def test_list_models_empty(self, client):
        """Returns empty list when no models are trained."""
        import friendlyface.api.app as app_module

        app_module._model_registry.clear()
        resp = await client.get("/recognition/models")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_list_models_after_train(self, client, tmp_path):
        """After training, model appears in the listing."""
        import friendlyface.api.app as app_module

        app_module._model_registry.clear()

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
        model_id = train_resp.json()["model_id"]

        resp = await client.get("/recognition/models")
        assert resp.status_code == 200
        models = resp.json()
        assert len(models) == 1
        assert models[0]["id"] == model_id
        assert "n_components" in models[0]
        assert "n_classes" in models[0]
        assert "cv_accuracy" in models[0]
        assert "dataset_hash" in models[0]
        assert "model_hash" in models[0]
        assert "created_at" in models[0]

    async def test_list_models_multiple(self, client, tmp_path):
        """Multiple trainings appear in the listing."""
        import friendlyface.api.app as app_module

        app_module._model_registry.clear()

        dataset_dir, labels = _create_test_dataset(tmp_path)

        # Train twice with different output dirs
        for i in range(2):
            output_dir = tmp_path / f"models_{i}"
            await client.post(
                "/recognition/train",
                json={
                    "dataset_path": str(dataset_dir),
                    "output_dir": str(output_dir),
                    "n_components": 5,
                    "labels": labels,
                },
            )

        resp = await client.get("/recognition/models")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# GET /recognition/models/{id}
# ---------------------------------------------------------------------------


class TestGetModelEndpoint:
    async def test_get_model_not_found(self, client):
        """Returns 404 for unknown model ID."""
        import friendlyface.api.app as app_module

        app_module._model_registry.clear()
        resp = await client.get("/recognition/models/nonexistent-id")
        assert resp.status_code == 404

    async def test_get_model_details(self, client, tmp_path):
        """Returns model metadata for a trained model."""
        import friendlyface.api.app as app_module

        app_module._model_registry.clear()

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
        model_id = train_resp.json()["model_id"]

        resp = await client.get(f"/recognition/models/{model_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == model_id
        assert data["n_components"] == 5
        assert data["n_samples"] == 12
        assert data["n_classes"] == 3

    async def test_get_model_includes_provenance_chain(self, client, tmp_path):
        """Model details include the full provenance chain."""
        import friendlyface.api.app as app_module

        app_module._model_registry.clear()

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
        model_id = train_resp.json()["model_id"]

        resp = await client.get(f"/recognition/models/{model_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "provenance_chain" in data
        assert len(data["provenance_chain"]) == 2  # PCA + SVM

        # First node is PCA training
        assert data["provenance_chain"][0]["entity_type"] == "training"
        assert data["provenance_chain"][0]["metadata"]["model_type"] == "PCA"

        # Second node is SVM training
        assert data["provenance_chain"][1]["entity_type"] == "training"
        assert data["provenance_chain"][1]["metadata"]["model_type"] == "SVM"

    async def test_get_model_event_references(self, client, tmp_path):
        """Model details include forensic event IDs."""
        import friendlyface.api.app as app_module

        app_module._model_registry.clear()

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
        data = train_resp.json()
        model_id = data["model_id"]

        resp = await client.get(f"/recognition/models/{model_id}")
        model = resp.json()
        assert model["pca_event_id"] == data["pca_event_id"]
        assert model["svm_event_id"] == data["svm_event_id"]
        assert model["pca_provenance_id"] == data["pca_provenance_id"]
        assert model["svm_provenance_id"] == data["svm_provenance_id"]


# ---------------------------------------------------------------------------
# Deep / Auto / Fallback engine modes (US-086, US-087)
# ---------------------------------------------------------------------------


def _make_face_bytes(seed: int = 42, size: tuple[int, int] = (200, 200)) -> bytes:
    """Create a synthetic RGB face image as PNG bytes."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(60, 200, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestDeepEngine:
    async def test_predict_deep_returns_enriched_response(self, client, monkeypatch):
        """Deep engine returns liveness, quality, and faces_detected."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "deep")
        face_bytes = _make_face_bytes(seed=1)
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "deep"
        assert "quality_score" in data
        assert "faces_detected" in data
        assert "liveness" in data

    async def test_predict_deep_gallery_match(self, client, monkeypatch):
        """Deep engine finds enrolled face and returns calibration."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "deep")
        face_bytes = _make_face_bytes(seed=10)

        enroll = await client.post(
            "/gallery/enroll",
            files={"image": ("face.png", face_bytes, "image/png")},
            params={"subject_id": "deep_match"},
        )
        assert enroll.status_code == 201

        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        data = resp.json()
        assert len(data["matches"]) >= 1
        assert data["matches"][0]["label"] == "deep_match"
        assert "calibration" in data

    async def test_predict_deep_liveness_fields(self, client, monkeypatch):
        """Liveness result includes is_live, score, and checks."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "deep")
        face_bytes = _make_face_bytes(seed=20)
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        liveness = resp.json()["liveness"]
        assert "is_live" in liveness
        assert "score" in liveness
        assert "checks" in liveness

    async def test_predict_deep_empty_image(self, client, monkeypatch):
        """Empty image returns 400 in deep mode."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "deep")
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert resp.status_code == 400

    async def test_predict_deep_records_event(self, client, monkeypatch):
        """Deep engine records a forensic event with engine=deep in payload."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "deep")
        face_bytes = _make_face_bytes(seed=30)
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        event_id = resp.json()["event_id"]
        event_resp = await client.get(f"/events/{event_id}")
        assert event_resp.status_code == 200
        assert event_resp.json()["payload"]["engine"] == "deep"


class TestAutoEngine:
    async def test_auto_resolves_to_deep(self, client, monkeypatch):
        """Auto mode resolves to deep when pipeline is available."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "auto")
        face_bytes = _make_face_bytes(seed=40)
        resp = await client.post(
            "/recognition/predict",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["engine"] == "deep"

    async def test_auto_resolves_to_fallback(self, client, monkeypatch):
        """Auto mode falls back to PCA+SVM when pipeline is None."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "auto")
        original_pipeline = app_module._recognition_pipeline
        original_pca = app_module._pca_model_path
        original_svm = app_module._svm_model_path
        app_module._recognition_pipeline = None
        app_module._pca_model_path = None
        app_module._svm_model_path = None
        try:
            face_bytes = _make_face_bytes(seed=50)
            resp = await client.post(
                "/recognition/predict",
                files={"image": ("face.png", face_bytes, "image/png")},
            )
            # Without PCA/SVM models, fallback returns 503
            assert resp.status_code == 503
        finally:
            app_module._recognition_pipeline = original_pipeline
            app_module._pca_model_path = original_pca
            app_module._svm_model_path = original_svm


class TestFallbackEngine:
    async def test_fallback_no_models_returns_503(self, client, monkeypatch):
        """Fallback mode returns 503 when no PCA/SVM models are loaded."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "fallback")
        original_pca = app_module._pca_model_path
        original_svm = app_module._svm_model_path
        app_module._pca_model_path = None
        app_module._svm_model_path = None
        try:
            face_bytes = _make_face_bytes(seed=60)
            resp = await client.post(
                "/recognition/predict",
                files={"image": ("face.png", face_bytes, "image/png")},
            )
            assert resp.status_code == 503
            assert "Models not loaded" in resp.json()["detail"]
        finally:
            app_module._pca_model_path = original_pca
            app_module._svm_model_path = original_svm


class TestLegacyRecognize:
    async def test_legacy_uses_active_engine(self, client, monkeypatch):
        """Legacy /recognize endpoint uses the active engine."""
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "deep")
        face_bytes = _make_face_bytes(seed=70)
        resp = await client.post(
            "/recognize",
            files={"image": ("face.png", face_bytes, "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["engine"] == "deep"


class TestResolveEngine:
    def test_default_is_auto(self, monkeypatch):
        monkeypatch.delenv("FF_RECOGNITION_ENGINE", raising=False)
        from friendlyface.config import settings

        assert settings.recognition_engine == "auto"

    def test_auto_resolves_to_deep_with_pipeline(self, monkeypatch):
        monkeypatch.delenv("FF_RECOGNITION_ENGINE", raising=False)
        original = app_module._recognition_pipeline
        app_module._recognition_pipeline = object()  # truthy sentinel
        try:
            assert app_module._resolve_engine() == "deep"
        finally:
            app_module._recognition_pipeline = original

    def test_auto_resolves_to_fallback_without_pipeline(self, monkeypatch):
        monkeypatch.delenv("FF_RECOGNITION_ENGINE", raising=False)
        original = app_module._recognition_pipeline
        app_module._recognition_pipeline = None
        try:
            assert app_module._resolve_engine() == "fallback"
        finally:
            app_module._recognition_pipeline = original

    def test_explicit_deep(self, monkeypatch):
        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "deep")
        assert app_module._resolve_engine() == "deep"
