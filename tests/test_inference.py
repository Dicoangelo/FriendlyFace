"""Tests for the face inference pipeline and /recognize API endpoint."""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from friendlyface.core.models import EventType
from friendlyface.recognition.inference import (
    InferenceResult,
    Match,
    _hash_bytes,
    _preprocess_image,
    load_pca_model,
    load_svm_model,
    run_inference,
)
from friendlyface.recognition.pca import IMAGE_SIZE


def _make_test_image(size: tuple[int, int] = IMAGE_SIZE, seed: int = 42) -> bytes:
    """Create a synthetic grayscale image and return raw PNG bytes."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=size, dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _train_and_save_models(
    tmp_path: Path,
    n_samples: int = 20,
    n_classes: int = 4,
    n_components: int = 5,
) -> tuple[Path, Path]:
    """Train minimal PCA+SVM models and save them for inference tests.

    Returns (pca_path, svm_path).
    """
    rng = np.random.default_rng(42)
    from friendlyface.recognition.pca import EXPECTED_PIXELS

    # Generate synthetic face data
    data = rng.random((n_samples, EXPECTED_PIXELS))
    labels = np.array([i % n_classes for i in range(n_samples)])

    # Train PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)
    pca_blob = {
        "pca": pca,
        "n_components": n_components,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_samples": n_samples,
        "dataset_hash": "test_hash_abc",
        "image_size": IMAGE_SIZE,
    }
    pca_path = tmp_path / "pca.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump(pca_blob, f)

    # Train SVM on PCA-reduced features
    reduced = pca.transform(data)
    svm = SVC(kernel="linear")
    svm.fit(reduced, labels)
    svm_blob = {
        "svm": svm,
        "hyperparameters": {"C": 1.0, "kernel": "linear"},
        "n_samples": n_samples,
        "n_features": n_components,
        "n_classes": n_classes,
        "cv_accuracy": 0.9,
        "cv_scores": [0.9, 0.9],
        "model_hash": "test_model_hash",
    }
    svm_path = tmp_path / "svm.pkl"
    with open(svm_path, "wb") as f:
        pickle.dump(svm_blob, f)

    return pca_path, svm_path


# ---------------------------------------------------------------------------
# Unit tests: model loading
# ---------------------------------------------------------------------------


class TestLoadModels:
    def test_load_pca_model(self, tmp_path):
        pca_path, _ = _train_and_save_models(tmp_path)
        blob = load_pca_model(pca_path)
        assert "pca" in blob
        assert isinstance(blob["pca"], PCA)

    def test_load_svm_model(self, tmp_path):
        _, svm_path = _train_and_save_models(tmp_path)
        blob = load_svm_model(svm_path)
        assert "svm" in blob
        assert isinstance(blob["svm"], SVC)

    def test_load_pca_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pca_model(tmp_path / "nonexistent.pkl")

    def test_load_svm_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_svm_model(tmp_path / "nonexistent.pkl")

    def test_load_pca_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        with open(bad, "wb") as f:
            pickle.dump({"not_pca": True}, f)
        with pytest.raises(ValueError, match="missing 'pca' key"):
            load_pca_model(bad)

    def test_load_svm_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        with open(bad, "wb") as f:
            pickle.dump({"not_svm": True}, f)
        with pytest.raises(ValueError, match="missing 'svm' key"):
            load_svm_model(bad)


# ---------------------------------------------------------------------------
# Unit tests: preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def test_correct_shape(self):
        image_bytes = _make_test_image(size=IMAGE_SIZE)
        features = _preprocess_image(image_bytes)
        assert features.shape == (1, IMAGE_SIZE[0] * IMAGE_SIZE[1])
        assert features.dtype == np.float64

    def test_resizes_non_standard(self):
        """Images not 112x112 are resized."""
        image_bytes = _make_test_image(size=(200, 200))
        features = _preprocess_image(image_bytes)
        assert features.shape == (1, IMAGE_SIZE[0] * IMAGE_SIZE[1])

    def test_hash_deterministic(self):
        image_bytes = _make_test_image()
        h1 = _hash_bytes(image_bytes)
        h2 = _hash_bytes(image_bytes)
        assert h1 == h2
        assert len(h1) == 64


# ---------------------------------------------------------------------------
# Unit tests: inference pipeline
# ---------------------------------------------------------------------------


class TestRunInference:
    def test_basic_inference(self, tmp_path):
        pca_path, svm_path = _train_and_save_models(tmp_path)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path)

        assert isinstance(result, InferenceResult)
        assert len(result.matches) > 0
        assert all(isinstance(m, Match) for m in result.matches)

    def test_top_k_respected(self, tmp_path):
        pca_path, svm_path = _train_and_save_models(tmp_path, n_classes=4)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path, top_k=2)
        assert len(result.matches) == 2

    def test_top_k_clamped_to_classes(self, tmp_path):
        """top_k larger than n_classes returns n_classes matches."""
        n_classes = 3
        pca_path, svm_path = _train_and_save_models(tmp_path, n_classes=n_classes)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path, top_k=100)
        assert len(result.matches) == n_classes

    def test_matches_sorted_by_confidence(self, tmp_path):
        pca_path, svm_path = _train_and_save_models(tmp_path)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path)
        confidences = [m.confidence for m in result.matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_input_hash_computed(self, tmp_path):
        pca_path, svm_path = _train_and_save_models(tmp_path)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path)
        assert result.input_hash == _hash_bytes(image_bytes)
        assert len(result.input_hash) == 64

    def test_forensic_event_created(self, tmp_path):
        pca_path, svm_path = _train_and_save_models(tmp_path)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path)

        event = result.inference_event
        assert event.event_type == EventType.INFERENCE_RESULT
        assert event.event_hash != ""
        assert event.verify()
        assert event.payload["input_hash"] == result.input_hash
        assert "matches" in event.payload
        assert event.payload["prediction"] in [m.label for m in result.matches]

    def test_forensic_event_chain(self, tmp_path):
        """Inference event respects hash chaining parameters."""
        pca_path, svm_path = _train_and_save_models(tmp_path)
        image_bytes = _make_test_image()

        result = run_inference(
            image_bytes,
            pca_path,
            svm_path,
            previous_hash="prev_hash_123",
            sequence_number=5,
        )

        assert result.inference_event.previous_hash == "prev_hash_123"
        assert result.inference_event.sequence_number == 5

    def test_provenance_node_created(self, tmp_path):
        pca_path, svm_path = _train_and_save_models(tmp_path)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path)

        node = result.provenance_node
        assert node.entity_type == "inference"
        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()
        assert node.metadata["input_hash"] == result.input_hash

    def test_provenance_node_with_parent(self, tmp_path):
        pca_path, svm_path = _train_and_save_models(tmp_path)
        image_bytes = _make_test_image()
        parent_id = uuid4()

        result = run_inference(
            image_bytes,
            pca_path,
            svm_path,
            model_provenance_id=parent_id,
        )

        node = result.provenance_node
        assert parent_id in node.parents
        assert len(node.relations) == 1

    def test_binary_classification(self, tmp_path):
        """Inference works with exactly 2 classes (binary SVM)."""
        pca_path, svm_path = _train_and_save_models(tmp_path, n_classes=2, n_samples=10)
        image_bytes = _make_test_image()

        result = run_inference(image_bytes, pca_path, svm_path, top_k=2)
        assert len(result.matches) == 2
        # Confidences should sum to ~1.0 for binary
        total = sum(m.confidence for m in result.matches)
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Integration test: POST /recognize
# ---------------------------------------------------------------------------


class TestRecognizeEndpoint:
    async def test_recognize_no_models_configured(self, client):
        """Returns 503 when model paths are not set and no deep pipeline."""
        import friendlyface.api.app as app_module

        # Ensure model paths and pipeline are None
        old_pca = app_module._pca_model_path
        old_svm = app_module._svm_model_path
        old_pipeline = app_module._recognition_pipeline
        app_module._pca_model_path = None
        app_module._svm_model_path = None
        app_module._recognition_pipeline = None
        try:
            image_bytes = _make_test_image()
            resp = await client.post(
                "/recognize",
                files={"image": ("face.png", image_bytes, "image/png")},
            )
            assert resp.status_code == 503
        finally:
            app_module._pca_model_path = old_pca
            app_module._svm_model_path = old_svm
            app_module._recognition_pipeline = old_pipeline

    async def test_recognize_success(self, client, tmp_path, monkeypatch):
        """Upload image -> get prediction + verify forensic event exists."""
        import friendlyface.api.app as app_module

        monkeypatch.setenv("FF_RECOGNITION_ENGINE", "fallback")
        pca_path, svm_path = _train_and_save_models(tmp_path)
        app_module._pca_model_path = str(pca_path)
        app_module._svm_model_path = str(svm_path)
        try:
            image_bytes = _make_test_image()
            resp = await client.post(
                "/recognize",
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

            # Verify the forensic event was persisted
            event_resp = await client.get(f"/events/{data['event_id']}")
            assert event_resp.status_code == 200
            event_data = event_resp.json()
            assert event_data["event_type"] == "inference_result"
            assert event_data["payload"]["input_hash"] == data["input_hash"]
        finally:
            app_module._pca_model_path = None
            app_module._svm_model_path = None

    async def test_recognize_empty_upload(self, client, tmp_path):
        """Returns 400 for empty image upload."""
        import friendlyface.api.app as app_module

        pca_path, svm_path = _train_and_save_models(tmp_path)
        app_module._pca_model_path = str(pca_path)
        app_module._svm_model_path = str(svm_path)
        try:
            resp = await client.post(
                "/recognize",
                files={"image": ("empty.png", b"", "image/png")},
            )
            assert resp.status_code == 400
        finally:
            app_module._pca_model_path = None
            app_module._svm_model_path = None
