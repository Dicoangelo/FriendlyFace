"""Tests for face embedding extraction (US-046, US-091)."""

from __future__ import annotations

import numpy as np
import pytest

from friendlyface.recognition.embeddings import (
    EMBEDDING_DIM,
    EmbeddingExtractor,
    FaceEmbedding,
    _infer_model_name,
)


def _aligned_face(seed: int = 0) -> np.ndarray:
    """Create a synthetic 112x112 RGB aligned face crop."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)


class TestFaceEmbedding:
    def test_dim_property(self):
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        emb = FaceEmbedding(vector=vec, model_version="test-v1", input_hash="abc123")
        assert emb.dim == EMBEDDING_DIM

    def test_cosine_similarity_identical(self):
        vec = np.random.default_rng(1).random(EMBEDDING_DIM).astype(np.float32)
        a = FaceEmbedding(vector=vec, model_version="v1", input_hash="a")
        b = FaceEmbedding(vector=vec.copy(), model_version="v1", input_hash="b")
        assert a.cosine_similarity(b) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_similarity_orthogonal(self):
        v1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        v2[1] = 1.0
        a = FaceEmbedding(vector=v1, model_version="v1", input_hash="a")
        b = FaceEmbedding(vector=v2, model_version="v1", input_hash="b")
        assert a.cosine_similarity(b) == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_zero_vector(self):
        zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        non_zero = np.ones(EMBEDDING_DIM, dtype=np.float32)
        a = FaceEmbedding(vector=zero, model_version="v1", input_hash="a")
        b = FaceEmbedding(vector=non_zero, model_version="v1", input_hash="b")
        assert a.cosine_similarity(b) == 0.0

    def test_cosine_similarity_range(self):
        rng = np.random.default_rng(42)
        v1 = rng.random(EMBEDDING_DIM).astype(np.float32)
        v2 = rng.random(EMBEDDING_DIM).astype(np.float32)
        a = FaceEmbedding(vector=v1, model_version="v1", input_hash="a")
        b = FaceEmbedding(vector=v2, model_version="v1", input_hash="b")
        sim = a.cosine_similarity(b)
        assert -1.0 <= sim <= 1.0


class TestEmbeddingExtractor:
    def test_fallback_backend(self):
        ext = EmbeddingExtractor()
        assert ext.backend == "fallback"
        assert ext.model_version == "pca-fallback-v1"

    def test_extract_produces_correct_dim(self):
        ext = EmbeddingExtractor()
        face = _aligned_face(seed=1)
        emb = ext.extract(face)
        assert emb.dim == EMBEDDING_DIM
        assert emb.vector.shape == (EMBEDDING_DIM,)
        assert emb.vector.dtype == np.float32

    def test_extract_normalized(self):
        ext = EmbeddingExtractor()
        face = _aligned_face(seed=2)
        emb = ext.extract(face)
        norm = float(np.linalg.norm(emb.vector))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_extract_deterministic(self):
        ext = EmbeddingExtractor()
        face = _aligned_face(seed=3)
        e1 = ext.extract(face)
        e2 = ext.extract(face)
        np.testing.assert_array_equal(e1.vector, e2.vector)
        assert e1.input_hash == e2.input_hash

    def test_different_faces_different_embeddings(self):
        ext = EmbeddingExtractor()
        e1 = ext.extract(_aligned_face(seed=10))
        e2 = ext.extract(_aligned_face(seed=20))
        assert not np.array_equal(e1.vector, e2.vector)

    def test_extract_grayscale(self):
        ext = EmbeddingExtractor()
        gray = np.random.default_rng(5).integers(0, 256, (112, 112), dtype=np.uint8)
        emb = ext.extract(gray)
        assert emb.dim == EMBEDDING_DIM

    def test_input_hash_is_hex(self):
        ext = EmbeddingExtractor()
        emb = ext.extract(_aligned_face(seed=7))
        assert len(emb.input_hash) == 16
        int(emb.input_hash, 16)  # Should not raise

    def test_model_version_in_embedding(self):
        ext = EmbeddingExtractor()
        emb = ext.extract(_aligned_face())
        assert emb.model_version == ext.model_version

    def test_model_info_fallback(self):
        ext = EmbeddingExtractor()
        info = ext.model_info
        assert info["backend"] == "fallback"
        assert info["model_version"] == "pca-fallback-v1"
        assert info["embedding_dim"] == EMBEDDING_DIM
        assert "model_path" not in info

    def test_model_name_override(self):
        ext = EmbeddingExtractor(model_name="custom-v2")
        # Without a model path + ORT, name is ignored (stays fallback)
        assert ext.model_version == "pca-fallback-v1"

    def test_model_path_without_ort_falls_back(self):
        # When ORT is not available, providing a path should fall back gracefully
        import friendlyface.recognition.embeddings as emb_module

        original = emb_module._HAS_ORT
        emb_module._HAS_ORT = False
        try:
            ext = EmbeddingExtractor(model_path="/nonexistent/model.onnx")
            assert ext.backend == "fallback"
        finally:
            emb_module._HAS_ORT = original


class TestInferModelName:
    def test_mobilefacenet(self):
        assert _infer_model_name("/path/mobilefacenet.onnx") == "onnx-mobilefacenet"

    def test_arcface_r100(self):
        assert _infer_model_name("/models/glintr100.onnx") == "onnx-arcface-r100"

    def test_arcface_r50(self):
        assert _infer_model_name("/models/w600k_r50.onnx") == "onnx-arcface-r50"

    def test_unknown_model(self):
        assert _infer_model_name("/models/custom_face.onnx") == "onnx-custom_face"
