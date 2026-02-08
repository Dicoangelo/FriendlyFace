"""Tests for face embedding extraction (US-046)."""

from __future__ import annotations

import numpy as np
import pytest

from friendlyface.recognition.embeddings import (
    EMBEDDING_DIM,
    EmbeddingExtractor,
    FaceEmbedding,
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
