"""ML integration tests with real ONNX model (US-095).

These tests require the ``[ml]`` extras (onnxruntime) and a real ONNX model
in ``models/mobilefacenet.onnx``. They are marked with ``@pytest.mark.ml``
and skipped in CI where ML extras are not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip entire module if onnxruntime is not installed
ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mobilefacenet.onnx"

pytestmark = [
    pytest.mark.ml,
    pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"Model not found: {MODEL_PATH}"),
]


def _synthetic_face(seed: int) -> np.ndarray:
    """Create a synthetic 112x112 RGB face image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# EmbeddingExtractor with real model
# ---------------------------------------------------------------------------


class TestONNXEmbeddings:
    def test_loads_onnx_model(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        assert ext.backend == "onnx"
        assert "mobilefacenet" in ext.model_version

    def test_produces_512d_embedding(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        emb = ext.extract(_synthetic_face(1))
        assert emb.dim == 512
        assert emb.vector.shape == (512,)
        assert emb.vector.dtype == np.float32

    def test_embedding_is_l2_normalized(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        emb = ext.extract(_synthetic_face(2))
        norm = float(np.linalg.norm(emb.vector))
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_deterministic_same_input(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        face = _synthetic_face(3)
        e1 = ext.extract(face)
        e2 = ext.extract(face)
        np.testing.assert_array_almost_equal(e1.vector, e2.vector, decimal=5)

    def test_different_faces_different_embeddings(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        e1 = ext.extract(_synthetic_face(10))
        e2 = ext.extract(_synthetic_face(99))
        # Different random images should produce different embeddings
        assert not np.array_equal(e1.vector, e2.vector)

    def test_model_info_has_onnx_details(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        info = ext.model_info
        assert info["backend"] == "onnx"
        assert "input_name" in info
        assert "output_shape" in info

    def test_input_shape_validation(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        wrong_size = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected"):
            ext.extract(wrong_size)

    def test_grayscale_input(self):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        gray = np.random.default_rng(7).integers(0, 256, (112, 112), dtype=np.uint8)
        emb = ext.extract(gray)
        assert emb.dim == 512


# ---------------------------------------------------------------------------
# Full pipeline with ONNX
# ---------------------------------------------------------------------------


class TestONNXPipeline:
    @pytest.fixture()
    async def _db_and_gallery(self, tmp_path):
        """Set up a temp DB + gallery for pipeline tests."""
        from friendlyface.storage.database import Database

        db = Database(str(tmp_path / "test.db"))
        await db.connect()

        from friendlyface.recognition.gallery import FaceGallery

        gallery = FaceGallery(db)
        yield db, gallery
        await db.close()

    async def test_pipeline_end_to_end(self, _db_and_gallery):
        from friendlyface.recognition.embeddings import EmbeddingExtractor
        from friendlyface.recognition.pipeline import RecognitionPipeline

        db, gallery = _db_and_gallery
        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))
        pipeline = RecognitionPipeline(gallery=gallery, extractor=ext)

        face = _synthetic_face(42)
        result = await pipeline.run(face)

        assert result.faces_detected >= 1
        assert result.embedding is not None
        assert result.embedding.dim == 512
        assert result.quality_score >= 0

    async def test_gallery_enroll_and_search(self, _db_and_gallery):
        from friendlyface.recognition.embeddings import EmbeddingExtractor

        db, gallery = _db_and_gallery
        ext = EmbeddingExtractor(model_path=str(MODEL_PATH))

        face = _synthetic_face(50)
        emb = ext.extract(face)

        # Enroll
        entry = await gallery.enroll("test-subject", emb, quality_score=0.9)
        assert entry["subject_id"] == "test-subject"

        # Search with same embedding — should find it
        matches = await gallery.search(emb, top_k=3)
        assert len(matches) >= 1
        assert matches[0].subject_id == "test-subject"
        assert matches[0].similarity > 0.99

    def test_model_manager_resolves_model(self):
        from friendlyface.recognition.model_manager import ModelManager

        manager = ModelManager(model_dir=str(MODEL_PATH.parent))
        resolved = manager.resolve()
        assert resolved is not None
        assert "mobilefacenet" in resolved
