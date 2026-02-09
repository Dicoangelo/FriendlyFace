"""Targeted tests to fill coverage gaps across modules.

Covers:
- Config validators (error paths)
- DID base58 decode + leading-zero edge cases
- Liveness API endpoint
- Detection backend property + heuristic detector
- Embeddings ONNX path (mocked)
- Core service edge cases
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image

import friendlyface.api.app as app_module
from friendlyface.api.app import _dashboard_cache, _db, _service, app, limiter
from friendlyface.recognition.gallery import FaceGallery


# ── Config validators ────────────────────────────────────────────────────


class TestConfigValidators:
    def test_invalid_storage(self):
        from pydantic import ValidationError

        from friendlyface.config import Settings

        with pytest.raises(ValidationError, match="FF_STORAGE"):
            Settings(storage="postgres")

    def test_valid_storage_sqlite(self):
        from friendlyface.config import Settings

        s = Settings(storage="SQLITE")
        assert s.storage == "sqlite"

    def test_valid_storage_supabase(self):
        from friendlyface.config import Settings

        s = Settings(storage="Supabase")
        assert s.storage == "supabase"

    def test_invalid_fl_mode(self):
        from pydantic import ValidationError

        from friendlyface.config import Settings

        with pytest.raises(ValidationError, match="FF_FL_MODE"):
            Settings(fl_mode="distributed")

    def test_invalid_log_format(self):
        from pydantic import ValidationError

        from friendlyface.config import Settings

        with pytest.raises(ValidationError, match="FF_LOG_FORMAT"):
            Settings(log_format="yaml")

    def test_invalid_log_level(self):
        from pydantic import ValidationError

        from friendlyface.config import Settings

        with pytest.raises(ValidationError, match="FF_LOG_LEVEL"):
            Settings(log_level="SUPERVERBOSE")

    def test_did_seed_wrong_length(self):
        from pydantic import ValidationError

        from friendlyface.config import Settings

        with pytest.raises(ValidationError, match="64 hex"):
            Settings(did_seed="abcdef")

    def test_did_seed_not_hex(self):
        from pydantic import ValidationError

        from friendlyface.config import Settings

        with pytest.raises(ValidationError, match="hexadecimal"):
            Settings(did_seed="zz" * 32)

    def test_did_seed_valid(self):
        from friendlyface.config import Settings

        seed = "aa" * 32
        s = Settings(did_seed=seed)
        assert s.did_seed == seed

    def test_api_key_set_empty(self):
        from friendlyface.config import Settings

        s = Settings(api_keys="")
        assert s.api_key_set == set()

    def test_api_key_set_parsed(self):
        from friendlyface.config import Settings

        s = Settings(api_keys="key1, key2, key3")
        assert s.api_key_set == {"key1", "key2", "key3"}

    def test_cors_origin_list(self):
        from friendlyface.config import Settings

        s = Settings(cors_origins="http://a.com, http://b.com")
        assert s.cors_origin_list == ["http://a.com", "http://b.com"]


# ── DID base58 encode/decode roundtrip ───────────────────────────────────


class TestBase58:
    def test_roundtrip(self):
        from friendlyface.crypto.did import _base58_decode, _base58_encode

        data = b"\xed\x01" + b"\x42" * 32
        encoded = _base58_encode(data)
        decoded = _base58_decode(encoded)
        assert decoded == data

    def test_leading_zeros(self):
        from friendlyface.crypto.did import _base58_decode, _base58_encode

        data = b"\x00\x00\x00" + b"\xff" * 10
        encoded = _base58_encode(data)
        assert encoded.startswith("111")
        decoded = _base58_decode(encoded)
        assert decoded == data

    def test_decode_single_char(self):
        from friendlyface.crypto.did import _base58_decode, _base58_encode

        data = b"\x01"
        encoded = _base58_encode(data)
        decoded = _base58_decode(encoded)
        assert decoded == data

    def test_stored_form_roundtrip(self):
        from friendlyface.crypto.did import Ed25519DIDKey

        key = Ed25519DIDKey()
        stored = key.to_stored_form()
        restored = Ed25519DIDKey.from_stored_form(stored["private_key"])
        assert restored.did == key.did
        sig = key.sign(b"roundtrip")
        assert restored.verify(b"roundtrip", sig) is True


# ── Detection module ─────────────────────────────────────────────────────


class TestFaceDetector:
    def test_heuristic_backend(self):
        from friendlyface.recognition.detection import FaceDetector

        detector = FaceDetector()
        assert detector.backend == "heuristic"

    def test_heuristic_detects_face(self):
        from friendlyface.recognition.detection import FaceDetector

        detector = FaceDetector()
        img = np.random.default_rng(1).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        faces = detector.detect(img)
        assert len(faces) == 1
        assert faces[0].confidence == 0.95
        assert faces[0].landmarks is not None
        assert faces[0].aligned is not None
        assert faces[0].aligned.shape == (112, 112, 3)

    def test_quality_score_range(self):
        from friendlyface.recognition.detection import _compute_quality

        img = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        q = _compute_quality(img)
        assert 0.0 <= q <= 1.0

    def test_quality_grayscale(self):
        from friendlyface.recognition.detection import _compute_quality

        gray = np.random.default_rng(3).integers(0, 256, (100, 100), dtype=np.uint8)
        q = _compute_quality(gray)
        assert 0.0 <= q <= 1.0


# ── Embeddings ONNX mock ─────────────────────────────────────────────────


class TestEmbeddingExtractorONNX:
    def test_onnx_path_with_mock(self):
        """Test ONNX extraction path using a mocked session."""
        from friendlyface.recognition.embeddings import EMBEDDING_DIM, EmbeddingExtractor

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        # Return a fake 512-d output
        mock_session.run.return_value = [np.random.default_rng(1).random((1, EMBEDDING_DIM)).astype(np.float32)]

        ext = EmbeddingExtractor()
        ext._session = mock_session
        ext._model_version = "arcface-r100-onnx"

        face = np.random.default_rng(2).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        emb = ext.extract(face)

        assert emb.dim == EMBEDDING_DIM
        assert emb.model_version == "arcface-r100-onnx"
        mock_session.run.assert_called_once()

    def test_onnx_grayscale_path(self):
        """ONNX path with 2D (grayscale) input."""
        from friendlyface.recognition.embeddings import EMBEDDING_DIM, EmbeddingExtractor

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.run.return_value = [np.random.default_rng(3).random((1, EMBEDDING_DIM)).astype(np.float32)]

        ext = EmbeddingExtractor()
        ext._session = mock_session

        gray = np.random.default_rng(4).integers(0, 256, (112, 112), dtype=np.uint8)
        emb = ext.extract(gray)
        assert emb.dim == EMBEDDING_DIM


# ── Liveness API endpoint ────────────────────────────────────────────────


@pytest_asyncio.fixture
async def api_client(tmp_path):
    """HTTP test client for liveness endpoint tests."""
    _db.db_path = tmp_path / "liveness_test.db"
    await _db.connect()
    await _db.run_migrations()
    await _service.initialize()

    from friendlyface.core.merkle import MerkleTree
    from friendlyface.core.provenance import ProvenanceDAG

    _service.merkle = MerkleTree()
    _service._event_index = {}
    _service.provenance = ProvenanceDAG()
    _dashboard_cache["data"] = None
    _dashboard_cache["timestamp"] = 0.0
    app_module._explanations.clear()
    app_module._model_registry.clear()
    app_module._fl_simulations.clear()
    app_module._latest_compliance_report = None

    app_module._gallery = FaceGallery(_db)
    from friendlyface.recognition.pipeline import RecognitionPipeline

    app_module._recognition_pipeline = RecognitionPipeline(gallery=app_module._gallery)
    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


@pytest.mark.asyncio
class TestLivenessAPI:
    async def test_liveness_endpoint_valid_image(self, api_client):
        """POST /api/v1/recognition/liveness with a valid image."""
        # Create a small test image
        img = Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = await api_client.post(
            "/api/v1/recognition/liveness",
            files={"image": ("test.png", buf, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "is_live" in data
        assert "score" in data
        assert "checks" in data
        assert "details" in data
        assert "threshold" in data
        assert data["threshold"] == 0.5

    async def test_liveness_custom_threshold(self, api_client):
        img = Image.fromarray(
            np.random.default_rng(7).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = await api_client.post(
            "/api/v1/recognition/liveness?threshold=0.9",
            files={"image": ("test.png", buf, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["threshold"] == 0.9

    async def test_liveness_empty_image(self, api_client):
        resp = await api_client.post(
            "/api/v1/recognition/liveness",
            files={"image": ("empty.png", b"", "image/png")},
        )
        # FastAPI/Pillow will reject this
        assert resp.status_code in (400, 422, 500)


# ── Provenance edge cases ────────────────────────────────────────────────


class TestProvenanceEdgeCases:
    def test_node_fields(self):
        from friendlyface.core.models import ProvenanceNode

        node = ProvenanceNode(
            entity_type="dataset",
            entity_id="faces-v1",
            metadata={"version": 1},
        )
        assert node.entity_type == "dataset"
        assert node.entity_id == "faces-v1"

    def test_dag_get_chain_nonexistent(self):
        from friendlyface.core.provenance import ProvenanceDAG

        dag = ProvenanceDAG()
        chain = dag.get_chain("nonexistent")
        assert chain == []
