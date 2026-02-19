"""ArcFace deep face embeddings (US-046, US-091).

Produces 512-dimensional face embeddings using an ArcFace ONNX model
when ``onnxruntime`` is available, or a PCA-based fallback for testing.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("friendlyface.recognition.embeddings")

EMBEDDING_DIM = 512
_EXPECTED_INPUT_SIZE = (112, 112)

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False


@dataclass
class FaceEmbedding:
    """A 512-d face embedding with metadata."""

    vector: np.ndarray  # (512,) float32
    model_version: str
    input_hash: str

    @property
    def dim(self) -> int:
        return len(self.vector)

    def cosine_similarity(self, other: FaceEmbedding) -> float:
        """Compute cosine similarity with another embedding."""
        a = self.vector
        b = other.vector
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm == 0:
            return 0.0
        return dot / norm


def _infer_model_name(model_path: str) -> str:
    """Infer a human-readable model name from the file path."""
    stem = Path(model_path).stem.lower()
    if "mobilefacenet" in stem or "mobile" in stem:
        return "onnx-mobilefacenet"
    if "r100" in stem or "buffalo_l" in stem or "glintr100" in stem:
        return "onnx-arcface-r100"
    if "r50" in stem or "w600k" in stem:
        return "onnx-arcface-r50"
    return f"onnx-{stem}"


class EmbeddingExtractor:
    """Extract face embeddings from aligned 112x112 face crops.

    Uses ONNX Runtime with an ArcFace model when available, otherwise
    falls back to a deterministic PCA-like projection for testing.
    """

    def __init__(self, model_path: str | None = None, model_name: str | None = None) -> None:
        self._session = None
        self._model_version = "pca-fallback-v1"
        self._model_path: str | None = None

        if model_path and _HAS_ORT:
            t0 = time.monotonic()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options=opts)
            elapsed = time.monotonic() - t0

            self._model_path = model_path
            self._model_version = model_name or _infer_model_name(model_path)

            # Log model metadata
            inp = self._session.get_inputs()[0]
            out = self._session.get_outputs()[0]
            logger.info(
                "ONNX model loaded: %s (%.2fs) | input=%s %s | output=%s %s",
                self._model_version,
                elapsed,
                inp.name,
                inp.shape,
                out.name,
                out.shape,
            )
        elif model_path and not _HAS_ORT:
            logger.warning("Model path provided but onnxruntime not installed — using fallback")
        elif not _HAS_ORT:
            logger.info("onnxruntime not available — using PCA fallback embeddings")

    @property
    def backend(self) -> str:
        return "onnx" if self._session is not None else "fallback"

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_info(self) -> dict:
        """Return model metadata for forensic logging and health checks."""
        info: dict = {
            "backend": self.backend,
            "model_version": self._model_version,
            "embedding_dim": EMBEDDING_DIM,
        }
        if self._session is not None and self._model_path:
            info["model_path"] = self._model_path
            inp = self._session.get_inputs()[0]
            out = self._session.get_outputs()[0]
            info["input_name"] = inp.name
            info["input_shape"] = inp.shape
            info["output_name"] = out.name
            info["output_shape"] = out.shape
        return info

    def extract(self, aligned_face: np.ndarray) -> FaceEmbedding:
        """Extract a 512-d embedding from an aligned face crop.

        Args:
            aligned_face: (112, 112) or (112, 112, 3) uint8 array.

        Returns:
            FaceEmbedding with normalized 512-d vector.
        """
        input_hash = hashlib.sha256(aligned_face.tobytes()).hexdigest()[:16]

        if self._session is not None:
            vector = self._extract_onnx(aligned_face)
        else:
            vector = self._extract_fallback(aligned_face)

        # L2-normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return FaceEmbedding(
            vector=vector.astype(np.float32),
            model_version=self._model_version,
            input_hash=input_hash,
        )

    def _extract_onnx(self, aligned_face: np.ndarray) -> np.ndarray:
        """Run ONNX inference for ArcFace embedding."""
        # Validate input dimensions
        h, w = aligned_face.shape[:2]
        if (h, w) != _EXPECTED_INPUT_SIZE:
            msg = f"Expected {_EXPECTED_INPUT_SIZE} input, got ({h}, {w})"
            raise ValueError(msg)

        # Prepare input: (1, 3, 112, 112) float32
        if aligned_face.ndim == 2:
            img = np.stack([aligned_face] * 3, axis=0)
        else:
            img = aligned_face.transpose(2, 0, 1)

        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        img = img[np.newaxis, ...]

        input_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]
        outputs = self._session.run(None, {input_name: img})  # type: ignore[union-attr]
        return outputs[0].flatten()[:EMBEDDING_DIM]

    def _extract_fallback(self, aligned_face: np.ndarray) -> np.ndarray:
        """Deterministic fallback embedding using image statistics.

        Produces a reproducible 512-d vector from pixel data without
        requiring a trained model. Not suitable for real recognition.
        """
        gray = aligned_face
        if gray.ndim == 3:
            gray = np.mean(aligned_face, axis=2)

        # Flatten and pad/truncate to EMBEDDING_DIM
        flat = gray.astype(np.float64).ravel()

        # Use DCT-like frequency decomposition
        # Split into blocks, compute stats per block
        n_blocks = EMBEDDING_DIM
        block_size = max(len(flat) // n_blocks, 1)
        features = np.zeros(EMBEDDING_DIM, dtype=np.float64)

        for i in range(min(n_blocks, len(flat) // block_size)):
            block = flat[i * block_size : (i + 1) * block_size]
            features[i] = np.mean(block)

        return features
