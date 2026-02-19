"""ONNX model management: resolution, download, hash verification (US-090).

Provides ``ModelManager`` for resolving the best available ONNX model path
from environment variables, local directories, and a known model registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("friendlyface.recognition.model_manager")

# Default model directory (relative to project root or absolute)
_DEFAULT_MODEL_DIR = "models"

# Known models with download URLs and SHA-256 hashes
_MODEL_REGISTRY: dict[str, dict] = {}
_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "MODEL_REGISTRY.json"


def _load_registry() -> dict[str, dict]:
    """Load model registry from JSON file."""
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY:
        return _MODEL_REGISTRY
    if _REGISTRY_PATH.exists():
        with open(_REGISTRY_PATH) as f:
            _MODEL_REGISTRY = json.load(f)
    return _MODEL_REGISTRY


@dataclass
class ModelInfo:
    """Metadata about a discovered ONNX model."""

    name: str
    path: str
    size_bytes: int
    sha256: str | None = None


class ModelManager:
    """Resolve, download, and verify ONNX face recognition models.

    Resolution order:
    1. ``FF_ONNX_MODEL_PATH`` environment variable (explicit path)
    2. ``model_dir`` parameter / ``FF_MODEL_DIR`` env var
    3. Default ``models/`` directory relative to project root

    Parameters
    ----------
    model_dir : str | None
        Directory to search for models. Falls back to ``FF_MODEL_DIR``
        env var, then ``models/`` relative to project root.
    """

    def __init__(self, model_dir: str | None = None) -> None:
        env_dir = os.environ.get("FF_MODEL_DIR")
        if model_dir:
            self._model_dir = Path(model_dir)
        elif env_dir:
            self._model_dir = Path(env_dir)
        else:
            # Default: models/ relative to project root
            self._model_dir = Path(__file__).resolve().parent.parent.parent / _DEFAULT_MODEL_DIR
        self._registry = _load_registry()

    @property
    def model_dir(self) -> Path:
        """Return the active model directory."""
        return self._model_dir

    def resolve(self) -> str | None:
        """Resolve the best available ONNX model path.

        Returns the path as a string, or None if no model is found.
        """
        # 1. Explicit env var
        env_path = os.environ.get("FF_ONNX_MODEL_PATH")
        if env_path:
            p = Path(env_path)
            if p.exists():
                logger.info("Model resolved from FF_ONNX_MODEL_PATH: %s", p)
                return str(p)
            logger.warning("FF_ONNX_MODEL_PATH set but file not found: %s", env_path)

        # 2. Search model directory for .onnx files
        if self._model_dir.exists():
            onnx_files = sorted(self._model_dir.glob("*.onnx"))
            if onnx_files:
                chosen = str(onnx_files[0])
                logger.info("Model resolved from model dir: %s", chosen)
                return chosen

        logger.info("No ONNX model found — embedding extractor will use fallback")
        return None

    def verify(self, path: str, expected_hash: str) -> bool:
        """Verify a model file against its expected SHA-256 hash.

        Parameters
        ----------
        path : str
            Path to the model file.
        expected_hash : str
            Expected SHA-256 hex digest.

        Returns
        -------
        bool
            True if the hash matches.
        """
        actual = self._compute_hash(path)
        matches = actual == expected_hash.lower()
        if not matches:
            logger.warning(
                "Hash mismatch for %s: expected %s, got %s",
                path,
                expected_hash[:16],
                actual[:16],
            )
        return matches

    def download(
        self,
        model_name: str,
        url: str | None = None,
        expected_hash: str | None = None,
    ) -> Path:
        """Download a model from URL with hash verification.

        Parameters
        ----------
        model_name : str
            Name for the downloaded file (without extension).
        url : str | None
            Download URL. If None, looks up ``model_name`` in the registry.
        expected_hash : str | None
            Expected SHA-256. If None, looks up in registry.

        Returns
        -------
        Path
            Path to the downloaded and verified model file.

        Raises
        ------
        ValueError
            If model_name not in registry and no URL provided.
        RuntimeError
            If download fails or hash verification fails.
        """
        import urllib.request

        # Look up registry if URL not provided
        if url is None:
            reg = self._registry.get(model_name)
            if reg is None:
                msg = f"Model '{model_name}' not in registry and no URL provided"
                raise ValueError(msg)
            url = reg["url"]
            if expected_hash is None:
                expected_hash = reg.get("sha256")

        self._model_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{model_name}.onnx" if not model_name.endswith(".onnx") else model_name
        dest = self._model_dir / filename
        tmp = dest.with_suffix(".tmp")

        logger.info("Downloading model '%s' from %s", model_name, url)
        try:
            urllib.request.urlretrieve(url, str(tmp))  # noqa: S310
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            msg = f"Download failed for '{model_name}': {exc}"
            raise RuntimeError(msg) from exc

        # Verify hash if provided
        if expected_hash:
            actual_hash = self._compute_hash(str(tmp))
            if actual_hash != expected_hash.lower():
                tmp.unlink(missing_ok=True)
                msg = (
                    f"Hash mismatch for '{model_name}': "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )
                raise RuntimeError(msg)

        shutil.move(str(tmp), str(dest))
        logger.info("Model '%s' downloaded to %s (%d bytes)", model_name, dest, dest.stat().st_size)
        return dest

    def list_models(self) -> list[ModelInfo]:
        """List all ONNX models in the model directory."""
        if not self._model_dir.exists():
            return []

        models: list[ModelInfo] = []
        for p in sorted(self._model_dir.glob("*.onnx")):
            models.append(
                ModelInfo(
                    name=p.stem,
                    path=str(p),
                    size_bytes=p.stat().st_size,
                )
            )
        return models

    def get_registry(self) -> dict[str, dict]:
        """Return the known model registry."""
        return dict(self._registry)

    @staticmethod
    def _compute_hash(path: str) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
