"""Pluggable recognition engine interface (US-007).

Defines ``RecognitionEngine`` ABC and three concrete implementations:

- **FallbackEngine** — wraps existing PCA+SVM pipeline (``inference.py``).
- **ONNXEngine** — loads ArcFace/AdaFace ONNX models for production inference.
- **ProxyEngine** — delegates to the compliance proxy (US-005) for external APIs.

Engine selection is driven by ``FF_RECOGNITION_ENGINE`` (fallback|onnx|proxy).
All engines produce identical forensic events so the forensic layer is engine-agnostic.
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from friendlyface.core.models import EventType, ForensicEvent, ProvenanceNode, ProvenanceRelation

logger = logging.getLogger("friendlyface.recognition.engine")


# ---------------------------------------------------------------------------
# Shared data classes
# ---------------------------------------------------------------------------


@dataclass
class EngineMatch:
    """A single recognition match produced by any engine."""

    label: int | str
    confidence: float


@dataclass
class EngineResult:
    """Unified result returned by every recognition engine."""

    matches: list[EngineMatch]
    input_hash: str
    engine_name: str
    model_info: dict[str, Any]
    inference_event: ForensicEvent
    provenance_node: ProvenanceNode


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class RecognitionEngine(ABC):
    """Abstract base class for pluggable recognition engines.

    Every engine must implement three methods so callers can train, predict,
    and introspect the underlying model without knowing which backend is active.
    """

    @abstractmethod
    def train(
        self,
        images: list[np.ndarray],
        labels: list[int | str],
        *,
        output_dir: Path | None = None,
        actor: str = "engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> dict[str, Any]:
        """Train the recognition model.

        Parameters
        ----------
        images : list[np.ndarray]
            Training images (grayscale or RGB depending on engine).
        labels : list[int | str]
            Corresponding identity labels.
        output_dir : Path | None
            Where to persist trained artefacts.
        actor, previous_hash, sequence_number :
            Forensic chain parameters.

        Returns
        -------
        dict with at least ``{"model_hash": str, "forensic_event": ForensicEvent}``.
        """

    @abstractmethod
    def predict(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        actor: str = "engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> EngineResult:
        """Run recognition on a single image.

        Parameters
        ----------
        image_bytes : bytes
            Raw image bytes.
        top_k : int
            Maximum matches to return.
        actor, previous_hash, sequence_number :
            Forensic chain parameters.

        Returns
        -------
        EngineResult with matches, forensic event, and provenance node.
        """

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the currently loaded model.

        Must include at least ``{"engine": str, "ready": bool}``.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_forensic_event(
    *,
    input_hash: str,
    payload: dict[str, Any],
    actor: str,
    previous_hash: str,
    sequence_number: int,
) -> ForensicEvent:
    """Build a sealed forensic event for an inference result."""
    return ForensicEvent(
        event_type=EventType.INFERENCE_RESULT,
        actor=actor,
        payload=payload,
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()


def _make_provenance_node(
    *,
    event: ForensicEvent,
    input_hash: str,
    prediction: int | str,
    top_k: int,
    confidence: float,
    model_provenance_id: Any | None = None,
) -> ProvenanceNode:
    """Build a sealed provenance node for an inference."""
    parent_ids = []
    relations = []
    if model_provenance_id is not None:
        parent_ids = [model_provenance_id]
        relations = [ProvenanceRelation.GENERATED_BY]

    return ProvenanceNode(
        entity_type="inference",
        entity_id=str(event.id),
        metadata={
            "input_hash": input_hash,
            "prediction": prediction,
            "top_k": top_k,
            "confidence": confidence,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()


# ---------------------------------------------------------------------------
# FallbackEngine — wraps existing PCA+SVM pipeline
# ---------------------------------------------------------------------------


class FallbackEngine(RecognitionEngine):
    """Wraps the existing PCA+SVM inference pipeline without modification.

    This is the default engine and preserves all current behaviour.
    """

    def __init__(
        self,
        pca_model_path: Path | None = None,
        svm_model_path: Path | None = None,
    ) -> None:
        self._pca_path = pca_model_path
        self._svm_path = svm_model_path

    # -- RecognitionEngine interface ----------------------------------------

    def train(
        self,
        images: list[np.ndarray],
        labels: list[int | str],
        *,
        output_dir: Path | None = None,
        actor: str = "fallback_engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> dict[str, Any]:
        """Train PCA+SVM on provided images/labels.

        Delegates to ``friendlyface.recognition.pca`` and
        ``friendlyface.recognition.svm`` for the actual computation.
        """
        from friendlyface.recognition.pca import PCA as SklearnPCA  # noqa: N811

        # delayed import to avoid circular deps
        from sklearn.svm import SVC

        if output_dir is None:
            raise ValueError("output_dir is required for FallbackEngine.train()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stack images into feature matrix
        X = np.stack([img.ravel().astype(np.float64) for img in images])
        y = np.array(labels)

        # PCA
        n_components = min(50, X.shape[0], X.shape[1])
        pca = SklearnPCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)

        # SVM
        svm = SVC(kernel="linear", probability=False)
        svm.fit(X_reduced, y)

        # Persist
        import pickle

        pca_path = output_dir / "pca_model.pkl"
        svm_path = output_dir / "svm_model.pkl"

        pca_blob = {"pca": pca, "n_components": n_components}
        svm_blob = {"svm": svm, "hyperparameters": {"kernel": "linear"}}

        with open(pca_path, "wb") as f:
            pickle.dump(pca_blob, f)
        with open(svm_path, "wb") as f:
            pickle.dump(svm_blob, f)

        self._pca_path = pca_path
        self._svm_path = svm_path

        model_hash = _hash_bytes(pickle.dumps(svm))

        event = ForensicEvent(
            event_type=EventType.TRAINING_COMPLETE,
            actor=actor,
            payload={
                "engine": "fallback",
                "n_samples": len(images),
                "n_components": n_components,
                "model_hash": model_hash,
            },
            previous_hash=previous_hash,
            sequence_number=sequence_number,
        ).seal()

        return {"model_hash": model_hash, "forensic_event": event}

    def predict(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        actor: str = "fallback_engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> EngineResult:
        """Run PCA+SVM inference via the existing pipeline."""
        from friendlyface.recognition.inference import (
            load_pca_model,
            load_svm_model,
            _preprocess_image,
        )

        if self._pca_path is None or self._svm_path is None:
            raise RuntimeError(
                "FallbackEngine: PCA and SVM model paths must be set. "
                "Call train() first or pass paths to __init__."
            )

        input_hash = _hash_bytes(image_bytes)

        pca_blob = load_pca_model(self._pca_path)
        svm_blob = load_svm_model(self._svm_path)
        pca = pca_blob["pca"]
        svm = svm_blob["svm"]

        features = _preprocess_image(image_bytes)
        reduced = pca.transform(features)

        prediction = int(svm.predict(reduced)[0])
        classes = svm.classes_

        if len(classes) == 2:
            raw_scores = svm.decision_function(reduced).ravel()
            scores = 1.0 / (1.0 + np.exp(-raw_scores))
            match_scores = {classes[1]: float(scores[0]), classes[0]: float(1.0 - scores[0])}
        else:
            raw_scores = svm.decision_function(reduced).ravel()
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            probs = exp_scores / exp_scores.sum()
            match_scores = {cls: float(prob) for cls, prob in zip(classes, probs)}

        sorted_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
        actual_k = min(top_k, len(sorted_matches))
        matches = [
            EngineMatch(label=int(lbl), confidence=conf)
            for lbl, conf in sorted_matches[:actual_k]
        ]

        event = _make_forensic_event(
            input_hash=input_hash,
            payload={
                "engine": "fallback",
                "input_hash": input_hash,
                "pca_model_path": str(self._pca_path),
                "svm_model_path": str(self._svm_path),
                "prediction": prediction,
                "top_k": actual_k,
                "matches": [{"label": m.label, "confidence": m.confidence} for m in matches],
            },
            actor=actor,
            previous_hash=previous_hash,
            sequence_number=sequence_number,
        )

        provenance = _make_provenance_node(
            event=event,
            input_hash=input_hash,
            prediction=prediction,
            top_k=actual_k,
            confidence=matches[0].confidence if matches else 0.0,
        )

        return EngineResult(
            matches=matches,
            input_hash=input_hash,
            engine_name="fallback",
            model_info=self.get_model_info(),
            inference_event=event,
            provenance_node=provenance,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "engine": "fallback",
            "ready": self._pca_path is not None and self._svm_path is not None,
            "pca_model_path": str(self._pca_path) if self._pca_path else None,
            "svm_model_path": str(self._svm_path) if self._svm_path else None,
        }


# ---------------------------------------------------------------------------
# ONNXEngine — ArcFace / AdaFace ONNX models
# ---------------------------------------------------------------------------


class ONNXEngine(RecognitionEngine):
    """Loads ArcFace or AdaFace ONNX models for production inference.

    Requires ``onnxruntime`` to be installed. Raises a helpful ``ImportError``
    if the package is missing.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError:
            raise ImportError(
                "ONNXEngine requires 'onnxruntime'. Install it with:\n"
                "  pip install onnxruntime          # CPU\n"
                "  pip install onnxruntime-gpu       # GPU (CUDA)\n"
                "See https://onnxruntime.ai/docs/install/ for details."
            ) from None

        self._ort = ort
        self._model_path = Path(model_path) if model_path else None
        self._session: Any = None

        if self._model_path and self._model_path.exists():
            self._load_model(self._model_path)

    def _load_model(self, path: Path) -> None:
        """Create an ONNX InferenceSession from the given model file."""
        self._session = self._ort.InferenceSession(
            str(path),
            providers=self._ort.get_available_providers(),
        )
        self._model_path = path
        logger.info("ONNXEngine loaded model: %s", path)

    def train(
        self,
        images: list[np.ndarray],
        labels: list[int | str],
        *,
        output_dir: Path | None = None,
        actor: str = "onnx_engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> dict[str, Any]:
        """ONNX models are pre-trained; this builds a gallery index.

        For ONNX engines, 'training' means computing embeddings for all known
        identities and storing them for nearest-neighbour lookup at predict time.
        """
        if self._session is None:
            raise RuntimeError("ONNXEngine: no model loaded. Set model_path or call _load_model().")

        embeddings = []
        for img in images:
            emb = self._extract_embedding(img)
            embeddings.append(emb)

        gallery = {
            "embeddings": np.stack(embeddings),
            "labels": list(labels),
        }
        self._gallery = gallery

        event = ForensicEvent(
            event_type=EventType.TRAINING_COMPLETE,
            actor=actor,
            payload={
                "engine": "onnx",
                "model_path": str(self._model_path),
                "n_identities": len(labels),
            },
            previous_hash=previous_hash,
            sequence_number=sequence_number,
        ).seal()

        return {"model_hash": _hash_bytes(np.stack(embeddings).tobytes()), "forensic_event": event}

    def predict(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        actor: str = "onnx_engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> EngineResult:
        """Run ONNX model inference and cosine-similarity gallery search."""
        if self._session is None:
            raise RuntimeError("ONNXEngine: no model loaded.")

        import io
        from PIL import Image

        input_hash = _hash_bytes(image_bytes)

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_arr = np.array(img)
        query_emb = self._extract_embedding(img_arr)

        # Gallery search via cosine similarity
        gallery = getattr(self, "_gallery", None)
        if gallery is None or len(gallery["labels"]) == 0:
            matches: list[EngineMatch] = []
            prediction: int | str = -1
        else:
            gallery_embs = gallery["embeddings"]
            sims = gallery_embs @ query_emb / (
                np.linalg.norm(gallery_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
            )
            top_indices = np.argsort(-sims)[:top_k]
            matches = [
                EngineMatch(label=gallery["labels"][i], confidence=float(sims[i]))
                for i in top_indices
            ]
            prediction = matches[0].label if matches else -1

        event = _make_forensic_event(
            input_hash=input_hash,
            payload={
                "engine": "onnx",
                "input_hash": input_hash,
                "model_path": str(self._model_path),
                "prediction": prediction,
                "top_k": min(top_k, len(matches)),
                "matches": [{"label": m.label, "confidence": m.confidence} for m in matches],
            },
            actor=actor,
            previous_hash=previous_hash,
            sequence_number=sequence_number,
        )

        provenance = _make_provenance_node(
            event=event,
            input_hash=input_hash,
            prediction=prediction,
            top_k=min(top_k, len(matches)),
            confidence=matches[0].confidence if matches else 0.0,
        )

        return EngineResult(
            matches=matches,
            input_hash=input_hash,
            engine_name="onnx",
            model_info=self.get_model_info(),
            inference_event=event,
            provenance_node=provenance,
        )

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Run the ONNX model to get a face embedding vector."""
        from PIL import Image as PILImage

        # Resize to model input (112x112 for ArcFace, AdaFace)
        if image.shape[:2] != (112, 112):
            pil_img = PILImage.fromarray(image).resize((112, 112), PILImage.Resampling.LANCZOS)
            image = np.array(pil_img)

        # CHW, float32, normalised to [-1, 1]
        blob = image.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
        blob = np.expand_dims(blob, axis=0)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})
        embedding = outputs[0].flatten().astype(np.float32)

        # L2-normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_model_info(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "engine": "onnx",
            "ready": self._session is not None,
            "model_path": str(self._model_path) if self._model_path else None,
        }
        if self._session is not None:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            meta["input_shape"] = [i.shape for i in inputs]
            meta["output_shape"] = [o.shape for o in outputs]
            meta["providers"] = self._session.get_providers()
        return meta


# ---------------------------------------------------------------------------
# ProxyEngine — delegates to compliance proxy (US-005)
# ---------------------------------------------------------------------------


class ProxyEngine(RecognitionEngine):
    """Delegates recognition to an external API via the compliance proxy.

    The proxy handles forensic logging of the upstream request/response,
    but this engine wraps the result into the standard ``EngineResult`` so
    callers see a uniform interface.
    """

    def __init__(
        self,
        upstream_url: str | None = None,
        upstream_key: str | None = None,
    ) -> None:
        from friendlyface.config import settings

        self._upstream_url = upstream_url or settings.proxy_upstream_url
        self._upstream_key = upstream_key or settings.proxy_upstream_key

        if not self._upstream_url:
            logger.warning(
                "ProxyEngine: FF_PROXY_UPSTREAM_URL not set. "
                "predict() will fail until configured."
            )

    def train(
        self,
        images: list[np.ndarray],
        labels: list[int | str],
        *,
        output_dir: Path | None = None,
        actor: str = "proxy_engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> dict[str, Any]:
        """Proxy engines do not support local training.

        Raises ``NotImplementedError`` — training is managed by the upstream service.
        """
        raise NotImplementedError(
            "ProxyEngine does not support local training. "
            "Training is managed by the upstream recognition service."
        )

    def predict(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        actor: str = "proxy_engine",
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> EngineResult:
        """Forward recognition to the upstream API synchronously.

        Uses ``httpx`` to POST the image bytes and wraps the response
        into the standard EngineResult format.
        """
        import httpx

        if not self._upstream_url:
            raise RuntimeError(
                "ProxyEngine: upstream URL not configured. "
                "Set FF_PROXY_UPSTREAM_URL or pass upstream_url to __init__."
            )

        input_hash = _hash_bytes(image_bytes)

        headers: dict[str, str] = {}
        if self._upstream_key:
            headers["Authorization"] = f"Bearer {self._upstream_key}"

        try:
            response = httpx.post(
                self._upstream_url,
                content=image_bytes,
                headers=headers,
                timeout=30.0,
            )
            upstream_data = response.json() if response.status_code == 200 else {}
            upstream_status = response.status_code
        except httpx.HTTPError as exc:
            logger.warning("ProxyEngine upstream request failed: %s", exc)
            upstream_data = {}
            upstream_status = 502

        # Parse upstream matches into EngineMatch objects
        raw_matches = upstream_data.get("matches", [])
        matches = [
            EngineMatch(
                label=m.get("label", m.get("id", -1)),
                confidence=float(m.get("confidence", m.get("similarity", 0.0))),
            )
            for m in raw_matches[:top_k]
        ]
        prediction = matches[0].label if matches else -1

        event = _make_forensic_event(
            input_hash=input_hash,
            payload={
                "engine": "proxy",
                "input_hash": input_hash,
                "upstream_url": self._upstream_url,
                "upstream_status": upstream_status,
                "prediction": prediction,
                "top_k": min(top_k, len(matches)),
                "matches": [{"label": m.label, "confidence": m.confidence} for m in matches],
            },
            actor=actor,
            previous_hash=previous_hash,
            sequence_number=sequence_number,
        )

        provenance = _make_provenance_node(
            event=event,
            input_hash=input_hash,
            prediction=prediction,
            top_k=min(top_k, len(matches)),
            confidence=matches[0].confidence if matches else 0.0,
        )

        return EngineResult(
            matches=matches,
            input_hash=input_hash,
            engine_name="proxy",
            model_info=self.get_model_info(),
            inference_event=event,
            provenance_node=provenance,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "engine": "proxy",
            "ready": bool(self._upstream_url),
            "upstream_url": self._upstream_url,
        }


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def get_engine(engine_name: str | None = None, **kwargs: Any) -> RecognitionEngine:
    """Return the configured recognition engine.

    Parameters
    ----------
    engine_name : str | None
        One of ``"fallback"``, ``"onnx"``, ``"proxy"``.
        If *None*, reads from ``FF_RECOGNITION_ENGINE`` setting
        (default ``"fallback"``).
    **kwargs :
        Forwarded to the engine constructor.

    Returns
    -------
    RecognitionEngine
        A ready-to-use engine instance.

    Raises
    ------
    ValueError
        If the engine name is not recognised.
    """
    if engine_name is None:
        from friendlyface.config import settings

        engine_name = settings.recognition_engine

    engine_name = engine_name.lower().strip()

    # "auto" and "deep" map to fallback for this factory
    if engine_name in ("fallback", "auto", "deep", "mediapipe"):
        return FallbackEngine(**kwargs)
    if engine_name == "onnx":
        if "model_path" not in kwargs:
            from friendlyface.config import settings

            # Use explicit path if set, otherwise default to models/arcface_r100.onnx
            if settings.onnx_model_path:
                kwargs["model_path"] = settings.onnx_model_path
            else:
                default_model = Path(settings.model_dir) / "arcface_r100.onnx"
                if default_model.exists():
                    kwargs["model_path"] = str(default_model)
        return ONNXEngine(**kwargs)
    if engine_name == "proxy":
        return ProxyEngine(**kwargs)

    raise ValueError(
        f"Unknown recognition engine '{engine_name}'. "
        f"Valid options: fallback, onnx, proxy"
    )
