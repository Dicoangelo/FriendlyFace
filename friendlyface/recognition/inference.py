"""Face inference pipeline with forensic logging.

Loads trained PCA and SVM models, runs prediction on a single face image,
and logs each inference as a ForensicEvent with full provenance.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from friendlyface.core.models import EventType, ForensicEvent, ProvenanceNode, ProvenanceRelation
from friendlyface.recognition.pca import IMAGE_SIZE


@dataclass
class Match:
    """A single prediction match with confidence score."""

    label: int
    confidence: float


@dataclass
class InferenceResult:
    """Container for inference outputs with forensic metadata."""

    matches: list[Match]
    input_hash: str
    inference_event: ForensicEvent
    provenance_node: ProvenanceNode


def _hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def load_pca_model(model_path: Path) -> dict[str, Any]:
    """Load a serialized PCA model blob from disk.

    Returns the dict containing 'pca', 'n_components', etc.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"PCA model not found: {model_path}")
    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    if "pca" not in blob:
        raise ValueError("Invalid PCA model file: missing 'pca' key")
    return blob


def load_svm_model(model_path: Path) -> dict[str, Any]:
    """Load a serialized SVM model blob from disk.

    Returns the dict containing 'svm', 'hyperparameters', etc.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found: {model_path}")
    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    if "svm" not in blob:
        raise ValueError("Invalid SVM model file: missing 'svm' key")
    return blob


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to a flattened feature vector.

    The image is converted to grayscale, resized to 112x112,
    and flattened into a 1-D vector of 12544 pixels.

    Returns an (1, 12544) float64 array.
    """
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    if img.size != IMAGE_SIZE:
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float64).ravel()
    return arr.reshape(1, -1)


def run_inference(
    image_bytes: bytes,
    pca_model_path: Path,
    svm_model_path: Path,
    *,
    top_k: int = 5,
    actor: str = "inference_engine",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    model_provenance_id: Any | None = None,
) -> InferenceResult:
    """Run face recognition inference on a single image.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the face image to recognize.
    pca_model_path:
        Path to the serialized PCA model (.pkl).
    svm_model_path:
        Path to the serialized SVM model (.pkl).
    top_k:
        Number of top matches to return.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain (for chaining).
    sequence_number:
        Sequence position in the event chain.
    model_provenance_id:
        Optional UUID of the SVM model provenance node.

    Returns
    -------
    InferenceResult with matches, input hash, forensic event,
    and provenance node.
    """
    # Hash the input image for forensic traceability
    input_hash = _hash_bytes(image_bytes)

    # Load models
    pca_blob = load_pca_model(pca_model_path)
    svm_blob = load_svm_model(svm_model_path)

    pca = pca_blob["pca"]
    svm = svm_blob["svm"]

    # Preprocess image and run PCA transform
    features = _preprocess_image(image_bytes)
    reduced = pca.transform(features)

    # Get SVM decision scores for ranking
    prediction = int(svm.predict(reduced)[0])
    classes = svm.classes_

    # Build top-K matches from decision function scores
    if len(classes) == 2:
        # Binary: decision_function returns single value
        raw_scores = svm.decision_function(reduced).ravel()
        # Convert to pseudo-probabilities via sigmoid
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
        match_scores = {classes[1]: float(scores[0]), classes[0]: float(1.0 - scores[0])}
    else:
        # Multiclass: decision_function with default 'ovr' shape returns (1, n_classes)
        raw_scores = svm.decision_function(reduced).ravel()
        # Normalize via softmax to get pseudo-probabilities
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / exp_scores.sum()
        match_scores = {cls: float(prob) for cls, prob in zip(classes, probs)}

    # Sort by confidence descending, take top-K
    sorted_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
    actual_k = min(top_k, len(sorted_matches))
    matches = [Match(label=int(lbl), confidence=conf) for lbl, conf in sorted_matches[:actual_k]]

    # Log forensic inference event
    inference_event = ForensicEvent(
        event_type=EventType.INFERENCE_RESULT,
        actor=actor,
        payload={
            "input_hash": input_hash,
            "pca_model_path": str(pca_model_path),
            "svm_model_path": str(svm_model_path),
            "prediction": prediction,
            "top_k": actual_k,
            "matches": [{"label": m.label, "confidence": m.confidence} for m in matches],
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # Create provenance node for this inference
    parent_ids = []
    relations = []
    if model_provenance_id is not None:
        parent_ids = [model_provenance_id]
        relations = [ProvenanceRelation.GENERATED_BY]

    provenance_node = ProvenanceNode(
        entity_type="inference",
        entity_id=str(inference_event.id),
        metadata={
            "input_hash": input_hash,
            "prediction": prediction,
            "top_k": actual_k,
            "confidence": matches[0].confidence if matches else 0.0,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    return InferenceResult(
        matches=matches,
        input_hash=input_hash,
        inference_event=inference_event,
        provenance_node=provenance_node,
    )
