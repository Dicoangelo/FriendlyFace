"""SDD saliency explainability for face recognition predictions.

Generates SDD (Spatial-Directional Decomposition) pixel-level saliency maps
showing which facial regions contribute most to a recognition decision.
Each explanation is logged as a ForensicEvent with full provenance tracking.

The implementation uses finite-difference gradient approximation on the
flattened image vector, decomposes gradients into canonical facial regions,
and scores each region's contribution to the final prediction.  Requires
only numpy -- no external saliency library.

References:
  - Li et al., "SDD Saliency" (arXiv:2505.03837)
  - Mohammed ICDF2C 2024 forensic provenance chain
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable
from uuid import UUID

import numpy as np
from PIL import Image

from friendlyface.core.models import (
    EventType,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
    canonical_json,
    sha256_hex,
)
from friendlyface.recognition.pca import IMAGE_SIZE

# ---------------------------------------------------------------------------
# Canonical facial region definitions on 112x112 grid
# ---------------------------------------------------------------------------

_CANONICAL_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "forehead": (0, 0, 25, 111),
    "left_eye": (26, 0, 50, 55),
    "right_eye": (26, 56, 50, 111),
    "nose": (51, 28, 75, 83),
    "mouth": (76, 20, 95, 91),
    "left_jaw": (76, 0, 111, 35),
    "right_jaw": (76, 76, 111, 111),
}

_IMG_H, _IMG_W = IMAGE_SIZE  # (112, 112)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FacialRegion:
    """A single facial region and its contribution to the prediction."""

    name: str
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    importance: float  # Contribution score [0, 1]
    pixel_count: int


@dataclass
class SDDExplanation:
    """Complete SDD saliency explanation artifact for a face recognition prediction.

    Contains the pixel-level saliency map, per-region importance scores,
    the dominant contributing region, and forensic metadata.
    """

    inference_event_id: str
    predicted_label: str
    original_confidence: float
    saliency_map: list[list[float]]  # 2-D pixel-level saliency (112x112)
    regions: list[FacialRegion]  # Sorted by importance descending
    dominant_region: str  # Name of highest-importance region
    artifact_hash: str = ""
    forensic_event: ForensicEvent | None = None
    provenance_node: ProvenanceNode | None = None

    def compute_artifact_hash(self) -> str:
        """Compute SHA-256 over the explanation artifact content."""
        hashable = {
            "inference_event_id": self.inference_event_id,
            "predicted_label": self.predicted_label,
            "original_confidence": self.original_confidence,
            "saliency_map_hash": sha256_hex(canonical_json({"saliency": self.saliency_map})),
            "regions": [
                {
                    "name": r.name,
                    "bbox": list(r.bbox),
                    "importance": r.importance,
                    "pixel_count": r.pixel_count,
                }
                for r in self.regions
            ],
            "dominant_region": self.dominant_region,
        }
        return sha256_hex(canonical_json(hashable))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image bytes into a (112, 112) float64 array in [0, 1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    if img.size != IMAGE_SIZE:
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def _finite_difference_gradients(
    flat_image: np.ndarray,
    predict_fn: Callable[[np.ndarray], tuple[str, float]],
    eps: float = 1e-3,
) -> np.ndarray:
    """Approximate per-pixel gradients via central finite differences.

    For each pixel *i*, computes:
        grad[i] = (conf(image + eps*e_i) - conf(image - eps*e_i)) / (2 * eps)

    where conf() is the confidence returned by ``predict_fn``.

    Parameters
    ----------
    flat_image:
        1-D array of length H*W (flattened image).
    predict_fn:
        Takes a flattened vector, returns (label, confidence).
    eps:
        Perturbation magnitude.

    Returns
    -------
    1-D gradient array of the same length as *flat_image*.
    """
    n = flat_image.shape[0]
    grads = np.zeros(n, dtype=np.float64)

    for i in range(n):
        perturbed_plus = flat_image.copy()
        perturbed_plus[i] += eps

        perturbed_minus = flat_image.copy()
        perturbed_minus[i] -= eps

        _, conf_plus = predict_fn(perturbed_plus)
        _, conf_minus = predict_fn(perturbed_minus)

        grads[i] = (conf_plus - conf_minus) / (2.0 * eps)

    return grads


def _score_regions(
    gradient_map: np.ndarray,
) -> list[FacialRegion]:
    """Score canonical facial regions by mean absolute gradient.

    Parameters
    ----------
    gradient_map:
        2-D array of shape (112, 112) containing per-pixel gradients.

    Returns
    -------
    List of ``FacialRegion`` objects sorted by importance descending,
    with importance values normalized to [0, 1].
    """
    abs_grads = np.abs(gradient_map)
    raw_scores: list[tuple[str, float, tuple[int, int, int, int], int]] = []

    for name, (r_min, c_min, r_max, c_max) in _CANONICAL_REGIONS.items():
        region_slice = abs_grads[r_min : r_max + 1, c_min : c_max + 1]
        mean_score = float(np.mean(region_slice)) if region_slice.size > 0 else 0.0
        pixel_count = int(region_slice.size)
        raw_scores.append((name, mean_score, (r_min, c_min, r_max, c_max), pixel_count))

    # Normalize to [0, 1]
    max_score = max(s[1] for s in raw_scores) if raw_scores else 1.0
    if max_score == 0.0:
        max_score = 1.0  # avoid division by zero

    regions = [
        FacialRegion(
            name=name,
            bbox=bbox,
            importance=score / max_score,
            pixel_count=pixel_count,
        )
        for name, score, bbox, pixel_count in raw_scores
    ]

    # Sort by importance descending
    regions.sort(key=lambda r: r.importance, reverse=True)
    return regions


def _normalize_saliency(gradient_map: np.ndarray) -> list[list[float]]:
    """Normalize absolute gradient map to [0, 1] and convert to nested list."""
    abs_map = np.abs(gradient_map)
    max_val = float(np.max(abs_map))
    if max_val == 0.0:
        max_val = 1.0
    normalized = abs_map / max_val
    return normalized.tolist()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_sdd_explanation(
    image_bytes: bytes,
    predict_fn: Callable[[np.ndarray], tuple[str, float]],
    features: np.ndarray,
    *,
    actor: str = "sdd_explainer",
    inference_event_id: str = "",
    parent_provenance_id: str | None = None,
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
) -> SDDExplanation:
    """Generate an SDD saliency explanation for a face recognition prediction.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the face image that was recognized.
    predict_fn:
        Callable that takes a flattened image vector (1-D numpy array of
        length H*W) and returns ``(label, confidence)`` where *label* is
        a string and *confidence* is a float.
    features:
        PCA-reduced feature vector for this image (used for metadata only;
        the gradient computation operates on the raw pixel space).
    actor:
        Actor identity for forensic event logging.
    inference_event_id:
        String identifier of the inference event being explained.
    parent_provenance_id:
        Optional string ID of a parent provenance node to link to.
    previous_hash:
        Previous hash in the forensic event chain.
    sequence_number:
        Sequence position in the event chain.

    Returns
    -------
    SDDExplanation with saliency map, region scores, forensic event,
    and provenance node.
    """
    # 1. Load image and get baseline prediction
    image_2d = _image_from_bytes(image_bytes)
    flat_image = image_2d.ravel()
    predicted_label, original_confidence = predict_fn(flat_image)

    # 2. Compute per-pixel gradients via finite differences
    gradients = _finite_difference_gradients(flat_image, predict_fn)
    gradient_map = gradients.reshape(_IMG_H, _IMG_W)

    # 3. Build normalized saliency map (112x112 nested list)
    saliency_map = _normalize_saliency(gradient_map)

    # 4. Score canonical facial regions
    regions = _score_regions(gradient_map)
    dominant_region = regions[0].name if regions else "unknown"

    # 5. Build explanation object and compute artifact hash
    explanation = SDDExplanation(
        inference_event_id=inference_event_id,
        predicted_label=predicted_label,
        original_confidence=original_confidence,
        saliency_map=saliency_map,
        regions=regions,
        dominant_region=dominant_region,
    )
    explanation.artifact_hash = explanation.compute_artifact_hash()

    # 6. Log forensic explanation event
    forensic_event = ForensicEvent(
        event_type=EventType.EXPLANATION_GENERATED,
        actor=actor,
        payload={
            "explanation_type": "SDD",
            "method": "spatial_directional_decomposition",
            "inference_event_id": inference_event_id,
            "predicted_label": predicted_label,
            "original_confidence": original_confidence,
            "artifact_hash": explanation.artifact_hash,
            "dominant_region": dominant_region,
            "num_regions": len(regions),
            "feature_vector_length": int(features.shape[0])
            if features.ndim == 1
            else int(features.size),
            "region_scores": [
                {
                    "name": r.name,
                    "importance": r.importance,
                    "bbox": list(r.bbox),
                    "pixel_count": r.pixel_count,
                }
                for r in regions
            ],
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # 7. Create provenance node for this explanation
    parent_ids: list[UUID] = []
    relations: list[ProvenanceRelation] = []
    if parent_provenance_id is not None:
        parent_ids = [UUID(parent_provenance_id)]
        relations = [ProvenanceRelation.DERIVED_FROM]

    provenance_node = ProvenanceNode(
        entity_type="explanation",
        entity_id=str(forensic_event.id),
        metadata={
            "explanation_type": "SDD",
            "method": "spatial_directional_decomposition",
            "inference_event_id": inference_event_id,
            "artifact_hash": explanation.artifact_hash,
            "predicted_label": predicted_label,
            "dominant_region": dominant_region,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    explanation.forensic_event = forensic_event
    explanation.provenance_node = provenance_node

    return explanation
