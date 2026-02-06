"""LIME explainability for face recognition predictions.

Generates LIME (Local Interpretable Model-agnostic Explanations) for
face recognition predictions, identifying top contributing facial
regions (superpixels) and logging each explanation as a ForensicEvent
with full provenance tracking.

References:
  - Ribeiro et al., "Why Should I Trust You?" (KDD 2016)
  - Mohammed ICDF2C 2024 forensic provenance chain
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any
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


@dataclass
class SuperpixelContribution:
    """A single superpixel region and its importance to the prediction."""

    superpixel_id: int
    importance: float
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)


@dataclass
class LimeExplanation:
    """Complete LIME explanation artifact for a face recognition prediction.

    Contains the feature importance map, top-K contributing regions,
    confidence decomposition, and forensic metadata.
    """

    inference_event_id: UUID
    predicted_label: int
    original_confidence: float
    top_regions: list[SuperpixelContribution]
    feature_importance_map: np.ndarray  # 2-D array same shape as input image
    confidence_decomposition: dict[str, float]  # base + per-region contributions
    num_superpixels: int
    num_samples: int
    artifact_hash: str = ""
    forensic_event: ForensicEvent | None = None
    provenance_node: ProvenanceNode | None = None

    def compute_artifact_hash(self) -> str:
        """Compute SHA-256 over the explanation artifact content."""
        hashable = {
            "inference_event_id": str(self.inference_event_id),
            "predicted_label": self.predicted_label,
            "original_confidence": self.original_confidence,
            "top_regions": [
                {
                    "superpixel_id": r.superpixel_id,
                    "importance": r.importance,
                    "bbox": list(r.bbox),
                }
                for r in self.top_regions
            ],
            "feature_importance_map_hash": sha256_hex(
                self.feature_importance_map.tobytes().hex()
            ),
            "confidence_decomposition": self.confidence_decomposition,
            "num_superpixels": self.num_superpixels,
            "num_samples": self.num_samples,
        }
        return sha256_hex(canonical_json(hashable))


def _image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load and prepare a grayscale image from raw bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    if img.size != IMAGE_SIZE:
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return img


def _segment_image(image_array: np.ndarray, num_superpixels: int = 50) -> np.ndarray:
    """Segment image into superpixel regions using a grid-based approach.

    Returns an integer mask array of shape (H, W) where each pixel
    is assigned a superpixel ID from 0 to num_superpixels-1.

    Uses a simple grid segmentation (no external dependency on skimage)
    for reliable, deterministic superpixel generation.
    """
    h, w = image_array.shape[:2]
    # Compute grid dimensions
    grid_size = max(1, int(np.ceil(np.sqrt(num_superpixels))))
    row_step = max(1, h // grid_size)
    col_step = max(1, w // grid_size)

    segments = np.zeros((h, w), dtype=np.int32)
    sp_id = 0
    for r in range(0, h, row_step):
        for c in range(0, w, col_step):
            r_end = min(r + row_step, h)
            c_end = min(c + col_step, w)
            segments[r:r_end, c:c_end] = sp_id
            sp_id += 1

    return segments


def _get_superpixel_bbox(
    segments: np.ndarray, sp_id: int
) -> tuple[int, int, int, int]:
    """Get bounding box (min_row, min_col, max_row, max_col) for a superpixel."""
    rows, cols = np.where(segments == sp_id)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def generate_lime_explanation(
    image_bytes: bytes,
    predict_fn: Any,
    *,
    inference_event_id: UUID,
    predicted_label: int,
    original_confidence: float,
    num_superpixels: int = 50,
    num_samples: int = 100,
    top_k: int = 5,
    actor: str = "lime_explainer",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    inference_provenance_id: UUID | None = None,
) -> LimeExplanation:
    """Generate a LIME explanation for a face recognition prediction.

    Parameters
    ----------
    image_bytes:
        Raw bytes of the face image that was recognized.
    predict_fn:
        Callable that takes a batch of images as a 4-D numpy array
        (N, H, W, 1) and returns prediction probabilities as (N, C).
    inference_event_id:
        UUID of the inference event being explained.
    predicted_label:
        The predicted class label from the inference.
    original_confidence:
        Confidence score of the original prediction.
    num_superpixels:
        Number of superpixel regions to segment the image into.
    num_samples:
        Number of perturbed samples for LIME fitting.
    top_k:
        Number of top contributing regions to include.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain.
    sequence_number:
        Sequence position in the event chain.
    inference_provenance_id:
        Optional UUID of the inference provenance node (parent).

    Returns
    -------
    LimeExplanation with feature importance map, top regions,
    confidence decomposition, forensic event, and provenance node.
    """
    from lime.lime_image import LimeImageExplainer

    # Prepare image
    img = _image_from_bytes(image_bytes)
    image_array = np.asarray(img, dtype=np.float64)

    # LIME expects a 3-D array (H, W, C) even for grayscale
    image_3d = image_array[:, :, np.newaxis]

    # Create LIME explainer and generate explanation
    explainer = LimeImageExplainer(random_state=42)

    def _wrapped_predict(images: np.ndarray) -> np.ndarray:
        """Wrap predict_fn to handle LIME's image format.

        LIME passes images as (N, H, W, C) float arrays.
        We forward them to the user's predict_fn as-is.
        """
        return predict_fn(images)

    lime_exp = explainer.explain_instance(
        image_3d,
        _wrapped_predict,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda img: _segment_image(
            img[:, :, 0] if img.ndim == 3 else img,
            num_superpixels,
        ),
    )

    # Extract explanation for the predicted label
    label_to_explain = predicted_label
    if label_to_explain not in lime_exp.local_exp:
        # Fall back to top label from LIME
        label_to_explain = lime_exp.top_labels[0]

    local_exp = lime_exp.local_exp[label_to_explain]
    segments = _segment_image(image_array, num_superpixels)

    # Build feature importance map
    importance_map = np.zeros(image_array.shape, dtype=np.float64)
    region_importances: dict[int, float] = {}
    for sp_id, weight in local_exp:
        mask = segments == sp_id
        importance_map[mask] = weight
        region_importances[sp_id] = weight

    # Sort by absolute importance, take top-K
    sorted_regions = sorted(
        region_importances.items(), key=lambda x: abs(x[1]), reverse=True
    )
    actual_k = min(top_k, len(sorted_regions))

    top_regions = [
        SuperpixelContribution(
            superpixel_id=sp_id,
            importance=weight,
            bbox=_get_superpixel_bbox(segments, sp_id),
        )
        for sp_id, weight in sorted_regions[:actual_k]
    ]

    # Build confidence decomposition
    intercept = float(lime_exp.intercept[label_to_explain])
    confidence_decomposition: dict[str, float] = {"base_score": intercept}
    for region in top_regions:
        confidence_decomposition[f"region_{region.superpixel_id}"] = region.importance

    # Create the explanation object
    explanation = LimeExplanation(
        inference_event_id=inference_event_id,
        predicted_label=predicted_label,
        original_confidence=original_confidence,
        top_regions=top_regions,
        feature_importance_map=importance_map,
        confidence_decomposition=confidence_decomposition,
        num_superpixels=num_superpixels,
        num_samples=num_samples,
    )
    explanation.artifact_hash = explanation.compute_artifact_hash()

    # Log forensic explanation event
    forensic_event = ForensicEvent(
        event_type=EventType.EXPLANATION_GENERATED,
        actor=actor,
        payload={
            "explanation_type": "LIME",
            "inference_event_id": str(inference_event_id),
            "predicted_label": predicted_label,
            "original_confidence": original_confidence,
            "num_superpixels": num_superpixels,
            "num_samples": num_samples,
            "top_k": actual_k,
            "artifact_hash": explanation.artifact_hash,
            "top_regions": [
                {
                    "superpixel_id": r.superpixel_id,
                    "importance": r.importance,
                    "bbox": list(r.bbox),
                }
                for r in top_regions
            ],
            "confidence_decomposition": confidence_decomposition,
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # Create provenance node for this explanation
    parent_ids: list[UUID] = []
    relations: list[ProvenanceRelation] = []
    if inference_provenance_id is not None:
        parent_ids = [inference_provenance_id]
        relations = [ProvenanceRelation.DERIVED_FROM]

    provenance_node = ProvenanceNode(
        entity_type="explanation",
        entity_id=str(forensic_event.id),
        metadata={
            "explanation_type": "LIME",
            "inference_event_id": str(inference_event_id),
            "artifact_hash": explanation.artifact_hash,
            "predicted_label": predicted_label,
            "top_k": actual_k,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    explanation.forensic_event = forensic_event
    explanation.provenance_node = provenance_node

    return explanation
