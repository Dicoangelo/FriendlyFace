"""SHAP explainability for face recognition predictions.

Generates SHAP (SHapley Additive exPlanations) values for face recognition
predictions, computing per-feature-dimension importance with a lightweight
KernelSHAP approximation.  Each explanation is logged as a ForensicEvent
with full provenance tracking.

The implementation uses a paired-sampling KernelSHAP approximation
(Lundberg & Lee, NeurIPS 2017) that requires only numpy -- no external
shap library.

References:
  - Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"
    (NeurIPS 2017)
  - Mohammed ICDF2C 2024 forensic provenance chain
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import numpy as np

from friendlyface.core.models import (
    EventType,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
    canonical_json,
    sha256_hex,
)


@dataclass
class ShapExplanation:
    """Complete SHAP explanation artifact for a face recognition prediction.

    Contains per-feature SHAP values, a feature importance ranking,
    the base (expected) value, and forensic metadata.
    """

    inference_event_id: UUID
    predicted_label: int
    original_confidence: float
    shap_values: np.ndarray  # 1-D array of length n_features
    feature_importance_ranking: list[int]  # feature indices sorted by |SHAP|
    base_value: float  # expected prediction over the background
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
            "shap_values_hash": sha256_hex(self.shap_values.tobytes().hex()),
            "feature_importance_ranking": self.feature_importance_ranking,
            "base_value": self.base_value,
            "num_samples": self.num_samples,
        }
        return sha256_hex(canonical_json(hashable))


def _kernel_shap(
    predict_fn: Any,
    instance: np.ndarray,
    background: np.ndarray,
    num_samples: int = 128,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Lightweight KernelSHAP approximation.

    Uses paired-sampling to estimate per-feature Shapley values for a
    single instance against a background dataset.

    Parameters
    ----------
    predict_fn:
        Callable mapping (N, D) feature arrays to (N,) prediction scores
        for the target class.
    instance:
        1-D array of shape (D,) -- the feature vector to explain.
    background:
        2-D array of shape (M, D) -- background/reference samples.
    num_samples:
        Number of coalition samples for the approximation.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    (shap_values, base_value) where shap_values is shape (D,) and
    base_value is the mean prediction over the background.
    """
    rng = np.random.default_rng(random_state)
    d = instance.shape[0]
    m = background.shape[0]

    # Base value: expected prediction over background
    bg_preds = predict_fn(background)
    base_value = float(np.mean(bg_preds))

    # Paired-sampling KernelSHAP
    shap_accum = np.zeros(d, dtype=np.float64)
    weight_accum = np.zeros(d, dtype=np.float64)

    for _ in range(num_samples):
        # Random coalition size (avoid empty and full)
        s = rng.integers(1, d)
        # Random feature subset of size s
        coalition = rng.choice(d, size=s, replace=False)
        mask = np.zeros(d, dtype=bool)
        mask[coalition] = True

        # Pick a random background sample
        bg_idx = rng.integers(0, m)
        bg_sample = background[bg_idx]

        # Construct synthetic instances
        x_with = bg_sample.copy()
        x_with[mask] = instance[mask]

        x_without = bg_sample.copy()
        x_without[~mask] = instance[~mask]

        # Predict
        preds = predict_fn(np.vstack([x_with[np.newaxis], x_without[np.newaxis]]))
        delta = preds[0] - preds[1]

        # Assign equal credit to coalition members
        contribution = delta / max(s, 1)
        shap_accum[mask] += contribution
        weight_accum[mask] += 1.0

        # Complement features get negative credit
        complement_size = d - s
        if complement_size > 0:
            neg_contribution = -delta / max(complement_size, 1)
            shap_accum[~mask] += neg_contribution
            weight_accum[~mask] += 1.0

    # Average contributions
    nonzero = weight_accum > 0
    shap_values = np.zeros(d, dtype=np.float64)
    shap_values[nonzero] = shap_accum[nonzero] / weight_accum[nonzero]

    return shap_values, base_value


def generate_shap_explanation(
    feature_vector: np.ndarray,
    predict_fn: Any,
    background: np.ndarray,
    *,
    inference_event_id: UUID,
    predicted_label: int,
    original_confidence: float,
    num_samples: int = 128,
    random_state: int = 42,
    actor: str = "shap_explainer",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    inference_provenance_id: UUID | None = None,
) -> ShapExplanation:
    """Generate a SHAP explanation for a face recognition prediction.

    Parameters
    ----------
    feature_vector:
        1-D array of shape (D,) -- PCA-projected feature vector for the
        instance being explained.
    predict_fn:
        Callable that takes a 2-D array (N, D) of feature vectors and
        returns a 1-D array (N,) of prediction scores (e.g. probability
        for the predicted class).
    background:
        2-D array of shape (M, D) -- background/reference feature vectors
        (e.g. from the training set projected through PCA).
    inference_event_id:
        UUID of the inference event being explained.
    predicted_label:
        The predicted class label from the inference.
    original_confidence:
        Confidence score of the original prediction.
    num_samples:
        Number of coalition samples for KernelSHAP approximation.
    random_state:
        Random seed for reproducibility.
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
    ShapExplanation with SHAP values, feature importance ranking, base value,
    forensic event, and provenance node.
    """
    instance = np.asarray(feature_vector, dtype=np.float64).ravel()
    bg = np.asarray(background, dtype=np.float64)
    if bg.ndim == 1:
        bg = bg.reshape(1, -1)

    # Compute SHAP values via lightweight KernelSHAP
    shap_values, base_value = _kernel_shap(
        predict_fn,
        instance,
        bg,
        num_samples=num_samples,
        random_state=random_state,
    )

    # Feature importance ranking: sorted by descending |SHAP value|
    feature_importance_ranking = list(np.argsort(-np.abs(shap_values)))

    # Create the explanation object
    explanation = ShapExplanation(
        inference_event_id=inference_event_id,
        predicted_label=predicted_label,
        original_confidence=original_confidence,
        shap_values=shap_values,
        feature_importance_ranking=feature_importance_ranking,
        base_value=base_value,
        num_samples=num_samples,
    )
    explanation.artifact_hash = explanation.compute_artifact_hash()

    # Log forensic explanation event
    top_k = min(10, len(feature_importance_ranking))
    forensic_event = ForensicEvent(
        event_type=EventType.EXPLANATION_GENERATED,
        actor=actor,
        payload={
            "explanation_type": "SHAP",
            "method": "shap",
            "inference_event_id": str(inference_event_id),
            "predicted_label": predicted_label,
            "original_confidence": original_confidence,
            "num_features": len(shap_values),
            "num_samples": num_samples,
            "artifact_hash": explanation.artifact_hash,
            "base_value": base_value,
            "top_features": [
                {
                    "feature_index": int(feature_importance_ranking[i]),
                    "shap_value": float(shap_values[feature_importance_ranking[i]]),
                }
                for i in range(top_k)
            ],
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
            "explanation_type": "SHAP",
            "method": "shap",
            "inference_event_id": str(inference_event_id),
            "artifact_hash": explanation.artifact_hash,
            "predicted_label": predicted_label,
            "num_features": len(shap_values),
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    explanation.forensic_event = forensic_event
    explanation.provenance_node = provenance_node

    return explanation
