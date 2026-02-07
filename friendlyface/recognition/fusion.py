"""Score-level multi-modal fusion (face + voice) with forensic logging.

Merges confidence scores from face and voice recognition pipelines using
weighted-sum fusion, producing a single ranked match list with full
forensic provenance linking both modality inference nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from friendlyface.core.models import (
    EventType,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
)


@dataclass
class FusedMatch:
    """A single fused match combining face and voice confidence scores."""

    label: str
    fused_confidence: float
    face_confidence: float | None
    voice_confidence: float | None


@dataclass
class FusionResult:
    """Container for multi-modal fusion outputs with forensic metadata."""

    fused_matches: list[FusedMatch]
    face_weight: float
    voice_weight: float
    fusion_method: str
    forensic_event: ForensicEvent
    provenance_node: ProvenanceNode


def fuse_scores(
    face_matches: list[dict[str, Any]],
    voice_matches: list[dict[str, Any]],
    face_weight: float = 0.6,
    voice_weight: float = 0.4,
    *,
    actor: str = "fusion_engine",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    face_provenance_id: Any | None = None,
    voice_provenance_id: Any | None = None,
) -> FusionResult:
    """Fuse face and voice match scores using weighted-sum fusion.

    Parameters
    ----------
    face_matches:
        List of ``{"label": str, "confidence": float}`` from face inference.
    voice_matches:
        List of ``{"label": str, "confidence": float}`` from voice inference.
    face_weight:
        Weight for face confidence scores.
    voice_weight:
        Weight for voice confidence scores.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain.
    sequence_number:
        Sequence position in the event chain.
    face_provenance_id:
        Optional UUID of the face inference provenance node.
    voice_provenance_id:
        Optional UUID of the voice inference provenance node.

    Returns
    -------
    FusionResult with fused matches, weights, forensic event,
    and provenance node linking both modality parents.

    Raises
    ------
    ValueError
        If weights do not sum to 1.0 (within tolerance).
    """
    # Validate weights
    if abs(face_weight + voice_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got face={face_weight} + voice={voice_weight}"
            f" = {face_weight + voice_weight}"
        )

    # Index face scores by label
    face_scores: dict[str, float] = {m["label"]: m["confidence"] for m in face_matches}
    voice_scores: dict[str, float] = {m["label"]: m["confidence"] for m in voice_matches}

    # Merge all labels from both modalities
    all_labels = set(face_scores.keys()) | set(voice_scores.keys())

    fused: list[FusedMatch] = []
    for label in all_labels:
        face_conf = face_scores.get(label)
        voice_conf = voice_scores.get(label)

        # Compute fused score using available modalities
        score = 0.0
        if face_conf is not None:
            score += face_weight * face_conf
        if voice_conf is not None:
            score += voice_weight * voice_conf

        fused.append(
            FusedMatch(
                label=label,
                fused_confidence=score,
                face_confidence=face_conf,
                voice_confidence=voice_conf,
            )
        )

    # Sort by fused confidence descending
    fused.sort(key=lambda m: m.fused_confidence, reverse=True)

    # Log forensic event
    forensic_event = ForensicEvent(
        event_type=EventType.INFERENCE_RESULT,
        actor=actor,
        payload={
            "fusion_method": "weighted_sum",
            "face_weight": face_weight,
            "voice_weight": voice_weight,
            "n_face_matches": len(face_matches),
            "n_voice_matches": len(voice_matches),
            "n_fused_matches": len(fused),
            "fused_matches": [
                {
                    "label": m.label,
                    "fused_confidence": m.fused_confidence,
                    "face_confidence": m.face_confidence,
                    "voice_confidence": m.voice_confidence,
                }
                for m in fused
            ],
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # Create provenance node linking both modality inference nodes
    parent_ids = []
    relations = []
    if face_provenance_id is not None:
        parent_ids.append(face_provenance_id)
        relations.append(ProvenanceRelation.DERIVED_FROM)
    if voice_provenance_id is not None:
        parent_ids.append(voice_provenance_id)
        relations.append(ProvenanceRelation.DERIVED_FROM)

    provenance_node = ProvenanceNode(
        entity_type="fusion",
        entity_id=str(forensic_event.id),
        metadata={
            "fusion_method": "weighted_sum",
            "face_weight": face_weight,
            "voice_weight": voice_weight,
            "n_fused_matches": len(fused),
            "top_confidence": fused[0].fused_confidence if fused else 0.0,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    return FusionResult(
        fused_matches=fused,
        face_weight=face_weight,
        voice_weight=voice_weight,
        fusion_method="weighted_sum",
        forensic_event=forensic_event,
        provenance_node=provenance_node,
    )
