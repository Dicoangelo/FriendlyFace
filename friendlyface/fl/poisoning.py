"""Norm-based data poisoning detection for federated learning.

Detects anomalous client updates by computing the L2 norm of
each client's weight delta (update - global weights) and flagging
updates that exceed a configurable threshold above the median.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from friendlyface.core.models import (
    EventType,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
)


@dataclass
class ClientPoisoningResult:
    """Poisoning detection result for a single client."""

    client_id: int
    update_norm: float
    flagged: bool
    threshold_used: float


@dataclass
class PoisoningDetectionResult:
    """Aggregate poisoning detection result for an FL round."""

    round_number: int
    n_clients: int
    median_norm: float
    threshold_multiplier: float
    effective_threshold: float
    client_results: list[ClientPoisoningResult] = field(default_factory=list)
    flagged_client_ids: list[int] = field(default_factory=list)
    alert_events: list[ForensicEvent] = field(default_factory=list)
    provenance_node: ProvenanceNode | None = None

    @property
    def has_poisoning(self) -> bool:
        return len(self.flagged_client_ids) > 0


def _compute_update_norm(
    client_weights: list[np.ndarray],
    global_weights: list[np.ndarray],
) -> float:
    """Compute L2 norm of the weight delta (client - global)."""
    delta_sq_sum = 0.0
    for cw, gw in zip(client_weights, global_weights):
        delta = cw - gw
        delta_sq_sum += float(np.sum(delta**2))
    return float(np.sqrt(delta_sq_sum))


def detect_poisoning(
    client_weights: list[list[np.ndarray]],
    global_weights: list[np.ndarray],
    client_ids: list[int],
    round_number: int,
    *,
    threshold_multiplier: float = 3.0,
    actor: str = "poisoning_detector",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    round_provenance_id: object | None = None,
) -> PoisoningDetectionResult:
    """Detect potentially poisoned client updates via norm-based anomaly detection.

    Parameters
    ----------
    client_weights:
        List of weight lists, one per client (post-local-training).
    global_weights:
        Global model weights from before local training.
    client_ids:
        Integer ID for each client (same order as client_weights).
    round_number:
        FL round number (for event metadata).
    threshold_multiplier:
        A client is flagged if its update norm > median_norm * threshold_multiplier.
        Default: 3.0.
    actor:
        Actor identity for forensic events.
    previous_hash:
        Previous hash in the event chain.
    sequence_number:
        Starting sequence number for alert events.
    round_provenance_id:
        Optional provenance ID of the FL round to link alerts to.

    Returns
    -------
    PoisoningDetectionResult with per-client norms, flags, and alert events.
    """
    if len(client_weights) != len(client_ids):
        raise ValueError(
            f"client_weights length ({len(client_weights)}) != "
            f"client_ids length ({len(client_ids)})"
        )
    if len(client_weights) == 0:
        raise ValueError("No client updates to analyze")

    # Compute L2 norms of weight deltas
    norms = [
        _compute_update_norm(cw, global_weights) for cw in client_weights
    ]

    median_norm = float(np.median(norms))
    effective_threshold = median_norm * threshold_multiplier

    client_results: list[ClientPoisoningResult] = []
    flagged_ids: list[int] = []
    alert_events: list[ForensicEvent] = []
    current_hash = previous_hash
    current_seq = sequence_number

    for cid, norm in zip(client_ids, norms):
        flagged = norm > effective_threshold
        client_results.append(
            ClientPoisoningResult(
                client_id=cid,
                update_norm=norm,
                flagged=flagged,
                threshold_used=effective_threshold,
            )
        )

        if flagged:
            flagged_ids.append(cid)
            # Create a SECURITY_ALERT forensic event for each flagged client
            alert = ForensicEvent(
                event_type=EventType.SECURITY_ALERT,
                actor=actor,
                payload={
                    "alert_type": "data_poisoning",
                    "round": round_number,
                    "client_id": cid,
                    "update_norm": norm,
                    "median_norm": median_norm,
                    "threshold_multiplier": threshold_multiplier,
                    "effective_threshold": effective_threshold,
                },
                previous_hash=current_hash,
                sequence_number=current_seq,
            ).seal()
            current_hash = alert.event_hash
            current_seq += 1
            alert_events.append(alert)

    # Build provenance node for this detection pass
    parent_ids = []
    relations = []
    if round_provenance_id is not None:
        parent_ids = [round_provenance_id]
        relations = [ProvenanceRelation.DERIVED_FROM]

    provenance_node = ProvenanceNode(
        entity_type="poisoning_detection",
        entity_id=f"poisoning_round_{round_number}",
        metadata={
            "round": round_number,
            "n_clients": len(client_ids),
            "median_norm": median_norm,
            "threshold_multiplier": threshold_multiplier,
            "effective_threshold": effective_threshold,
            "flagged_client_ids": flagged_ids,
            "n_flagged": len(flagged_ids),
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    return PoisoningDetectionResult(
        round_number=round_number,
        n_clients=len(client_ids),
        median_norm=median_norm,
        threshold_multiplier=threshold_multiplier,
        effective_threshold=effective_threshold,
        client_results=client_results,
        flagged_client_ids=flagged_ids,
        alert_events=alert_events,
        provenance_node=provenance_node,
    )
