"""Simulated federated learning engine with forensic event logging.

Implements FedAvg aggregation over configurable simulated clients,
with each FL round logged as a ForensicEvent(FL_ROUND) and provenance
nodes tracking client contributions to the aggregated global model.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from friendlyface.core.models import (
    EventType,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
)
from friendlyface.fl.poisoning import PoisoningDetectionResult, detect_poisoning


def _hash_weights(weights: list[np.ndarray]) -> str:
    """Compute a SHA-256 digest of serialized model weights."""
    return hashlib.sha256(pickle.dumps(weights)).hexdigest()


def _fedavg(
    client_weights: list[list[np.ndarray]],
    client_sizes: list[int],
) -> list[np.ndarray]:
    """Federated averaging: weighted mean of client model weights.

    Parameters
    ----------
    client_weights:
        List of weight lists, one per client. Each inner list contains
        numpy arrays (one per layer).
    client_sizes:
        Number of training samples per client, used for weighted averaging.

    Returns
    -------
    Aggregated weight list (same structure as a single client's weights).
    """
    total = sum(client_sizes)
    n_layers = len(client_weights[0])
    aggregated: list[np.ndarray] = []
    for layer_idx in range(n_layers):
        weighted_sum = sum(w[layer_idx] * (n / total) for w, n in zip(client_weights, client_sizes))
        aggregated.append(weighted_sum)
    return aggregated


@dataclass
class ClientUpdate:
    """Result from a single client's local training."""

    client_id: int
    weights: list[np.ndarray]
    n_samples: int
    local_loss: float


@dataclass
class FLRoundResult:
    """Result from a single FL round."""

    round_number: int
    client_updates: list[ClientUpdate]
    global_weights: list[np.ndarray]
    global_model_hash: str
    event: ForensicEvent
    provenance_node: ProvenanceNode
    poisoning_result: PoisoningDetectionResult | None = None


@dataclass
class FLSimulationResult:
    """Complete result of an FL simulation run."""

    n_rounds: int
    n_clients: int
    rounds: list[FLRoundResult] = field(default_factory=list)
    final_model_hash: str = ""


def _simulate_client_training(
    client_id: int,
    global_weights: list[np.ndarray],
    client_data_size: int,
    rng: np.random.Generator,
) -> ClientUpdate:
    """Simulate local training for a single client.

    Each client starts from the global weights and applies a small
    random perturbation to simulate local SGD updates.
    """
    local_weights = [w + rng.normal(0, 0.01, size=w.shape) for w in global_weights]
    local_loss = float(rng.uniform(0.1, 1.0))
    return ClientUpdate(
        client_id=client_id,
        weights=local_weights,
        n_samples=client_data_size,
        local_loss=local_loss,
    )


def run_fl_simulation(
    n_clients: int = 5,
    n_rounds: int = 3,
    weight_shapes: list[tuple[int, ...]] | None = None,
    client_data_sizes: list[int] | None = None,
    *,
    actor: str = "fl_engine",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    parent_provenance_id: Any | None = None,
    seed: int = 42,
    enable_poisoning_detection: bool = False,
    poisoning_threshold: float = 3.0,
) -> FLSimulationResult:
    """Run a simulated federated learning process.

    Parameters
    ----------
    n_clients:
        Number of simulated clients (default: 5).
    n_rounds:
        Number of FL communication rounds.
    weight_shapes:
        Shapes of model weight arrays (simulates a neural network's layers).
        Defaults to [(128, 64), (64,)] — a single hidden layer.
    client_data_sizes:
        Number of training samples per client. Defaults to 100 per client.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain.
    sequence_number:
        Starting sequence number for events.
    parent_provenance_id:
        Optional UUID of a parent provenance node (e.g., dataset).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    FLSimulationResult with per-round events and provenance nodes.
    """
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")

    if weight_shapes is None:
        weight_shapes = [(128, 64), (64,)]
    if client_data_sizes is None:
        client_data_sizes = [100] * n_clients
    if len(client_data_sizes) != n_clients:
        raise ValueError(
            f"client_data_sizes length ({len(client_data_sizes)}) != n_clients ({n_clients})"
        )

    rng = np.random.default_rng(seed)

    # Initialize global model weights
    global_weights = [rng.standard_normal(shape) for shape in weight_shapes]

    current_hash = previous_hash
    current_seq = sequence_number
    rounds: list[FLRoundResult] = []

    for round_num in range(1, n_rounds + 1):
        # Save pre-round global weights for poisoning detection
        pre_round_weights = [w.copy() for w in global_weights]

        # Each client trains locally starting from global weights
        client_updates: list[ClientUpdate] = []
        for client_id in range(n_clients):
            update = _simulate_client_training(
                client_id=client_id,
                global_weights=global_weights,
                client_data_size=client_data_sizes[client_id],
                rng=rng,
            )
            client_updates.append(update)

        # FedAvg aggregation
        client_w = [u.weights for u in client_updates]
        client_n = [u.n_samples for u in client_updates]
        global_weights = _fedavg(client_w, client_n)

        # Hash the aggregated global model
        model_hash = _hash_weights(global_weights)

        # Build per-client metadata for the event
        client_meta = [
            {
                "client_id": u.client_id,
                "n_samples": u.n_samples,
                "local_loss": u.local_loss,
            }
            for u in client_updates
        ]

        # Log FL round as a forensic event
        event = ForensicEvent(
            event_type=EventType.FL_ROUND,
            actor=actor,
            payload={
                "round": round_num,
                "n_clients": n_clients,
                "global_model_hash": model_hash,
                "aggregation_strategy": "FedAvg",
                "client_updates": client_meta,
            },
            previous_hash=current_hash,
            sequence_number=current_seq,
        ).seal()

        current_hash = event.event_hash
        current_seq += 1

        # Provenance node for this round
        parent_ids = []
        relations = []
        if round_num == 1 and parent_provenance_id is not None:
            # First round links to the dataset/parent
            parent_ids = [parent_provenance_id]
            relations = [ProvenanceRelation.DERIVED_FROM]
        elif rounds:
            # Subsequent rounds link to the previous round's provenance
            parent_ids = [rounds[-1].provenance_node.id]
            relations = [ProvenanceRelation.DERIVED_FROM]

        provenance_node = ProvenanceNode(
            entity_type="fl_round",
            entity_id=f"round_{round_num}",
            metadata={
                "round": round_num,
                "n_clients": n_clients,
                "global_model_hash": model_hash,
                "aggregation_strategy": "FedAvg",
                "client_contributions": [
                    {"client_id": u.client_id, "n_samples": u.n_samples} for u in client_updates
                ],
            },
            parents=parent_ids,
            relations=relations,
        ).seal()

        # Optional poisoning detection
        poisoning_result = None
        if enable_poisoning_detection:
            poisoning_result = detect_poisoning(
                client_weights=client_w,
                global_weights=pre_round_weights,
                client_ids=[u.client_id for u in client_updates],
                round_number=round_num,
                threshold_multiplier=poisoning_threshold,
                actor=actor,
                previous_hash=current_hash,
                sequence_number=current_seq,
                round_provenance_id=provenance_node.id,
            )
            # Advance the hash chain past any alert events
            if poisoning_result.alert_events:
                current_hash = poisoning_result.alert_events[-1].event_hash
                current_seq += len(poisoning_result.alert_events)

        round_result = FLRoundResult(
            round_number=round_num,
            client_updates=client_updates,
            global_weights=global_weights,
            global_model_hash=model_hash,
            event=event,
            provenance_node=provenance_node,
            poisoning_result=poisoning_result,
        )
        rounds.append(round_result)

    return FLSimulationResult(
        n_rounds=n_rounds,
        n_clients=n_clients,
        rounds=rounds,
        final_model_hash=_hash_weights(global_weights),
    )
