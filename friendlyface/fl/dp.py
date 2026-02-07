"""Differential privacy for federated learning rounds.

Adds gradient clipping and calibrated Gaussian noise to FedAvg
aggregation, tracking cumulative privacy budget via simple composition.
Each DP-FedAvg round is logged as a ForensicEvent(FL_ROUND) with
privacy metadata and linked provenance.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass

import numpy as np

from friendlyface.core.models import (
    EventType,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
)


def _hash_weights(weights: list[np.ndarray]) -> str:
    """Compute a SHA-256 digest of serialized model weights."""
    return hashlib.sha256(pickle.dumps(weights)).hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """Differential privacy configuration for a federated learning run."""

    epsilon: float = 1.0  # Privacy budget per round
    delta: float = 1e-5  # Privacy failure probability
    max_grad_norm: float = 1.0  # Gradient clipping bound
    noise_multiplier: float = 0.0  # Auto-computed from epsilon/delta if 0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class DPRoundResult:
    """Result from a single DP-FedAvg round."""

    round_number: int
    global_weights: list[np.ndarray]
    global_model_hash: str
    privacy_spent: float  # Cumulative epsilon spent
    noise_scale: float  # Actual noise sigma used
    clipped_clients: list[str]  # Client IDs whose gradients were clipped
    event: ForensicEvent
    provenance_node: ProvenanceNode


# ---------------------------------------------------------------------------
# Core DP primitives
# ---------------------------------------------------------------------------


def clip_gradient(
    update: list[np.ndarray],
    max_norm: float,
) -> tuple[list[np.ndarray], bool]:
    """Clip a gradient update to ``max_norm`` (L2).

    Parameters
    ----------
    update:
        List of numpy arrays representing the gradient (or weight delta).
    max_norm:
        Maximum allowed L2 norm.

    Returns
    -------
    Tuple of (clipped_update, was_clipped).
    """
    total_norm = math.sqrt(sum(float(np.sum(u**2)) for u in update))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        clipped = [u * scale for u in update]
        return clipped, True
    return [u.copy() for u in update], False


def add_dp_noise(
    aggregated: list[np.ndarray],
    noise_scale: float,
    seed: int,
) -> list[np.ndarray]:
    """Add calibrated Gaussian noise to aggregated gradients.

    Parameters
    ----------
    aggregated:
        List of numpy arrays (the averaged gradient).
    noise_scale:
        Standard deviation (sigma) of the Gaussian noise.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Noisy copy of the aggregated arrays.
    """
    rng = np.random.default_rng(seed)
    return [a + rng.normal(0.0, noise_scale, size=a.shape) for a in aggregated]


def compute_noise_multiplier(
    epsilon: float,
    delta: float,
    n_clients: int,
) -> float:
    """Compute the Gaussian noise multiplier (sigma) from privacy budget.

    Uses the analytic Gaussian mechanism bound:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    For FedAvg with per-client clipping to ``max_grad_norm = 1`` and
    ``n_clients`` participants, the L2 sensitivity of the average is
    ``1 / n_clients``.  The caller multiplies the returned sigma by
    ``max_grad_norm / n_clients`` externally (or this function bakes in
    the ``1 / n_clients`` factor).

    Parameters
    ----------
    epsilon:
        Privacy budget (per round).
    delta:
        Privacy failure probability.
    n_clients:
        Number of participating clients (affects sensitivity).

    Returns
    -------
    Noise multiplier sigma (already accounts for ``1 / n_clients``).
    """
    sensitivity = 1.0 / n_clients
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# ---------------------------------------------------------------------------
# DP-FedAvg round
# ---------------------------------------------------------------------------


def dp_fedavg_round(
    client_updates: list[list[np.ndarray]],
    global_weights: list[np.ndarray],
    dp_config: DPConfig,
    round_number: int,
    client_ids: list[str],
    *,
    cumulative_epsilon: float = 0.0,
    actor: str = "dp_fl_engine",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    parent_provenance_id: object | None = None,
    seed: int = 42,
) -> DPRoundResult:
    """Execute one DP-FedAvg round.

    Steps:
    1. Compute each client's gradient delta (update - global_weights).
    2. Clip each delta to ``dp_config.max_grad_norm``.
    3. Average the clipped deltas.
    4. Add calibrated Gaussian noise.
    5. Apply: ``new_global = global_weights + noisy_avg_delta``.

    Parameters
    ----------
    client_updates:
        List of weight lists, one per client (post-local-training).
    global_weights:
        Global model weights from before local training.
    dp_config:
        Differential privacy configuration.
    round_number:
        Current FL round number.
    client_ids:
        String identifiers for each client (same order as client_updates).
    cumulative_epsilon:
        Privacy budget already spent in prior rounds.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain.
    sequence_number:
        Sequence number for the forensic event.
    parent_provenance_id:
        Optional provenance node ID to link as parent.
    seed:
        Random seed for noise generation.

    Returns
    -------
    DPRoundResult with updated global weights, privacy accounting,
    forensic event, and provenance node.
    """
    if len(client_updates) != len(client_ids):
        raise ValueError(
            f"client_updates length ({len(client_updates)}) != "
            f"client_ids length ({len(client_ids)})"
        )
    if len(client_updates) == 0:
        raise ValueError("No client updates provided")

    n_clients = len(client_updates)

    # --- 1. Compute deltas and clip ---
    clipped_deltas: list[list[np.ndarray]] = []
    clipped_clients: list[str] = []

    for cid, client_w in zip(client_ids, client_updates):
        delta = [cw - gw for cw, gw in zip(client_w, global_weights)]
        clipped_delta, was_clipped = clip_gradient(delta, dp_config.max_grad_norm)
        clipped_deltas.append(clipped_delta)
        if was_clipped:
            clipped_clients.append(cid)

    # --- 2. Average clipped deltas ---
    n_layers = len(global_weights)
    avg_delta: list[np.ndarray] = []
    for layer_idx in range(n_layers):
        layer_sum = sum(d[layer_idx] for d in clipped_deltas)
        avg_delta.append(layer_sum / n_clients)

    # --- 3. Compute noise scale and add noise ---
    if dp_config.noise_multiplier > 0:
        noise_scale = dp_config.noise_multiplier
    else:
        noise_scale = compute_noise_multiplier(dp_config.epsilon, dp_config.delta, n_clients)

    noisy_delta = add_dp_noise(avg_delta, noise_scale, seed)

    # --- 4. Apply update ---
    new_global = [gw + nd for gw, nd in zip(global_weights, noisy_delta)]

    # --- 5. Privacy accounting (simple composition) ---
    privacy_spent = cumulative_epsilon + dp_config.epsilon

    # --- 6. Hash the new model ---
    model_hash = _hash_weights(new_global)

    # --- 7. Forensic event ---
    event = ForensicEvent(
        event_type=EventType.FL_ROUND,
        actor=actor,
        payload={
            "round": round_number,
            "n_clients": n_clients,
            "global_model_hash": model_hash,
            "aggregation_strategy": "DP-FedAvg",
            "dp_config": {
                "epsilon": dp_config.epsilon,
                "delta": dp_config.delta,
                "max_grad_norm": dp_config.max_grad_norm,
                "noise_multiplier": dp_config.noise_multiplier,
            },
            "noise_scale": noise_scale,
            "clipped_clients": clipped_clients,
            "privacy_spent": privacy_spent,
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # --- 8. Provenance node ---
    parent_ids = []
    relations = []
    if parent_provenance_id is not None:
        parent_ids = [parent_provenance_id]
        relations = [ProvenanceRelation.DERIVED_FROM]

    provenance_node = ProvenanceNode(
        entity_type="dp_fl_round",
        entity_id=f"dp_round_{round_number}",
        metadata={
            "round": round_number,
            "n_clients": n_clients,
            "global_model_hash": model_hash,
            "aggregation_strategy": "DP-FedAvg",
            "noise_scale": noise_scale,
            "clipped_clients": clipped_clients,
            "privacy_spent": privacy_spent,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    return DPRoundResult(
        round_number=round_number,
        global_weights=new_global,
        global_model_hash=model_hash,
        privacy_spent=privacy_spent,
        noise_scale=noise_scale,
        clipped_clients=clipped_clients,
        event=event,
        provenance_node=provenance_node,
    )
