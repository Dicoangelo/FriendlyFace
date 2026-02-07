"""Tests for differential privacy module in federated learning."""

from __future__ import annotations

import math
from uuid import uuid4

import numpy as np
import pytest

from friendlyface.core.models import EventType, ProvenanceRelation
from friendlyface.fl.dp import (
    DPConfig,
    DPRoundResult,
    add_dp_noise,
    clip_gradient,
    compute_noise_multiplier,
    dp_fedavg_round,
)


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


class TestClipGradient:
    def test_clips_when_norm_exceeds_threshold(self):
        """Gradient with norm > max_norm should be scaled down."""
        update = [np.array([3.0, 4.0])]  # L2 norm = 5.0
        clipped, was_clipped = clip_gradient(update, max_norm=1.0)

        assert was_clipped is True
        result_norm = math.sqrt(sum(float(np.sum(u**2)) for u in clipped))
        assert result_norm == pytest.approx(1.0, abs=1e-8)

    def test_no_clip_when_within_threshold(self):
        """Gradient with norm <= max_norm should be unchanged."""
        update = [np.array([0.3, 0.4])]  # L2 norm = 0.5
        clipped, was_clipped = clip_gradient(update, max_norm=1.0)

        assert was_clipped is False
        np.testing.assert_allclose(clipped[0], update[0])

    def test_preserves_direction_after_clipping(self):
        """Clipped gradient should point in the same direction."""
        update = [np.array([6.0, 8.0])]  # norm = 10
        clipped, _ = clip_gradient(update, max_norm=2.0)

        # Direction: [0.6, 0.8] scaled to norm 2.0 -> [1.2, 1.6]
        np.testing.assert_allclose(clipped[0], [1.2, 1.6], atol=1e-10)

    def test_multi_layer_clipping(self):
        """Clipping works across multiple layers (single global norm)."""
        update = [np.array([3.0]), np.array([4.0])]  # total norm = 5.0
        clipped, was_clipped = clip_gradient(update, max_norm=1.0)

        assert was_clipped is True
        total_norm = math.sqrt(sum(float(np.sum(u**2)) for u in clipped))
        assert total_norm == pytest.approx(1.0, abs=1e-8)

    def test_exact_threshold_not_clipped(self):
        """Gradient exactly at max_norm should not be clipped."""
        update = [np.array([0.6, 0.8])]  # norm = 1.0
        clipped, was_clipped = clip_gradient(update, max_norm=1.0)

        assert was_clipped is False
        np.testing.assert_allclose(clipped[0], update[0])

    def test_returns_copy_when_not_clipped(self):
        """Should return a copy, not a reference to the original."""
        update = [np.array([0.1, 0.2])]
        clipped, _ = clip_gradient(update, max_norm=10.0)

        clipped[0][0] = 999.0
        assert update[0][0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# DP noise addition
# ---------------------------------------------------------------------------


class TestAddDPNoise:
    def test_noise_is_nonzero(self):
        """Noise should actually modify the aggregated values."""
        aggregated = [np.zeros((5,))]
        noisy = add_dp_noise(aggregated, noise_scale=1.0, seed=42)

        assert not np.allclose(noisy[0], 0.0)

    def test_seeded_reproducibility(self):
        """Same seed should produce identical noise."""
        agg = [np.ones((10,))]
        r1 = add_dp_noise(agg, noise_scale=0.5, seed=123)
        r2 = add_dp_noise(agg, noise_scale=0.5, seed=123)

        np.testing.assert_array_equal(r1[0], r2[0])

    def test_different_seeds_different_noise(self):
        """Different seeds should produce different noise."""
        agg = [np.ones((10,))]
        r1 = add_dp_noise(agg, noise_scale=0.5, seed=1)
        r2 = add_dp_noise(agg, noise_scale=0.5, seed=2)

        assert not np.allclose(r1[0], r2[0])

    def test_noise_scale_affects_magnitude(self):
        """Larger noise_scale should produce larger perturbations on average."""
        agg = [np.zeros((1000,))]
        small = add_dp_noise(agg, noise_scale=0.01, seed=42)
        large = add_dp_noise(agg, noise_scale=10.0, seed=42)

        assert np.std(small[0]) < np.std(large[0])

    def test_multi_layer_noise(self):
        """Noise is added independently to each layer."""
        agg = [np.zeros((5,)), np.zeros((3,))]
        noisy = add_dp_noise(agg, noise_scale=1.0, seed=42)

        assert len(noisy) == 2
        assert noisy[0].shape == (5,)
        assert noisy[1].shape == (3,)
        assert not np.allclose(noisy[0], 0.0)
        assert not np.allclose(noisy[1], 0.0)


# ---------------------------------------------------------------------------
# Noise multiplier computation
# ---------------------------------------------------------------------------


class TestComputeNoiseMultiplier:
    def test_positive_result(self):
        sigma = compute_noise_multiplier(epsilon=1.0, delta=1e-5, n_clients=10)
        assert sigma > 0

    def test_smaller_epsilon_means_more_noise(self):
        """Tighter privacy budget requires more noise."""
        s1 = compute_noise_multiplier(epsilon=1.0, delta=1e-5, n_clients=10)
        s2 = compute_noise_multiplier(epsilon=0.1, delta=1e-5, n_clients=10)
        assert s2 > s1

    def test_more_clients_means_less_noise(self):
        """More clients reduce sensitivity, thus less noise."""
        s1 = compute_noise_multiplier(epsilon=1.0, delta=1e-5, n_clients=5)
        s2 = compute_noise_multiplier(epsilon=1.0, delta=1e-5, n_clients=50)
        assert s2 < s1

    def test_analytic_value(self):
        """Check against the analytic formula directly."""
        eps, delta, n = 1.0, 1e-5, 10
        expected = (1.0 / n) * math.sqrt(2.0 * math.log(1.25 / delta)) / eps
        actual = compute_noise_multiplier(eps, delta, n)
        assert actual == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Full dp_fedavg_round
# ---------------------------------------------------------------------------


class TestDPFedAvgRound:
    @pytest.fixture()
    def simple_setup(self):
        """Two clients, single-layer model, small perturbations."""
        rng = np.random.default_rng(42)
        global_weights = [rng.standard_normal((10,))]
        client_updates = [
            [global_weights[0] + rng.normal(0, 0.01, size=(10,))],
            [global_weights[0] + rng.normal(0, 0.01, size=(10,))],
        ]
        dp_config = DPConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        return global_weights, client_updates, dp_config

    def test_returns_dp_round_result(self, simple_setup):
        gw, cu, cfg = simple_setup
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0", "c1"],
        )
        assert isinstance(result, DPRoundResult)

    def test_global_weights_updated(self, simple_setup):
        gw, cu, cfg = simple_setup
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0", "c1"],
        )
        # New weights should differ from old ones (noise + averaging)
        assert not np.allclose(result.global_weights[0], gw[0])

    def test_model_hash_is_valid(self, simple_setup):
        gw, cu, cfg = simple_setup
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0", "c1"],
        )
        assert len(result.global_model_hash) == 64  # SHA-256 hex

    def test_noise_scale_positive(self, simple_setup):
        gw, cu, cfg = simple_setup
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0", "c1"],
        )
        assert result.noise_scale > 0

    def test_custom_noise_multiplier(self):
        """When noise_multiplier is set, use it instead of computing."""
        gw = [np.zeros((5,))]
        cu = [[np.ones((5,))], [np.ones((5,))]]
        cfg = DPConfig(noise_multiplier=0.5)
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0", "c1"],
        )
        assert result.noise_scale == 0.5

    def test_clipped_clients_tracked(self):
        """Clients whose deltas exceed max_grad_norm are recorded."""
        gw = [np.zeros((2,))]
        # Client 0: large delta (norm=5), Client 1: small delta (norm=0.1)
        cu = [
            [np.array([3.0, 4.0])],  # norm 5
            [np.array([0.06, 0.08])],  # norm 0.1
        ]
        cfg = DPConfig(max_grad_norm=1.0)
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["big", "small"],
        )
        assert "big" in result.clipped_clients
        assert "small" not in result.clipped_clients

    def test_reproducible_with_seed(self, simple_setup):
        gw, cu, cfg = simple_setup
        r1 = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0", "c1"],
            seed=99,
        )
        r2 = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0", "c1"],
            seed=99,
        )
        assert r1.global_model_hash == r2.global_model_hash

    def test_input_validation_length_mismatch(self):
        gw = [np.zeros((5,))]
        cu = [[np.ones((5,))]]
        cfg = DPConfig()
        with pytest.raises(ValueError, match="client_updates length"):
            dp_fedavg_round(
                client_updates=cu,
                global_weights=gw,
                dp_config=cfg,
                round_number=1,
                client_ids=["a", "b"],
            )

    def test_input_validation_empty_updates(self):
        gw = [np.zeros((5,))]
        cfg = DPConfig()
        with pytest.raises(ValueError, match="No client updates"):
            dp_fedavg_round(
                client_updates=[],
                global_weights=gw,
                dp_config=cfg,
                round_number=1,
                client_ids=[],
            )


# ---------------------------------------------------------------------------
# Privacy budget tracking
# ---------------------------------------------------------------------------


class TestPrivacyBudget:
    def test_first_round_spends_epsilon(self):
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig(epsilon=0.5)
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0"],
        )
        assert result.privacy_spent == pytest.approx(0.5)

    def test_cumulative_epsilon_tracking(self):
        """Multiple rounds accumulate epsilon via simple composition."""
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig(epsilon=0.5)

        # Simulate 3 rounds
        cumulative = 0.0
        for r in range(1, 4):
            result = dp_fedavg_round(
                client_updates=cu,
                global_weights=gw,
                dp_config=cfg,
                round_number=r,
                client_ids=["c0"],
                cumulative_epsilon=cumulative,
            )
            cumulative = result.privacy_spent

        assert cumulative == pytest.approx(1.5)  # 0.5 * 3

    def test_privacy_spent_in_event_payload(self):
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig(epsilon=1.0)
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0"],
            cumulative_epsilon=2.0,
        )
        assert result.event.payload["privacy_spent"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ForensicEvent creation with DP metadata
# ---------------------------------------------------------------------------


class TestForensicEventDP:
    @pytest.fixture()
    def dp_result(self):
        gw = [np.zeros((5,))]
        cu = [[np.array([3.0, 4.0, 0.0, 0.0, 0.0])], [np.ones((5,)) * 0.1]]
        cfg = DPConfig(epsilon=0.5, delta=1e-6, max_grad_norm=1.0)
        return dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=3,
            client_ids=["alice", "bob"],
            actor="test_dp",
            previous_hash="prev_abc",
            sequence_number=7,
        )

    def test_event_type_is_fl_round(self, dp_result):
        assert dp_result.event.event_type == EventType.FL_ROUND

    def test_event_is_sealed_and_verifiable(self, dp_result):
        assert dp_result.event.event_hash != ""
        assert dp_result.event.verify()

    def test_event_actor(self, dp_result):
        assert dp_result.event.actor == "test_dp"

    def test_event_chain_fields(self, dp_result):
        assert dp_result.event.previous_hash == "prev_abc"
        assert dp_result.event.sequence_number == 7

    def test_payload_contains_dp_config(self, dp_result):
        payload = dp_result.event.payload
        assert payload["aggregation_strategy"] == "DP-FedAvg"
        dp_cfg = payload["dp_config"]
        assert dp_cfg["epsilon"] == 0.5
        assert dp_cfg["delta"] == 1e-6
        assert dp_cfg["max_grad_norm"] == 1.0

    def test_payload_contains_noise_scale(self, dp_result):
        assert dp_result.event.payload["noise_scale"] > 0

    def test_payload_contains_clipped_clients(self, dp_result):
        clipped = dp_result.event.payload["clipped_clients"]
        assert isinstance(clipped, list)
        # alice has norm 5 > max_grad_norm 1.0
        assert "alice" in clipped

    def test_payload_contains_round_metadata(self, dp_result):
        payload = dp_result.event.payload
        assert payload["round"] == 3
        assert payload["n_clients"] == 2
        assert payload["global_model_hash"]


# ---------------------------------------------------------------------------
# Provenance node
# ---------------------------------------------------------------------------


class TestProvenanceNodeDP:
    def test_entity_type(self):
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig()
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0"],
        )
        assert result.provenance_node.entity_type == "dp_fl_round"

    def test_entity_id_includes_round(self):
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig()
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=5,
            client_ids=["c0"],
        )
        assert result.provenance_node.entity_id == "dp_round_5"

    def test_provenance_is_sealed(self):
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig()
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0"],
        )
        assert result.provenance_node.node_hash != ""
        assert result.provenance_node.node_hash == result.provenance_node.compute_hash()

    def test_provenance_links_to_parent(self):
        parent_id = uuid4()
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig()
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0"],
            parent_provenance_id=parent_id,
        )
        assert parent_id in result.provenance_node.parents
        assert result.provenance_node.relations == [ProvenanceRelation.DERIVED_FROM]

    def test_provenance_no_parent_by_default(self):
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig()
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0"],
        )
        assert result.provenance_node.parents == []
        assert result.provenance_node.relations == []

    def test_provenance_metadata_has_dp_fields(self):
        gw = [np.zeros((3,))]
        cu = [[np.ones((3,))]]
        cfg = DPConfig()
        result = dp_fedavg_round(
            client_updates=cu,
            global_weights=gw,
            dp_config=cfg,
            round_number=1,
            client_ids=["c0"],
        )
        meta = result.provenance_node.metadata
        assert "noise_scale" in meta
        assert "clipped_clients" in meta
        assert "privacy_spent" in meta
        assert meta["aggregation_strategy"] == "DP-FedAvg"
