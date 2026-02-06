"""Tests for federated learning simulation engine."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from friendlyface.core.models import EventType, ProvenanceRelation
from friendlyface.fl.engine import (
    ClientUpdate,
    FLRoundResult,
    FLSimulationResult,
    _fedavg,
    _hash_weights,
    run_fl_simulation,
)


class TestInputValidation:
    def test_n_clients_must_be_positive(self):
        with pytest.raises(ValueError, match="n_clients must be >= 1"):
            run_fl_simulation(n_clients=0)

    def test_n_rounds_must_be_positive(self):
        with pytest.raises(ValueError, match="n_rounds must be >= 1"):
            run_fl_simulation(n_rounds=0)

    def test_client_data_sizes_length_mismatch(self):
        with pytest.raises(ValueError, match="client_data_sizes length"):
            run_fl_simulation(n_clients=3, client_data_sizes=[100, 200])


class TestHashWeights:
    def test_deterministic(self):
        rng = np.random.default_rng(42)
        weights = [rng.standard_normal((10, 5)), rng.standard_normal((5,))]
        h1 = _hash_weights(weights)
        h2 = _hash_weights(weights)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_weights_different_hash(self):
        rng = np.random.default_rng(42)
        w1 = [rng.standard_normal((10, 5))]
        w2 = [rng.standard_normal((10, 5))]
        assert _hash_weights(w1) != _hash_weights(w2)


class TestFedAvg:
    def test_equal_weights_averages_correctly(self):
        """Two clients with equal data sizes: plain average."""
        w1 = [np.array([1.0, 2.0])]
        w2 = [np.array([3.0, 4.0])]
        result = _fedavg([w1, w2], [100, 100])
        np.testing.assert_allclose(result[0], [2.0, 3.0])

    def test_weighted_average(self):
        """Client with more data gets more weight."""
        w1 = [np.array([0.0])]
        w2 = [np.array([10.0])]
        # Client 2 has 3x the data
        result = _fedavg([w1, w2], [100, 300])
        np.testing.assert_allclose(result[0], [7.5])

    def test_multi_layer_weights(self):
        """FedAvg handles multiple weight arrays (layers)."""
        w1 = [np.array([1.0]), np.array([10.0, 20.0])]
        w2 = [np.array([3.0]), np.array([30.0, 40.0])]
        result = _fedavg([w1, w2], [100, 100])
        np.testing.assert_allclose(result[0], [2.0])
        np.testing.assert_allclose(result[1], [20.0, 30.0])


class TestRunFLSimulation:
    def test_basic_simulation(self):
        result = run_fl_simulation(n_clients=3, n_rounds=2)

        assert isinstance(result, FLSimulationResult)
        assert result.n_rounds == 2
        assert result.n_clients == 3
        assert len(result.rounds) == 2
        assert result.final_model_hash
        assert len(result.final_model_hash) == 64

    def test_default_five_clients(self):
        result = run_fl_simulation(n_rounds=1)
        assert result.n_clients == 5
        assert len(result.rounds[0].client_updates) == 5

    def test_round_result_structure(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1)
        rr = result.rounds[0]

        assert isinstance(rr, FLRoundResult)
        assert rr.round_number == 1
        assert len(rr.client_updates) == 2
        assert len(rr.global_weights) > 0
        assert rr.global_model_hash
        assert rr.event is not None
        assert rr.provenance_node is not None

    def test_client_update_structure(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1)
        cu = result.rounds[0].client_updates[0]

        assert isinstance(cu, ClientUpdate)
        assert cu.client_id == 0
        assert cu.n_samples == 100
        assert 0.0 <= cu.local_loss <= 1.0
        assert len(cu.weights) > 0

    def test_custom_weight_shapes(self):
        shapes = [(64, 32), (32,), (32, 10), (10,)]
        result = run_fl_simulation(n_clients=2, n_rounds=1, weight_shapes=shapes)
        rr = result.rounds[0]
        assert len(rr.global_weights) == 4
        assert rr.global_weights[0].shape == (64, 32)
        assert rr.global_weights[1].shape == (32,)
        assert rr.global_weights[2].shape == (32, 10)
        assert rr.global_weights[3].shape == (10,)

    def test_custom_client_data_sizes(self):
        sizes = [50, 200, 150]
        result = run_fl_simulation(n_clients=3, n_rounds=1, client_data_sizes=sizes)
        updates = result.rounds[0].client_updates
        assert updates[0].n_samples == 50
        assert updates[1].n_samples == 200
        assert updates[2].n_samples == 150

    def test_reproducible_with_seed(self):
        r1 = run_fl_simulation(n_clients=2, n_rounds=2, seed=99)
        r2 = run_fl_simulation(n_clients=2, n_rounds=2, seed=99)
        assert r1.final_model_hash == r2.final_model_hash
        for rnd1, rnd2 in zip(r1.rounds, r2.rounds):
            assert rnd1.global_model_hash == rnd2.global_model_hash

    def test_different_seeds_different_results(self):
        r1 = run_fl_simulation(n_clients=2, n_rounds=1, seed=1)
        r2 = run_fl_simulation(n_clients=2, n_rounds=1, seed=2)
        assert r1.final_model_hash != r2.final_model_hash

    def test_global_model_hash_changes_per_round(self):
        result = run_fl_simulation(n_clients=3, n_rounds=3)
        hashes = [r.global_model_hash for r in result.rounds]
        # Each round should produce a different hash
        assert len(set(hashes)) == 3

    def test_final_model_hash_matches_last_round(self):
        result = run_fl_simulation(n_clients=2, n_rounds=3)
        assert result.final_model_hash == result.rounds[-1].global_model_hash


class TestForensicEvents:
    def test_event_type_is_fl_round(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1)
        event = result.rounds[0].event
        assert event.event_type == EventType.FL_ROUND

    def test_event_is_sealed_and_verifiable(self):
        result = run_fl_simulation(n_clients=2, n_rounds=2)
        for rr in result.rounds:
            assert rr.event.event_hash != ""
            assert rr.event.verify()

    def test_event_payload_contains_round_metadata(self):
        result = run_fl_simulation(n_clients=3, n_rounds=1)
        payload = result.rounds[0].event.payload
        assert payload["round"] == 1
        assert payload["n_clients"] == 3
        assert payload["global_model_hash"]
        assert payload["aggregation_strategy"] == "FedAvg"
        assert len(payload["client_updates"]) == 3

    def test_event_payload_client_details(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1, client_data_sizes=[50, 150])
        clients = result.rounds[0].event.payload["client_updates"]
        assert clients[0]["client_id"] == 0
        assert clients[0]["n_samples"] == 50
        assert "local_loss" in clients[0]
        assert clients[1]["client_id"] == 1
        assert clients[1]["n_samples"] == 150

    def test_events_form_hash_chain(self):
        result = run_fl_simulation(n_clients=2, n_rounds=3)
        # First event chains from GENESIS
        assert result.rounds[0].event.previous_hash == "GENESIS"
        # Each subsequent event chains from the previous
        for i in range(1, len(result.rounds)):
            assert result.rounds[i].event.previous_hash == result.rounds[i - 1].event.event_hash

    def test_custom_previous_hash_and_sequence(self):
        result = run_fl_simulation(
            n_clients=2,
            n_rounds=2,
            previous_hash="abc123",
            sequence_number=10,
        )
        assert result.rounds[0].event.previous_hash == "abc123"
        assert result.rounds[0].event.sequence_number == 10
        assert result.rounds[1].event.sequence_number == 11

    def test_event_actor(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1, actor="test_actor")
        assert result.rounds[0].event.actor == "test_actor"


class TestProvenanceNodes:
    def test_provenance_entity_type_is_fl_round(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1)
        node = result.rounds[0].provenance_node
        assert node.entity_type == "fl_round"

    def test_provenance_is_sealed_and_verifiable(self):
        result = run_fl_simulation(n_clients=2, n_rounds=2)
        for rr in result.rounds:
            assert rr.provenance_node.node_hash != ""
            assert rr.provenance_node.node_hash == rr.provenance_node.compute_hash()

    def test_provenance_entity_id_has_round_number(self):
        result = run_fl_simulation(n_clients=2, n_rounds=3)
        for i, rr in enumerate(result.rounds, 1):
            assert rr.provenance_node.entity_id == f"round_{i}"

    def test_provenance_metadata_contains_round_info(self):
        result = run_fl_simulation(n_clients=3, n_rounds=1)
        meta = result.rounds[0].provenance_node.metadata
        assert meta["round"] == 1
        assert meta["n_clients"] == 3
        assert meta["global_model_hash"]
        assert meta["aggregation_strategy"] == "FedAvg"
        assert len(meta["client_contributions"]) == 3

    def test_provenance_client_contributions(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1, client_data_sizes=[50, 150])
        contribs = result.rounds[0].provenance_node.metadata["client_contributions"]
        assert contribs[0] == {"client_id": 0, "n_samples": 50}
        assert contribs[1] == {"client_id": 1, "n_samples": 150}

    def test_provenance_chain_links_rounds(self):
        """Subsequent rounds link to previous round's provenance."""
        result = run_fl_simulation(n_clients=2, n_rounds=3)
        # Round 1: no parents (no parent_provenance_id given)
        assert result.rounds[0].provenance_node.parents == []
        # Round 2: parent is round 1
        assert result.rounds[1].provenance_node.parents == [result.rounds[0].provenance_node.id]
        assert result.rounds[1].provenance_node.relations == [ProvenanceRelation.DERIVED_FROM]
        # Round 3: parent is round 2
        assert result.rounds[2].provenance_node.parents == [result.rounds[1].provenance_node.id]

    def test_provenance_first_round_links_to_parent(self):
        parent_id = uuid4()
        result = run_fl_simulation(n_clients=2, n_rounds=2, parent_provenance_id=parent_id)
        # Round 1 links to the provided parent
        assert parent_id in result.rounds[0].provenance_node.parents
        assert result.rounds[0].provenance_node.relations == [ProvenanceRelation.DERIVED_FROM]
        # Round 2 links to round 1 (not the original parent)
        assert result.rounds[1].provenance_node.parents == [result.rounds[0].provenance_node.id]

    def test_provenance_no_parent_by_default(self):
        result = run_fl_simulation(n_clients=2, n_rounds=1)
        assert result.rounds[0].provenance_node.parents == []
        assert result.rounds[0].provenance_node.relations == []
