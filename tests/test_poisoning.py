"""Tests for data poisoning detection in federated learning."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from friendlyface.core.models import EventType, ProvenanceRelation
from friendlyface.fl.poisoning import (
    ClientPoisoningResult,
    PoisoningDetectionResult,
    _compute_update_norm,
    detect_poisoning,
)


# ---------------------------------------------------------------------------
# Helper: build synthetic client weights with optional poisoned clients
# ---------------------------------------------------------------------------


def _make_weights(
    n_clients: int,
    global_weights: list[np.ndarray],
    *,
    poisoned_ids: set[int] | None = None,
    poison_scale: float = 100.0,
    normal_scale: float = 0.01,
    seed: int = 42,
) -> list[list[np.ndarray]]:
    """Generate per-client weights with optional poisoned outliers."""
    rng = np.random.default_rng(seed)
    poisoned_ids = poisoned_ids or set()
    client_weights = []
    for cid in range(n_clients):
        scale = poison_scale if cid in poisoned_ids else normal_scale
        perturbed = [w + rng.normal(0, scale, size=w.shape) for w in global_weights]
        client_weights.append(perturbed)
    return client_weights


# ---------------------------------------------------------------------------
# Tests: _compute_update_norm
# ---------------------------------------------------------------------------


class TestComputeUpdateNorm:
    def test_zero_delta_gives_zero_norm(self):
        w = [np.array([1.0, 2.0, 3.0])]
        assert _compute_update_norm(w, w) == 0.0

    def test_known_norm(self):
        global_w = [np.array([0.0, 0.0])]
        client_w = [np.array([3.0, 4.0])]
        norm = _compute_update_norm(client_w, global_w)
        assert abs(norm - 5.0) < 1e-10

    def test_multi_layer(self):
        g = [np.array([0.0]), np.array([0.0, 0.0])]
        c = [np.array([3.0]), np.array([4.0, 0.0])]
        # sqrt(9 + 16) = 5
        assert abs(_compute_update_norm(c, g) - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# Tests: detect_poisoning — input validation
# ---------------------------------------------------------------------------


class TestDetectPoisoningValidation:
    def test_mismatched_lengths_raises(self):
        g = [np.array([1.0])]
        cw = [g, g, g]
        with pytest.raises(ValueError, match="client_weights length"):
            detect_poisoning(
                client_weights=cw,
                global_weights=g,
                client_ids=[0, 1],
                round_number=1,
            )

    def test_empty_clients_raises(self):
        with pytest.raises(ValueError, match="No client updates"):
            detect_poisoning(
                client_weights=[],
                global_weights=[np.array([1.0])],
                client_ids=[],
                round_number=1,
            )


# ---------------------------------------------------------------------------
# Tests: detect_poisoning — clean clients (no poisoning)
# ---------------------------------------------------------------------------


class TestDetectPoisoningClean:
    def test_no_flags_when_all_clean(self):
        g = [np.zeros((10, 5))]
        cw = _make_weights(5, g, normal_scale=0.01)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=list(range(5)),
            round_number=1,
        )
        assert isinstance(result, PoisoningDetectionResult)
        assert result.flagged_client_ids == []
        assert not result.has_poisoning
        assert len(result.alert_events) == 0
        assert result.n_clients == 5
        assert result.round_number == 1

    def test_client_results_structure(self):
        g = [np.zeros((5,))]
        cw = _make_weights(3, g, normal_scale=0.01)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=[0, 1, 2],
            round_number=1,
        )
        assert len(result.client_results) == 3
        for cr in result.client_results:
            assert isinstance(cr, ClientPoisoningResult)
            assert cr.update_norm > 0
            assert not cr.flagged
            assert cr.threshold_used > 0

    def test_median_norm_computed(self):
        g = [np.zeros((5,))]
        cw = _make_weights(4, g, normal_scale=0.01)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=[0, 1, 2, 3],
            round_number=1,
        )
        assert result.median_norm > 0
        assert result.effective_threshold == result.median_norm * result.threshold_multiplier


# ---------------------------------------------------------------------------
# Tests: detect_poisoning — poisoned clients
# ---------------------------------------------------------------------------


class TestDetectPoisoningPoisoned:
    def test_flags_poisoned_client(self):
        g = [np.zeros((10, 5))]
        cw = _make_weights(5, g, poisoned_ids={2}, poison_scale=100.0)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=list(range(5)),
            round_number=1,
        )
        assert result.has_poisoning
        assert 2 in result.flagged_client_ids
        # Poisoned client's norm should be much higher
        poisoned_cr = [cr for cr in result.client_results if cr.client_id == 2][0]
        clean_cr = [cr for cr in result.client_results if cr.client_id == 0][0]
        assert poisoned_cr.update_norm > clean_cr.update_norm * 10
        assert poisoned_cr.flagged

    def test_multiple_poisoned_clients(self):
        g = [np.zeros((10, 5))]
        cw = _make_weights(5, g, poisoned_ids={1, 3}, poison_scale=100.0)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=list(range(5)),
            round_number=1,
        )
        assert set(result.flagged_client_ids) == {1, 3}
        assert len(result.alert_events) == 2

    def test_configurable_threshold(self):
        """Lower threshold flags more aggressively."""
        g = [np.zeros((10, 5))]
        # Use a moderate poison scale that might only be caught at lower threshold
        cw = _make_weights(5, g, poisoned_ids={0}, poison_scale=5.0, normal_scale=1.0)

        strict = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=list(range(5)),
            round_number=1,
            threshold_multiplier=1.5,
        )
        relaxed = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=list(range(5)),
            round_number=1,
            threshold_multiplier=100.0,
        )
        # Strict should flag at least as many as relaxed
        assert len(strict.flagged_client_ids) >= len(relaxed.flagged_client_ids)


# ---------------------------------------------------------------------------
# Tests: forensic events for flagged clients
# ---------------------------------------------------------------------------


class TestPoisoningForensicEvents:
    def _get_poisoned_result(self) -> PoisoningDetectionResult:
        g = [np.zeros((10, 5))]
        cw = _make_weights(5, g, poisoned_ids={2}, poison_scale=100.0)
        return detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=list(range(5)),
            round_number=1,
            actor="test_detector",
        )

    def test_alert_event_type_is_security_alert(self):
        result = self._get_poisoned_result()
        assert len(result.alert_events) == 1
        alert = result.alert_events[0]
        assert alert.event_type == EventType.SECURITY_ALERT

    def test_alert_event_is_sealed_and_verifiable(self):
        result = self._get_poisoned_result()
        for alert in result.alert_events:
            assert alert.event_hash != ""
            assert alert.verify()

    def test_alert_event_payload(self):
        result = self._get_poisoned_result()
        alert = result.alert_events[0]
        payload = alert.payload
        assert payload["alert_type"] == "data_poisoning"
        assert payload["round"] == 1
        assert payload["client_id"] == 2
        assert payload["update_norm"] > 0
        assert payload["median_norm"] > 0
        assert payload["threshold_multiplier"] == 3.0
        assert payload["effective_threshold"] > 0

    def test_alert_event_actor(self):
        result = self._get_poisoned_result()
        assert result.alert_events[0].actor == "test_detector"

    def test_alert_events_form_hash_chain(self):
        g = [np.zeros((10, 5))]
        cw = _make_weights(5, g, poisoned_ids={1, 3}, poison_scale=100.0)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=list(range(5)),
            round_number=1,
            previous_hash="abc123",
            sequence_number=10,
        )
        assert len(result.alert_events) == 2
        # First alert chains from provided previous_hash
        assert result.alert_events[0].previous_hash == "abc123"
        assert result.alert_events[0].sequence_number == 10
        # Second alert chains from first
        assert result.alert_events[1].previous_hash == result.alert_events[0].event_hash
        assert result.alert_events[1].sequence_number == 11


# ---------------------------------------------------------------------------
# Tests: provenance node
# ---------------------------------------------------------------------------


class TestPoisoningProvenance:
    def test_provenance_entity_type(self):
        g = [np.zeros((5,))]
        cw = _make_weights(3, g)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=[0, 1, 2],
            round_number=2,
        )
        assert result.provenance_node is not None
        assert result.provenance_node.entity_type == "poisoning_detection"
        assert result.provenance_node.entity_id == "poisoning_round_2"

    def test_provenance_is_sealed(self):
        g = [np.zeros((5,))]
        cw = _make_weights(3, g)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=[0, 1, 2],
            round_number=1,
        )
        node = result.provenance_node
        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()

    def test_provenance_metadata(self):
        g = [np.zeros((5,))]
        cw = _make_weights(3, g, poisoned_ids={1}, poison_scale=100.0)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=[0, 1, 2],
            round_number=1,
        )
        meta = result.provenance_node.metadata
        assert meta["round"] == 1
        assert meta["n_clients"] == 3
        assert meta["median_norm"] > 0
        assert meta["threshold_multiplier"] == 3.0
        assert 1 in meta["flagged_client_ids"]
        assert meta["n_flagged"] == 1

    def test_provenance_links_to_round(self):
        parent_id = uuid4()
        g = [np.zeros((5,))]
        cw = _make_weights(3, g)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=[0, 1, 2],
            round_number=1,
            round_provenance_id=parent_id,
        )
        node = result.provenance_node
        assert parent_id in node.parents
        assert ProvenanceRelation.DERIVED_FROM in node.relations

    def test_provenance_no_parent_without_round_id(self):
        g = [np.zeros((5,))]
        cw = _make_weights(3, g)
        result = detect_poisoning(
            client_weights=cw,
            global_weights=g,
            client_ids=[0, 1, 2],
            round_number=1,
        )
        assert result.provenance_node.parents == []


# ---------------------------------------------------------------------------
# Tests: FL engine integration
# ---------------------------------------------------------------------------


class TestFLEngineIntegration:
    def test_poisoning_result_none_when_disabled(self):
        from friendlyface.fl.engine import run_fl_simulation

        result = run_fl_simulation(n_clients=3, n_rounds=1)
        assert result.rounds[0].poisoning_result is None

    def test_poisoning_result_present_when_enabled(self):
        from friendlyface.fl.engine import run_fl_simulation

        result = run_fl_simulation(n_clients=3, n_rounds=2, enable_poisoning_detection=True)
        for rr in result.rounds:
            assert rr.poisoning_result is not None
            assert isinstance(rr.poisoning_result, PoisoningDetectionResult)

    def test_engine_poisoning_result_fields(self):
        from friendlyface.fl.engine import run_fl_simulation

        result = run_fl_simulation(n_clients=3, n_rounds=1, enable_poisoning_detection=True)
        pr = result.rounds[0].poisoning_result
        assert pr.n_clients == 3
        assert pr.round_number == 1
        assert pr.median_norm > 0
        assert pr.provenance_node is not None

    def test_engine_poisoning_links_to_round_provenance(self):
        from friendlyface.fl.engine import run_fl_simulation

        result = run_fl_simulation(n_clients=3, n_rounds=1, enable_poisoning_detection=True)
        rr = result.rounds[0]
        pr = rr.poisoning_result
        assert rr.provenance_node.id in pr.provenance_node.parents

    def test_engine_clean_simulation_no_flags(self):
        """Normal simulation with small perturbations shouldn't flag anyone."""
        from friendlyface.fl.engine import run_fl_simulation

        result = run_fl_simulation(n_clients=5, n_rounds=3, enable_poisoning_detection=True)
        for rr in result.rounds:
            assert not rr.poisoning_result.has_poisoning

    def test_engine_hash_chain_continues_past_alerts(self):
        """Event hash chain should remain consistent with poisoning enabled."""
        from friendlyface.fl.engine import run_fl_simulation

        result = run_fl_simulation(n_clients=3, n_rounds=2, enable_poisoning_detection=True)
        # Round events still form a chain (though alert events may be interspersed)
        r1 = result.rounds[0]
        r2 = result.rounds[1]
        # Round 2's event should chain from round 1's event or alert chain
        if r1.poisoning_result and r1.poisoning_result.alert_events:
            expected_prev = r1.poisoning_result.alert_events[-1].event_hash
        else:
            expected_prev = r1.event.event_hash
        assert r2.event.previous_hash == expected_prev


# ---------------------------------------------------------------------------
# Tests: API endpoint
# ---------------------------------------------------------------------------


class TestFLSecurityAPI:
    @pytest.mark.asyncio
    async def test_simulate_with_poisoning_detection(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/simulate",
            json={
                "n_clients": 3,
                "n_rounds": 2,
                "enable_poisoning_detection": True,
                "seed": 42,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "simulation_id" in data
        assert data["n_rounds"] == 2
        assert data["n_clients"] == 3
        assert len(data["rounds"]) == 2
        for rnd in data["rounds"]:
            assert "poisoning" in rnd
            assert "flagged_client_ids" in rnd["poisoning"]

    @pytest.mark.asyncio
    async def test_security_endpoint_returns_results(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        # Run a simulation first
        resp = await client.post(
            "/fl/simulate",
            json={
                "n_clients": 3,
                "n_rounds": 1,
                "enable_poisoning_detection": True,
            },
        )
        sim_id = resp.json()["simulation_id"]

        # Query security for round 1
        resp = await client.get(f"/fl/rounds/{sim_id}/1/security")
        assert resp.status_code == 200
        data = resp.json()
        assert data["round"] == 1
        assert data["poisoning_detection_enabled"] is True
        assert data["n_clients"] == 3
        assert "median_norm" in data
        assert "effective_threshold" in data
        assert "client_results" in data
        assert len(data["client_results"]) == 3

    @pytest.mark.asyncio
    async def test_security_endpoint_simulation_not_found(self, client):
        resp = await client.get("/fl/rounds/nonexistent/1/security")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_security_endpoint_round_not_found(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/simulate",
            json={
                "n_clients": 2,
                "n_rounds": 1,
                "enable_poisoning_detection": True,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/99/security")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_security_endpoint_detection_disabled(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/simulate",
            json={
                "n_clients": 2,
                "n_rounds": 1,
                "enable_poisoning_detection": False,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1/security")
        assert resp.status_code == 200
        data = resp.json()
        assert data["poisoning_detection_enabled"] is False
