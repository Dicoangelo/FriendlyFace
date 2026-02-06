"""Integration tests for the FL API endpoints (US-008).

Tests:
  POST /fl/start          — start FL simulation with configurable parameters
  GET  /fl/rounds         — list completed FL rounds with summary
  GET  /fl/rounds/{id}/{n}  — get round details with client contributions and security
  GET  /fl/status         — current FL training status
  Full lifecycle: start -> rounds -> round details -> status -> completion
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# POST /fl/start
# ---------------------------------------------------------------------------


class TestFLStartEndpoint:
    @pytest.mark.asyncio
    async def test_start_returns_201(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 2,
                "seed": 42,
            },
        )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_start_returns_simulation_id(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 2,
            },
        )
        data = resp.json()
        assert "simulation_id" in data
        assert isinstance(data["simulation_id"], str)
        assert len(data["simulation_id"]) > 0

    @pytest.mark.asyncio
    async def test_start_configurable_clients_and_rounds(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 7,
                "n_rounds": 4,
            },
        )
        data = resp.json()
        assert data["n_clients"] == 7
        assert data["n_rounds"] == 4
        assert len(data["rounds"]) == 4

    @pytest.mark.asyncio
    async def test_start_with_poisoning_detection(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 1,
                "enable_poisoning_detection": True,
                "poisoning_threshold": 3.0,
            },
        )
        data = resp.json()
        assert data["rounds"][0]["poisoning"]["n_flagged"] >= 0

    @pytest.mark.asyncio
    async def test_start_without_poisoning_detection(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 2,
                "n_rounds": 1,
                "enable_poisoning_detection": False,
            },
        )
        data = resp.json()
        # No poisoning key in round summary when disabled
        assert "poisoning" not in data["rounds"][0]

    @pytest.mark.asyncio
    async def test_start_rounds_include_event_ids(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 2,
                "n_rounds": 2,
            },
        )
        data = resp.json()
        for rnd in data["rounds"]:
            assert "event_id" in rnd
            assert "recorded_event_id" in rnd

    @pytest.mark.asyncio
    async def test_start_returns_final_model_hash(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 2,
            },
        )
        data = resp.json()
        assert "final_model_hash" in data
        assert isinstance(data["final_model_hash"], str)
        assert len(data["final_model_hash"]) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_start_default_parameters(self, client):
        """Calling /fl/start with no body should use defaults."""
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post("/fl/start", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert data["n_clients"] == 5  # default
        assert data["n_rounds"] == 3  # default

    @pytest.mark.asyncio
    async def test_simulate_legacy_still_works(self, client):
        """POST /fl/simulate should still work as a legacy alias."""
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/simulate",
            json={
                "n_clients": 2,
                "n_rounds": 1,
            },
        )
        assert resp.status_code == 201
        assert "simulation_id" in resp.json()


# ---------------------------------------------------------------------------
# GET /fl/status
# ---------------------------------------------------------------------------


class TestFLStatusEndpoint:
    @pytest.mark.asyncio
    async def test_status_empty(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.get("/fl/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_simulations"] == 0
        assert data["simulations"] == []

    @pytest.mark.asyncio
    async def test_status_after_one_simulation(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        # Start a simulation
        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 2,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get("/fl/status")
        data = resp.json()
        assert data["total_simulations"] == 1
        sim = data["simulations"][0]
        assert sim["simulation_id"] == sim_id
        assert sim["n_clients"] == 3
        assert sim["n_rounds"] == 2
        assert sim["completed_rounds"] == 2
        assert sim["status"] == "completed"
        assert "final_model_hash" in sim

    @pytest.mark.asyncio
    async def test_status_multiple_simulations(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        await client.post("/fl/start", json={"n_clients": 2, "n_rounds": 1})
        await client.post("/fl/start", json={"n_clients": 3, "n_rounds": 2})

        resp = await client.get("/fl/status")
        data = resp.json()
        assert data["total_simulations"] == 2
        assert len(data["simulations"]) == 2


# ---------------------------------------------------------------------------
# GET /fl/rounds
# ---------------------------------------------------------------------------


class TestFLRoundsListEndpoint:
    @pytest.mark.asyncio
    async def test_rounds_empty(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.get("/fl/rounds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rounds"] == 0
        assert data["rounds"] == []

    @pytest.mark.asyncio
    async def test_rounds_lists_all_rounds(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 3,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get("/fl/rounds")
        data = resp.json()
        assert data["total_rounds"] == 3
        assert len(data["rounds"]) == 3
        for rnd in data["rounds"]:
            assert rnd["simulation_id"] == sim_id
            assert "round" in rnd
            assert "global_model_hash" in rnd
            assert "event_id" in rnd
            assert "provenance_node_id" in rnd

    @pytest.mark.asyncio
    async def test_rounds_across_simulations(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        await client.post("/fl/start", json={"n_clients": 2, "n_rounds": 2})
        await client.post("/fl/start", json={"n_clients": 3, "n_rounds": 1})

        resp = await client.get("/fl/rounds")
        data = resp.json()
        assert data["total_rounds"] == 3  # 2 + 1

    @pytest.mark.asyncio
    async def test_rounds_include_security_status(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 1,
                "enable_poisoning_detection": True,
            },
        )

        resp = await client.get("/fl/rounds")
        data = resp.json()
        rnd = data["rounds"][0]
        assert "security_status" in rnd
        sec = rnd["security_status"]
        assert "has_poisoning" in sec
        assert "flagged_client_ids" in sec

    @pytest.mark.asyncio
    async def test_rounds_security_none_when_disabled(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        await client.post(
            "/fl/start",
            json={
                "n_clients": 2,
                "n_rounds": 1,
                "enable_poisoning_detection": False,
            },
        )

        resp = await client.get("/fl/rounds")
        data = resp.json()
        assert data["rounds"][0]["security_status"] is None


# ---------------------------------------------------------------------------
# GET /fl/rounds/{simulation_id}/{round_number}
# ---------------------------------------------------------------------------


class TestFLRoundDetailsEndpoint:
    @pytest.mark.asyncio
    async def test_round_details_success(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 2,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation_id"] == sim_id
        assert data["round"] == 1
        assert data["n_clients"] == 3

    @pytest.mark.asyncio
    async def test_round_details_includes_client_contributions(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 4,
                "n_rounds": 1,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1")
        data = resp.json()
        assert "client_contributions" in data
        contributions = data["client_contributions"]
        assert len(contributions) == 4
        for contrib in contributions:
            assert "client_id" in contrib
            assert "n_samples" in contrib
            assert "local_loss" in contrib

    @pytest.mark.asyncio
    async def test_round_details_includes_event_references(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 2,
                "n_rounds": 1,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1")
        data = resp.json()
        assert "event_id" in data
        assert "provenance_node_id" in data

    @pytest.mark.asyncio
    async def test_round_details_includes_security_status(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 1,
                "enable_poisoning_detection": True,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1")
        data = resp.json()
        sec = data["security_status"]
        assert sec is not None
        assert sec["poisoning_detection_enabled"] is True
        assert "has_poisoning" in sec
        assert "median_norm" in sec
        assert "flagged_client_ids" in sec
        assert "provenance_node_id" in sec

    @pytest.mark.asyncio
    async def test_round_details_security_none_when_disabled(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 2,
                "n_rounds": 1,
                "enable_poisoning_detection": False,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1")
        data = resp.json()
        assert data["security_status"] is None

    @pytest.mark.asyncio
    async def test_round_details_simulation_not_found(self, client):
        resp = await client.get("/fl/rounds/nonexistent/1")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_round_details_round_not_found(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 2,
                "n_rounds": 1,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/99")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_round_details_global_model_hash(self, client):
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 2,
            },
        )
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1")
        data = resp.json()
        assert "global_model_hash" in data
        assert isinstance(data["global_model_hash"], str)
        assert len(data["global_model_hash"]) == 64


# ---------------------------------------------------------------------------
# Integration: FL lifecycle (start -> rounds -> round details -> completion)
# ---------------------------------------------------------------------------


class TestFLLifecycle:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, client):
        """Complete FL lifecycle: start -> list rounds -> round details -> status."""
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        # 1. Start FL simulation
        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 3,
                "n_rounds": 3,
                "enable_poisoning_detection": True,
                "seed": 42,
            },
        )
        assert resp.status_code == 201
        start_data = resp.json()
        sim_id = start_data["simulation_id"]
        assert start_data["n_rounds"] == 3
        assert start_data["n_clients"] == 3
        assert len(start_data["rounds"]) == 3

        # 2. List all rounds
        resp = await client.get("/fl/rounds")
        assert resp.status_code == 200
        rounds_data = resp.json()
        assert rounds_data["total_rounds"] == 3
        round_numbers = [r["round"] for r in rounds_data["rounds"]]
        assert sorted(round_numbers) == [1, 2, 3]

        # 3. Get details for each round
        for round_num in [1, 2, 3]:
            resp = await client.get(f"/fl/rounds/{sim_id}/{round_num}")
            assert resp.status_code == 200
            detail = resp.json()
            assert detail["round"] == round_num
            assert detail["simulation_id"] == sim_id
            assert len(detail["client_contributions"]) == 3
            assert detail["security_status"] is not None
            assert "event_id" in detail
            assert "provenance_node_id" in detail

        # 4. Check security for each round
        for round_num in [1, 2, 3]:
            resp = await client.get(f"/fl/rounds/{sim_id}/{round_num}/security")
            assert resp.status_code == 200
            sec_data = resp.json()
            assert sec_data["poisoning_detection_enabled"] is True
            assert sec_data["n_clients"] == 3

        # 5. Verify status shows completion
        resp = await client.get("/fl/status")
        assert resp.status_code == 200
        status_data = resp.json()
        assert status_data["total_simulations"] == 1
        sim_status = status_data["simulations"][0]
        assert sim_status["simulation_id"] == sim_id
        assert sim_status["status"] == "completed"
        assert sim_status["completed_rounds"] == 3

    @pytest.mark.asyncio
    async def test_lifecycle_events_recorded_in_forensic_chain(self, client):
        """FL events should be persisted in the forensic event chain."""
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        # Start simulation
        resp = await client.post(
            "/fl/start",
            json={
                "n_clients": 2,
                "n_rounds": 2,
            },
        )
        assert resp.status_code == 201
        start_data = resp.json()

        # Verify recorded events exist in the event store
        for rnd in start_data["rounds"]:
            recorded_id = rnd["recorded_event_id"]
            resp = await client.get(f"/events/{recorded_id}")
            assert resp.status_code == 200
            event_data = resp.json()
            assert event_data["event_type"] == "fl_round"
            assert event_data["payload"]["round"] == rnd["round"]

    @pytest.mark.asyncio
    async def test_lifecycle_multiple_simulations(self, client):
        """Multiple simulations should all be tracked independently."""
        import friendlyface.api.app as app_module

        app_module._fl_simulations.clear()

        # Start two simulations
        resp1 = await client.post("/fl/start", json={"n_clients": 2, "n_rounds": 1})
        resp2 = await client.post("/fl/start", json={"n_clients": 4, "n_rounds": 2})
        sim1 = resp1.json()["simulation_id"]
        sim2 = resp2.json()["simulation_id"]

        # Status shows both
        resp = await client.get("/fl/status")
        data = resp.json()
        assert data["total_simulations"] == 2
        sim_ids = {s["simulation_id"] for s in data["simulations"]}
        assert sim1 in sim_ids
        assert sim2 in sim_ids

        # Rounds lists all
        resp = await client.get("/fl/rounds")
        data = resp.json()
        assert data["total_rounds"] == 3  # 1 + 2

        # Each simulation's rounds are independent
        sim1_rounds = [r for r in data["rounds"] if r["simulation_id"] == sim1]
        sim2_rounds = [r for r in data["rounds"] if r["simulation_id"] == sim2]
        assert len(sim1_rounds) == 1
        assert len(sim2_rounds) == 2
