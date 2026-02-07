"""Tests for US-053: FL Simulation Gating.

Verifies that all FL endpoints include:
- X-FL-Mode response header
- "mode" field in response body
- fl_mode config validation
"""

from __future__ import annotations

import pytest
from friendlyface.config import Settings


class TestFLModeConfig:
    """Test FL mode configuration validation."""

    def test_default_fl_mode(self):
        s = Settings(api_keys="", _env_file=None)
        assert s.fl_mode == "simulation"

    def test_fl_mode_simulation(self):
        s = Settings(api_keys="", fl_mode="simulation", _env_file=None)
        assert s.fl_mode == "simulation"

    def test_fl_mode_production(self):
        s = Settings(api_keys="", fl_mode="production", _env_file=None)
        assert s.fl_mode == "production"

    def test_fl_mode_case_insensitive(self):
        s = Settings(api_keys="", fl_mode="SIMULATION", _env_file=None)
        assert s.fl_mode == "simulation"

    def test_fl_mode_invalid(self):
        with pytest.raises(ValueError, match="FF_FL_MODE must be"):
            Settings(api_keys="", fl_mode="invalid", _env_file=None)


class TestFLGatingHeaders:
    """Test that FL endpoints return X-FL-Mode header and mode in body."""

    @pytest.mark.asyncio
    async def test_fl_start_has_mode_header(self, client):
        resp = await client.post("/fl/start", json={"n_clients": 2, "n_rounds": 1, "seed": 42})
        assert resp.status_code == 201
        assert resp.headers.get("x-fl-mode") == "simulation"
        body = resp.json()
        assert body["mode"] == "simulation"

    @pytest.mark.asyncio
    async def test_fl_simulate_has_mode_header(self, client):
        resp = await client.post("/fl/simulate", json={"n_clients": 2, "n_rounds": 1, "seed": 42})
        assert resp.status_code == 201
        assert resp.headers.get("x-fl-mode") == "simulation"
        assert resp.json()["mode"] == "simulation"

    @pytest.mark.asyncio
    async def test_fl_dp_start_has_mode_header(self, client):
        resp = await client.post(
            "/fl/dp-start",
            json={"n_clients": 2, "n_rounds": 1, "epsilon": 1.0, "delta": 1e-5, "seed": 42},
        )
        assert resp.status_code == 201
        assert resp.headers.get("x-fl-mode") == "simulation"
        assert resp.json()["mode"] == "simulation"

    @pytest.mark.asyncio
    async def test_fl_status_has_mode_header(self, client):
        resp = await client.get("/fl/status")
        assert resp.status_code == 200
        assert resp.headers.get("x-fl-mode") == "simulation"
        assert resp.json()["mode"] == "simulation"

    @pytest.mark.asyncio
    async def test_fl_rounds_has_mode_header(self, client):
        resp = await client.get("/fl/rounds")
        assert resp.status_code == 200
        assert resp.headers.get("x-fl-mode") == "simulation"
        assert resp.json()["mode"] == "simulation"

    @pytest.mark.asyncio
    async def test_fl_round_details_has_mode(self, client):
        # First create a simulation
        resp = await client.post("/fl/start", json={"n_clients": 2, "n_rounds": 1, "seed": 42})
        assert resp.status_code == 201
        sim_id = resp.json()["simulation_id"]

        # Get round details
        resp = await client.get(f"/fl/rounds/{sim_id}/1")
        assert resp.status_code == 200
        assert resp.headers.get("x-fl-mode") == "simulation"
        assert resp.json()["mode"] == "simulation"

    @pytest.mark.asyncio
    async def test_fl_round_security_has_mode(self, client):
        resp = await client.post("/fl/start", json={"n_clients": 2, "n_rounds": 1, "seed": 42})
        assert resp.status_code == 201
        sim_id = resp.json()["simulation_id"]

        resp = await client.get(f"/fl/rounds/{sim_id}/1/security")
        assert resp.status_code == 200
        assert resp.headers.get("x-fl-mode") == "simulation"
        assert resp.json()["mode"] == "simulation"
