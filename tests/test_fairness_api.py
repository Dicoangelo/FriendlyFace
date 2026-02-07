"""Integration tests for the fairness bias audit API endpoints (US-011).

Tests:
  POST  /fairness/audit         - manual bias audit trigger
  GET   /fairness/audits        - list completed audits
  GET   /fairness/audits/{id}   - full audit details
  GET   /fairness/status        - current fairness health
  POST  /fairness/config        - configure auto-audit interval
  GET   /fairness/config        - get auto-audit config
  Auto-trigger                  - audit fires after N recognition events
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unbiased_groups():
    """Two groups with identical metrics."""
    return [
        {
            "group_name": "group_a",
            "true_positives": 80,
            "false_positives": 10,
            "true_negatives": 90,
            "false_negatives": 20,
        },
        {
            "group_name": "group_b",
            "true_positives": 80,
            "false_positives": 10,
            "true_negatives": 90,
            "false_negatives": 20,
        },
    ]


def _biased_groups():
    """Two groups with significantly different metrics."""
    return [
        {
            "group_name": "group_a",
            "true_positives": 90,
            "false_positives": 5,
            "true_negatives": 95,
            "false_negatives": 10,
        },
        {
            "group_name": "group_b",
            "true_positives": 40,
            "false_positives": 30,
            "true_negatives": 70,
            "false_negatives": 60,
        },
    ]


# ---------------------------------------------------------------------------
# POST /fairness/audit - manual bias audit
# ---------------------------------------------------------------------------


class TestManualBiasAudit:
    async def test_trigger_unbiased_audit(self, client):
        resp = await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["compliant"] is True
        assert data["demographic_parity_gap"] == pytest.approx(0.0)
        assert data["equalized_odds_gap"] == pytest.approx(0.0)
        assert data["fairness_score"] == pytest.approx(1.0)
        assert len(data["alerts"]) == 0
        assert data["audit_id"] is not None
        assert data["event_id"] is not None
        assert "group_a" in data["groups_evaluated"]
        assert "group_b" in data["groups_evaluated"]

    async def test_trigger_biased_audit(self, client):
        resp = await client.post(
            "/fairness/audit",
            json={
                "groups": _biased_groups(),
                "demographic_parity_threshold": 0.1,
                "equalized_odds_threshold": 0.1,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["compliant"] is False
        assert data["demographic_parity_gap"] > 0.1
        assert len(data["alerts"]) > 0

    async def test_audit_with_custom_thresholds(self, client):
        resp = await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
                "demographic_parity_threshold": 0.001,
                "equalized_odds_threshold": 0.001,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["compliant"] is True

    async def test_audit_with_metadata(self, client):
        resp = await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
                "metadata": {"model_version": "v2.1"},
            },
        )
        assert resp.status_code == 201

    async def test_audit_fewer_than_two_groups_rejected(self, client):
        resp = await client.post(
            "/fairness/audit",
            json={
                "groups": [_unbiased_groups()[0]],
            },
        )
        assert resp.status_code == 400
        assert "2 demographic groups" in resp.json()["detail"]

    async def test_audit_missing_group_field_rejected(self, client):
        resp = await client.post(
            "/fairness/audit",
            json={
                "groups": [
                    {"group_name": "a", "true_positives": 10},
                    {"group_name": "b", "true_positives": 10},
                ],
            },
        )
        assert resp.status_code == 400
        assert "Missing required field" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /fairness/audits - list audits
# ---------------------------------------------------------------------------


class TestListBiasAudits:
    async def test_empty_list(self, client):
        resp = await client.get("/fairness/audits")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    async def test_list_after_audit(self, client):
        await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        resp = await client.get("/fairness/audits")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        audit = data["items"][0]
        assert "audit_id" in audit
        assert "demographic_parity_gap" in audit
        assert "equalized_odds_gap" in audit
        assert "compliant" in audit
        assert "fairness_score" in audit

    async def test_list_multiple_audits(self, client):
        await client.post("/fairness/audit", json={"groups": _unbiased_groups()})
        await client.post("/fairness/audit", json={"groups": _biased_groups()})
        resp = await client.get("/fairness/audits")
        data = resp.json()
        assert data["total"] >= 2


# ---------------------------------------------------------------------------
# GET /fairness/audits/{id} - single audit detail
# ---------------------------------------------------------------------------


class TestGetBiasAudit:
    async def test_get_audit_by_id(self, client):
        create_resp = await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        audit_id = create_resp.json()["audit_id"]

        resp = await client.get(f"/fairness/audits/{audit_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == audit_id
        assert "demographic_parity_gap" in data
        assert "equalized_odds_gap" in data
        assert "details" in data
        assert "per_group_metrics" in data["details"]

    async def test_get_audit_not_found(self, client):
        resp = await client.get("/fairness/audits/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    async def test_audit_details_contain_per_group_breakdowns(self, client):
        create_resp = await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        audit_id = create_resp.json()["audit_id"]

        resp = await client.get(f"/fairness/audits/{audit_id}")
        data = resp.json()
        per_group = data["details"]["per_group_metrics"]
        assert len(per_group) == 2
        for group in per_group:
            assert "group" in group
            assert "positive_prediction_rate" in group
            assert "true_positive_rate" in group
            assert "false_positive_rate" in group
            assert "sample_count" in group


# ---------------------------------------------------------------------------
# GET /fairness/status - fairness health
# ---------------------------------------------------------------------------


class TestFairnessStatus:
    async def test_status_unknown_when_no_audits(self, client):
        resp = await client.get("/fairness/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unknown"
        assert data["total_audits"] == 0

    async def test_status_pass_after_compliant_audit(self, client):
        await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        resp = await client.get("/fairness/status")
        data = resp.json()
        assert data["status"] == "pass"
        assert data["compliant"] is True
        assert data["fairness_score"] >= 0.7

    async def test_status_fail_after_non_compliant_audit(self, client):
        await client.post(
            "/fairness/audit",
            json={
                "groups": _biased_groups(),
                "demographic_parity_threshold": 0.01,
                "equalized_odds_threshold": 0.01,
            },
        )
        resp = await client.get("/fairness/status")
        data = resp.json()
        assert data["status"] == "fail"
        assert data["compliant"] is False

    async def test_status_reflects_latest_audit(self, client):
        await client.post(
            "/fairness/audit",
            json={
                "groups": _biased_groups(),
                "demographic_parity_threshold": 0.01,
                "equalized_odds_threshold": 0.01,
            },
        )
        await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        resp = await client.get("/fairness/status")
        data = resp.json()
        assert data["status"] == "pass"
        assert data["total_audits"] >= 2

    async def test_status_includes_summary_fields(self, client):
        await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        resp = await client.get("/fairness/status")
        data = resp.json()
        assert "fairness_score" in data
        assert "latest_audit_id" in data
        assert "total_audits" in data
        assert "compliant_audits" in data
        assert "demographic_parity_gap" in data
        assert "equalized_odds_gap" in data


# ---------------------------------------------------------------------------
# POST /fairness/config + GET /fairness/config
# ---------------------------------------------------------------------------


class TestAutoAuditConfig:
    async def test_get_default_config(self, client):
        resp = await client.get("/fairness/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "auto_audit_interval" in data

    async def test_set_config(self, client):
        resp = await client.post(
            "/fairness/config",
            json={
                "auto_audit_interval": 10,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["auto_audit_interval"] == 10

        resp2 = await client.get("/fairness/config")
        assert resp2.json()["auto_audit_interval"] == 10


# ---------------------------------------------------------------------------
# Auto-trigger
# ---------------------------------------------------------------------------


class TestAutoAuditTrigger:
    async def test_auto_audit_triggers_after_n_events(self, client):
        import friendlyface.api.app as app_module

        app_module._auto_audit_interval = 3
        app_module._recognition_event_count = 0

        resp = await client.get("/fairness/audits")
        initial_count = resp.json()["total"]

        app_module._recognition_event_count = 2
        await app_module._maybe_auto_audit()

        resp = await client.get("/fairness/audits")
        new_count = resp.json()["total"]
        assert new_count > initial_count

        audits = resp.json()["items"]
        found_auto = False
        for audit in audits:
            detail_resp = await client.get(f"/fairness/audits/{audit['audit_id']}")
            details = detail_resp.json().get("details", {})
            meta = details.get("metadata", {})
            if meta.get("trigger") == "auto":
                found_auto = True
                break
        assert found_auto, "Expected an auto-triggered audit"

    async def test_counter_resets_after_auto_audit(self, client):
        import friendlyface.api.app as app_module

        app_module._auto_audit_interval = 2
        app_module._recognition_event_count = 1

        await app_module._maybe_auto_audit()

        assert app_module._recognition_event_count == 0

    async def test_no_auto_audit_below_threshold(self, client):
        import friendlyface.api.app as app_module

        app_module._auto_audit_interval = 100
        app_module._recognition_event_count = 0

        initial_resp = await client.get("/fairness/audits")
        initial_count = initial_resp.json()["total"]

        for _ in range(5):
            await app_module._maybe_auto_audit()

        resp = await client.get("/fairness/audits")
        assert resp.json()["total"] == initial_count


# ---------------------------------------------------------------------------
# Chain integrity after fairness operations
# ---------------------------------------------------------------------------


class TestFairnessChainIntegrity:
    async def test_chain_valid_after_manual_audit(self, client):
        await client.post(
            "/fairness/audit",
            json={
                "groups": _unbiased_groups(),
            },
        )
        resp = await client.get("/chain/integrity")
        assert resp.json()["valid"] is True

    async def test_chain_valid_after_biased_audit_with_alerts(self, client):
        await client.post(
            "/fairness/audit",
            json={
                "groups": _biased_groups(),
                "demographic_parity_threshold": 0.01,
                "equalized_odds_threshold": 0.01,
            },
        )
        resp = await client.get("/chain/integrity")
        assert resp.json()["valid"] is True
