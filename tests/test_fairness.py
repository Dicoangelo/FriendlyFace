"""Tests for the bias audit engine (friendlyface.fairness.auditor)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from friendlyface.core.models import BiasAuditRecord, EventType
from friendlyface.core.service import ForensicService
from friendlyface.fairness.auditor import (
    BiasAuditor,
    FairnessThresholds,
    GroupMetrics,
    GroupResult,
    check_alerts,
    compute_demographic_parity_gap,
    compute_equalized_odds_gap,
    compute_group_metrics,
    compute_overall_fairness_score,
)
from friendlyface.storage.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(tmp_path / "fairness_test.db")
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def service(db):
    svc = ForensicService(db)
    await svc.initialize()
    return svc


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_unbiased_groups() -> list[GroupResult]:
    """Two groups with identical metrics -> zero gaps."""
    return [
        GroupResult(
            group_name="group_a",
            true_positives=80,
            false_positives=10,
            true_negatives=90,
            false_negatives=20,
        ),
        GroupResult(
            group_name="group_b",
            true_positives=80,
            false_positives=10,
            true_negatives=90,
            false_negatives=20,
        ),
    ]


def make_biased_groups() -> list[GroupResult]:
    """Two groups with significantly different metrics -> large gaps."""
    return [
        GroupResult(
            group_name="group_a",
            true_positives=90,
            false_positives=5,
            true_negatives=95,
            false_negatives=10,
        ),
        GroupResult(
            group_name="group_b",
            true_positives=40,
            false_positives=30,
            true_negatives=70,
            false_negatives=60,
        ),
    ]


def make_three_groups_mixed() -> list[GroupResult]:
    """Three groups with varying levels of bias."""
    return [
        GroupResult(
            group_name="group_a",
            true_positives=80,
            false_positives=10,
            true_negatives=90,
            false_negatives=20,
        ),
        GroupResult(
            group_name="group_b",
            true_positives=75,
            false_positives=15,
            true_negatives=85,
            false_negatives=25,
        ),
        GroupResult(
            group_name="group_c",
            true_positives=50,
            false_positives=40,
            true_negatives=60,
            false_negatives=50,
        ),
    ]


# ---------------------------------------------------------------------------
# GroupResult property tests
# ---------------------------------------------------------------------------


class TestGroupResult:
    def test_total(self):
        g = GroupResult(
            "test", true_positives=10, false_positives=5, true_negatives=80, false_negatives=5
        )
        assert g.total == 100

    def test_positive_prediction_rate(self):
        g = GroupResult(
            "test", true_positives=10, false_positives=10, true_negatives=70, false_negatives=10
        )
        # 20 positive predictions out of 100 total = 0.2
        assert g.positive_prediction_rate == pytest.approx(0.2)

    def test_true_positive_rate(self):
        g = GroupResult(
            "test", true_positives=80, false_positives=10, true_negatives=90, false_negatives=20
        )
        # TPR = 80 / (80 + 20) = 0.8
        assert g.true_positive_rate == pytest.approx(0.8)

    def test_false_positive_rate(self):
        g = GroupResult(
            "test", true_positives=80, false_positives=10, true_negatives=90, false_negatives=20
        )
        # FPR = 10 / (10 + 90) = 0.1
        assert g.false_positive_rate == pytest.approx(0.1)

    def test_zero_actual_positives_tpr(self):
        g = GroupResult(
            "test", true_positives=0, false_positives=5, true_negatives=95, false_negatives=0
        )
        assert g.true_positive_rate == 0.0

    def test_zero_actual_negatives_fpr(self):
        g = GroupResult(
            "test", true_positives=50, false_positives=0, true_negatives=0, false_negatives=50
        )
        assert g.false_positive_rate == 0.0

    def test_empty_group(self):
        g = GroupResult(
            "empty", true_positives=0, false_positives=0, true_negatives=0, false_negatives=0
        )
        assert g.total == 0
        assert g.positive_prediction_rate == 0.0
        assert g.true_positive_rate == 0.0
        assert g.false_positive_rate == 0.0


# ---------------------------------------------------------------------------
# Pure metric computation tests
# ---------------------------------------------------------------------------


class TestComputeGroupMetrics:
    def test_basic(self):
        g = GroupResult(
            "test", true_positives=80, false_positives=10, true_negatives=90, false_negatives=20
        )
        m = compute_group_metrics(g)
        assert m.group_name == "test"
        assert m.sample_count == 200
        assert m.positive_prediction_rate == pytest.approx(0.45)
        assert m.true_positive_rate == pytest.approx(0.8)
        assert m.false_positive_rate == pytest.approx(0.1)


class TestDemographicParityGap:
    def test_identical_groups(self):
        metrics = [
            GroupMetrics("a", 0.5, 0.8, 0.1, 100),
            GroupMetrics("b", 0.5, 0.9, 0.2, 100),
        ]
        assert compute_demographic_parity_gap(metrics) == pytest.approx(0.0)

    def test_different_groups(self):
        metrics = [
            GroupMetrics("a", 0.3, 0.8, 0.1, 100),
            GroupMetrics("b", 0.7, 0.9, 0.2, 100),
        ]
        assert compute_demographic_parity_gap(metrics) == pytest.approx(0.4)

    def test_single_group_returns_zero(self):
        metrics = [GroupMetrics("a", 0.5, 0.8, 0.1, 100)]
        assert compute_demographic_parity_gap(metrics) == 0.0

    def test_three_groups(self):
        metrics = [
            GroupMetrics("a", 0.3, 0.8, 0.1, 100),
            GroupMetrics("b", 0.5, 0.9, 0.2, 100),
            GroupMetrics("c", 0.8, 0.7, 0.3, 100),
        ]
        assert compute_demographic_parity_gap(metrics) == pytest.approx(0.5)


class TestEqualizedOddsGap:
    def test_identical_groups(self):
        metrics = [
            GroupMetrics("a", 0.5, 0.8, 0.1, 100),
            GroupMetrics("b", 0.5, 0.8, 0.1, 100),
        ]
        assert compute_equalized_odds_gap(metrics) == pytest.approx(0.0)

    def test_tpr_gap_dominates(self):
        metrics = [
            GroupMetrics("a", 0.5, 0.9, 0.1, 100),
            GroupMetrics("b", 0.5, 0.5, 0.1, 100),
        ]
        # TPR gap = 0.4, FPR gap = 0.0, max = 0.4
        assert compute_equalized_odds_gap(metrics) == pytest.approx(0.4)

    def test_fpr_gap_dominates(self):
        metrics = [
            GroupMetrics("a", 0.5, 0.8, 0.05, 100),
            GroupMetrics("b", 0.5, 0.8, 0.35, 100),
        ]
        # TPR gap = 0.0, FPR gap = 0.3, max = 0.3
        assert compute_equalized_odds_gap(metrics) == pytest.approx(0.3)

    def test_single_group_returns_zero(self):
        metrics = [GroupMetrics("a", 0.5, 0.8, 0.1, 100)]
        assert compute_equalized_odds_gap(metrics) == 0.0


class TestOverallFairnessScore:
    def test_perfect_fairness(self):
        score = compute_overall_fairness_score(0.0, 0.0, FairnessThresholds(0.1, 0.1))
        assert score == pytest.approx(1.0)

    def test_at_threshold(self):
        score = compute_overall_fairness_score(0.1, 0.1, FairnessThresholds(0.1, 0.1))
        # Each metric scores 0.5, average = 0.5
        assert score == pytest.approx(0.5)

    def test_at_double_threshold(self):
        score = compute_overall_fairness_score(0.2, 0.2, FairnessThresholds(0.1, 0.1))
        assert score == pytest.approx(0.0)

    def test_mixed_gaps(self):
        score = compute_overall_fairness_score(0.0, 0.1, FairnessThresholds(0.1, 0.1))
        # DP: 1.0, EO: 0.5, average = 0.75
        assert score == pytest.approx(0.75)

    def test_beyond_double_threshold_clamps_to_zero(self):
        score = compute_overall_fairness_score(0.5, 0.5, FairnessThresholds(0.1, 0.1))
        assert score == pytest.approx(0.0)


class TestCheckAlerts:
    def test_no_alerts_when_compliant(self):
        alerts = check_alerts(0.05, 0.05, FairnessThresholds(0.1, 0.1), ["a", "b"])
        assert alerts == []

    def test_dp_alert(self):
        alerts = check_alerts(0.15, 0.05, FairnessThresholds(0.1, 0.1), ["a", "b"])
        assert len(alerts) == 1
        assert alerts[0].metric == "demographic_parity"
        assert alerts[0].gap == 0.15
        assert "a" in alerts[0].groups
        assert "b" in alerts[0].groups

    def test_eo_alert(self):
        alerts = check_alerts(0.05, 0.15, FairnessThresholds(0.1, 0.1), ["a", "b"])
        assert len(alerts) == 1
        assert alerts[0].metric == "equalized_odds"

    def test_both_alerts(self):
        alerts = check_alerts(0.15, 0.15, FairnessThresholds(0.1, 0.1), ["a", "b"])
        assert len(alerts) == 2
        metrics = {a.metric for a in alerts}
        assert "demographic_parity" in metrics
        assert "equalized_odds" in metrics

    def test_at_threshold_no_alert(self):
        """Exactly at threshold should NOT trigger (strict >)."""
        alerts = check_alerts(0.1, 0.1, FairnessThresholds(0.1, 0.1), ["a", "b"])
        assert alerts == []


# ---------------------------------------------------------------------------
# BiasAuditor integration tests (async, with ForensicService)
# ---------------------------------------------------------------------------


class TestBiasAuditor:
    @pytest.mark.asyncio
    async def test_unbiased_audit(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()
        record, alerts = await auditor.audit(groups)

        assert isinstance(record, BiasAuditRecord)
        assert record.demographic_parity_gap == pytest.approx(0.0)
        assert record.equalized_odds_gap == pytest.approx(0.0)
        assert record.compliant is True
        assert len(alerts) == 0
        assert record.event_id is not None
        assert "group_a" in record.groups_evaluated
        assert "group_b" in record.groups_evaluated

    @pytest.mark.asyncio
    async def test_biased_audit_triggers_alerts(self, service):
        auditor = BiasAuditor(service, thresholds=FairnessThresholds(0.1, 0.1))
        groups = make_biased_groups()
        record, alerts = await auditor.audit(groups)

        assert record.compliant is False
        assert len(alerts) > 0
        assert record.demographic_parity_gap > 0.1
        assert record.equalized_odds_gap > 0.1

    @pytest.mark.asyncio
    async def test_forensic_event_logged(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()
        record, _ = await auditor.audit(groups)

        # Check that the event was recorded in the forensic chain
        event = await service.get_event(record.event_id)
        assert event is not None
        assert event.event_type == EventType.BIAS_AUDIT
        assert event.actor == "bias_auditor"
        assert "demographic_parity_gap" in event.payload
        assert "equalized_odds_gap" in event.payload
        assert event.verify()

    @pytest.mark.asyncio
    async def test_alert_events_logged(self, service):
        auditor = BiasAuditor(service, thresholds=FairnessThresholds(0.01, 0.01))
        groups = make_biased_groups()
        await auditor.audit(groups)

        # Should have at least 3 events: 1 BIAS_AUDIT + 2 SECURITY_ALERTs
        events = await service.get_all_events()
        bias_events = [e for e in events if e.event_type == EventType.BIAS_AUDIT]
        alert_events = [e for e in events if e.event_type == EventType.SECURITY_ALERT]

        assert len(bias_events) >= 1
        assert len(alert_events) >= 1
        for ae in alert_events:
            assert ae.payload["alert_type"] == "bias_threshold_breach"

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, service):
        # Very strict thresholds
        strict = FairnessThresholds(0.001, 0.001)
        auditor = BiasAuditor(service, thresholds=strict)
        groups = make_unbiased_groups()
        record, alerts = await auditor.audit(groups)

        # Unbiased groups have 0 gap, so even strict thresholds pass
        assert record.compliant is True
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_custom_actor(self, service):
        auditor = BiasAuditor(service, actor="custom_auditor")
        groups = make_unbiased_groups()
        record, _ = await auditor.audit(groups)

        event = await service.get_event(record.event_id)
        assert event.actor == "custom_auditor"

    @pytest.mark.asyncio
    async def test_metadata_included(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()
        meta = {"model_version": "v2.1", "dataset": "lfw"}
        record, _ = await auditor.audit(groups, metadata=meta)

        assert record.details["metadata"] == meta
        event = await service.get_event(record.event_id)
        assert event.payload["metadata"] == meta

    @pytest.mark.asyncio
    async def test_three_groups(self, service):
        auditor = BiasAuditor(service)
        groups = make_three_groups_mixed()
        record, alerts = await auditor.audit(groups)

        assert len(record.groups_evaluated) == 3
        assert record.details["per_group_metrics"] is not None
        assert len(record.details["per_group_metrics"]) == 3

    @pytest.mark.asyncio
    async def test_fewer_than_two_groups_raises(self, service):
        auditor = BiasAuditor(service)
        with pytest.raises(ValueError, match="At least 2"):
            await auditor.audit([GroupResult("only", 10, 5, 85, 10)])

    @pytest.mark.asyncio
    async def test_details_contain_fairness_score(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()
        record, _ = await auditor.audit(groups)

        assert "fairness_score" in record.details
        assert 0.0 <= record.details["fairness_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_details_contain_thresholds(self, service):
        thresholds = FairnessThresholds(0.08, 0.12)
        auditor = BiasAuditor(service, thresholds=thresholds)
        groups = make_unbiased_groups()
        record, _ = await auditor.audit(groups)

        assert record.details["thresholds"]["demographic_parity"] == 0.08
        assert record.details["thresholds"]["equalized_odds"] == 0.12

    @pytest.mark.asyncio
    async def test_bias_audit_persisted_to_db(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()
        record, _ = await auditor.audit(groups)

        # Verify it was persisted (insert_bias_audit was called)
        # The record should have a valid id
        assert record.id is not None
        assert record.event_id is not None

    @pytest.mark.asyncio
    async def test_hash_chain_integrity_after_audit(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()
        await auditor.audit(groups)

        integrity = await service.verify_chain_integrity()
        assert integrity["valid"] is True

    @pytest.mark.asyncio
    async def test_biased_data_fairness_score_low(self, service):
        auditor = BiasAuditor(service, thresholds=FairnessThresholds(0.1, 0.1))
        groups = make_biased_groups()
        record, _ = await auditor.audit(groups)

        assert record.details["fairness_score"] < 0.5

    @pytest.mark.asyncio
    async def test_unbiased_data_fairness_score_high(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()
        record, _ = await auditor.audit(groups)

        assert record.details["fairness_score"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_multiple_audits_chain_correctly(self, service):
        auditor = BiasAuditor(service)
        groups = make_unbiased_groups()

        r1, _ = await auditor.audit(groups)
        r2, _ = await auditor.audit(groups)

        # Both should have different event IDs
        assert r1.event_id != r2.event_id

        # Chain should still be valid
        integrity = await service.verify_chain_integrity()
        assert integrity["valid"] is True
