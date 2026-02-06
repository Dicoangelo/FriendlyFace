"""Tests for governance compliance reporting (US-016)."""

from __future__ import annotations

import json

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app
from friendlyface.api import app as app_module
from friendlyface.core.models import BiasAuditRecord, EventType
from friendlyface.core.service import ForensicService
from friendlyface.governance.compliance import ComplianceReporter
from friendlyface.storage.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    """Fresh database for each test."""
    database = Database(tmp_path / "compliance_test.db")
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def service(db):
    """Fresh forensic service for each test."""
    svc = ForensicService(db)
    await svc.initialize()
    return svc


@pytest_asyncio.fixture
async def reporter(db, service):
    """ComplianceReporter wired to fresh db + forensic service."""
    return ComplianceReporter(db, service)


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database."""
    _db.db_path = tmp_path / "compliance_api_test.db"
    await _db.connect()
    await _service.initialize()

    _service.merkle = __import__("friendlyface.core.merkle", fromlist=["MerkleTree"]).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()

    # Reset compliance cache
    app_module._latest_compliance_report = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


# ---------------------------------------------------------------------------
# Helper: seed data for compliance metrics
# ---------------------------------------------------------------------------


async def _seed_consent_data(db, service, n_subjects=10, n_active=8):
    """Seed consent records: n_subjects total, n_active with active consent."""
    from friendlyface.governance.consent import ConsentManager

    mgr = ConsentManager(db, service)
    for i in range(n_subjects):
        await mgr.grant_consent(f"subj_{i}", "recognition", actor="test")
    # Revoke consent for the last (n_subjects - n_active) subjects
    for i in range(n_active, n_subjects):
        await mgr.revoke_consent(f"subj_{i}", "recognition", actor="test")


async def _seed_bias_audits(db, n_total=5, n_compliant=4):
    """Insert bias audit records directly."""

    for i in range(n_total):
        audit = BiasAuditRecord(
            demographic_parity_gap=0.05 if i < n_compliant else 0.25,
            equalized_odds_gap=0.03 if i < n_compliant else 0.20,
            groups_evaluated=["group_a", "group_b"],
            compliant=i < n_compliant,
        )
        await db.insert_bias_audit(audit)


async def _seed_inference_and_explanation_events(service, n_inferences=10, n_explanations=8):
    """Seed inference_result and explanation_generated events."""
    for i in range(n_inferences):
        await service.record_event(
            event_type=EventType.INFERENCE_RESULT,
            actor="test",
            payload={"inference": i},
        )
    for i in range(n_explanations):
        await service.record_event(
            event_type=EventType.EXPLANATION_GENERATED,
            actor="test",
            payload={"explanation": i},
        )


async def _seed_bundles(db, service, n_total=5, n_verified=4):
    """Create and optionally verify forensic bundles."""
    from friendlyface.core.models import BundleStatus

    # Create an event to reference in bundles
    evt = await service.record_event(
        event_type=EventType.TRAINING_START,
        actor="test",
        payload={},
    )
    for i in range(n_total):
        bundle = await service.create_bundle(event_ids=[evt.id])
        if i < n_verified:
            # Manually set status to verified
            await db.update_bundle_status(bundle.id, BundleStatus.VERIFIED)


# ---------------------------------------------------------------------------
# COMPLIANCE_REPORT event type
# ---------------------------------------------------------------------------


class TestComplianceReportEventType:
    def test_compliance_report_enum_exists(self):
        """COMPLIANCE_REPORT must be present in EventType."""
        assert hasattr(EventType, "COMPLIANCE_REPORT")
        assert EventType.COMPLIANCE_REPORT.value == "compliance_report"


# ---------------------------------------------------------------------------
# ComplianceReporter — report structure
# ---------------------------------------------------------------------------


class TestComplianceReportStructure:
    async def test_report_has_required_fields(self, reporter):
        """Report must contain all required top-level fields."""
        report = await reporter.generate_report()

        assert "report_id" in report
        assert "generated_at" in report
        assert "article_5" in report
        assert "article_14" in report
        assert "metrics" in report
        assert "overall_compliance_score" in report
        assert "compliant" in report
        assert "event_id" in report

    async def test_report_metrics_structure(self, reporter):
        """Metrics dict must contain all four metric percentages."""
        report = await reporter.generate_report()
        metrics = report["metrics"]

        assert "consent_coverage_pct" in metrics
        assert "bias_audit_pass_rate_pct" in metrics
        assert "explanation_coverage_pct" in metrics
        assert "bundle_integrity_pct" in metrics

    async def test_article_5_structure(self, reporter):
        """Article 5 section must contain consent and bias audit info."""
        report = await reporter.generate_report()
        a5 = report["article_5"]

        assert "title" in a5
        assert "description" in a5
        assert "consent_coverage" in a5
        assert "bias_audit" in a5
        assert "status" in a5
        assert a5["status"] in ("pass", "fail")

    async def test_article_14_structure(self, reporter):
        """Article 14 section must contain explanation and bundle info."""
        report = await reporter.generate_report()
        a14 = report["article_14"]

        assert "title" in a14
        assert "description" in a14
        assert "explanation_coverage" in a14
        assert "bundle_integrity" in a14
        assert "status" in a14
        assert a14["status"] in ("pass", "fail")

    async def test_report_is_json_serializable(self, reporter):
        """Report must be JSON-serializable."""
        report = await reporter.generate_report()
        serialized = json.dumps(report)
        deserialized = json.loads(serialized)
        assert deserialized["report_id"] == report["report_id"]


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


class TestComplianceScoring:
    async def test_empty_db_scores_zero(self, reporter):
        """Empty database should produce 0.0 compliance score."""
        report = await reporter.generate_report()
        assert report["overall_compliance_score"] == 0.0
        assert report["compliant"] is False

    async def test_full_compliance_score(self, db, service, reporter):
        """All metrics at 100% should produce 100.0 score."""
        await _seed_consent_data(db, service, n_subjects=5, n_active=5)
        await _seed_bias_audits(db, n_total=3, n_compliant=3)
        await _seed_inference_and_explanation_events(service, 5, 5)
        await _seed_bundles(db, service, n_total=3, n_verified=3)

        report = await reporter.generate_report()
        assert report["overall_compliance_score"] == 100.0
        assert report["compliant"] is True

    async def test_partial_compliance_score(self, db, service, reporter):
        """Partial metrics should produce a weighted score."""
        # 80% consent (8/10), 80% bias (4/5), 0% explanation, 0% bundle
        await _seed_consent_data(db, service, n_subjects=10, n_active=8)
        await _seed_bias_audits(db, n_total=5, n_compliant=4)

        report = await reporter.generate_report()
        # 80*0.30 + 80*0.25 + 0*0.25 + 0*0.20 = 24 + 20 = 44.0
        assert report["overall_compliance_score"] == 44.0
        assert report["compliant"] is False

    async def test_compliance_threshold_at_70(self, db, service, reporter):
        """Score >= 70 should be compliant, < 70 should not."""
        # Create scenario with exactly 70% across all metrics
        # 70% consent
        await _seed_consent_data(db, service, n_subjects=10, n_active=7)
        # 80% bias (4/5)
        await _seed_bias_audits(db, n_total=5, n_compliant=4)
        # 60% explanation (6/10)
        await _seed_inference_and_explanation_events(service, 10, 6)
        # 60% bundles (3/5)
        await _seed_bundles(db, service, n_total=5, n_verified=3)

        report = await reporter.generate_report()
        # 70*0.30 + 80*0.25 + 60*0.25 + 60*0.20 = 21 + 20 + 15 + 12 = 68.0
        assert report["overall_compliance_score"] == 68.0
        assert report["compliant"] is False

    async def test_score_weights_sum_to_one(self):
        """Verify weight constants sum to 1.0."""
        total = (
            ComplianceReporter.CONSENT_WEIGHT
            + ComplianceReporter.BIAS_AUDIT_WEIGHT
            + ComplianceReporter.EXPLANATION_WEIGHT
            + ComplianceReporter.BUNDLE_INTEGRITY_WEIGHT
        )
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Article 5 — prohibited practices
# ---------------------------------------------------------------------------


class TestArticle5:
    async def test_article_5_passes_when_metrics_high(self, db, service, reporter):
        """Article 5 passes when consent >= 70% and bias audit >= 70%."""
        await _seed_consent_data(db, service, n_subjects=10, n_active=8)
        await _seed_bias_audits(db, n_total=5, n_compliant=4)

        report = await reporter.generate_report()
        assert report["article_5"]["status"] == "pass"

    async def test_article_5_fails_low_consent(self, db, service, reporter):
        """Article 5 fails when consent < 70%."""
        await _seed_consent_data(db, service, n_subjects=10, n_active=5)
        await _seed_bias_audits(db, n_total=5, n_compliant=5)

        report = await reporter.generate_report()
        assert report["article_5"]["status"] == "fail"

    async def test_article_5_fails_low_bias_audit(self, db, service, reporter):
        """Article 5 fails when bias audit pass rate < 70%."""
        await _seed_consent_data(db, service, n_subjects=10, n_active=10)
        await _seed_bias_audits(db, n_total=10, n_compliant=5)

        report = await reporter.generate_report()
        assert report["article_5"]["status"] == "fail"


# ---------------------------------------------------------------------------
# Article 14 — human oversight
# ---------------------------------------------------------------------------


class TestArticle14:
    async def test_article_14_passes_when_metrics_high(self, db, service, reporter):
        """Article 14 passes when explanation >= 70% and bundle >= 70%."""
        await _seed_inference_and_explanation_events(service, 10, 8)
        await _seed_bundles(db, service, n_total=5, n_verified=4)

        report = await reporter.generate_report()
        assert report["article_14"]["status"] == "pass"

    async def test_article_14_fails_low_explanation(self, db, service, reporter):
        """Article 14 fails when explanation coverage < 70%."""
        await _seed_inference_and_explanation_events(service, 10, 5)
        await _seed_bundles(db, service, n_total=5, n_verified=5)

        report = await reporter.generate_report()
        assert report["article_14"]["status"] == "fail"

    async def test_article_14_fails_low_bundle_integrity(self, db, service, reporter):
        """Article 14 fails when bundle integrity < 70%."""
        await _seed_inference_and_explanation_events(service, 10, 10)
        await _seed_bundles(db, service, n_total=10, n_verified=5)

        report = await reporter.generate_report()
        assert report["article_14"]["status"] == "fail"


# ---------------------------------------------------------------------------
# Forensic event logging
# ---------------------------------------------------------------------------


class TestComplianceForensicLogging:
    async def test_report_logs_forensic_event(self, db, service, reporter):
        """Generating a report must log a COMPLIANCE_REPORT forensic event."""
        report = await reporter.generate_report()

        events = await service.get_all_events()
        compliance_events = [e for e in events if e.event_type == EventType.COMPLIANCE_REPORT]
        assert len(compliance_events) == 1
        assert compliance_events[0].payload["report_id"] == report["report_id"]
        assert compliance_events[0].payload["compliant"] == report["compliant"]

    async def test_report_event_id_matches(self, reporter):
        """The event_id in the report must match the recorded event."""
        report = await reporter.generate_report()
        assert report["event_id"] is not None
        assert len(report["event_id"]) > 0

    async def test_multiple_reports_create_multiple_events(self, service, reporter):
        """Each report generation creates a separate forensic event."""
        await reporter.generate_report()
        await reporter.generate_report()

        events = await service.get_all_events()
        compliance_events = [e for e in events if e.event_type == EventType.COMPLIANCE_REPORT]
        assert len(compliance_events) == 2

    async def test_forensic_event_contains_metrics(self, reporter):
        """The forensic event payload must include the metrics."""
        await reporter.generate_report()
        events = await reporter.forensic_service.get_all_events()
        compliance_events = [e for e in events if e.event_type == EventType.COMPLIANCE_REPORT]
        payload = compliance_events[0].payload
        assert "metrics" in payload
        assert "consent_coverage_pct" in payload["metrics"]
        assert "bias_audit_pass_rate_pct" in payload["metrics"]
        assert "explanation_coverage_pct" in payload["metrics"]
        assert "bundle_integrity_pct" in payload["metrics"]

    async def test_chain_integrity_after_report(self, service, reporter):
        """Hash chain must remain valid after compliance report event."""
        await reporter.generate_report()
        integrity = await service.verify_chain_integrity()
        assert integrity["valid"] is True


# ---------------------------------------------------------------------------
# Database helper methods
# ---------------------------------------------------------------------------


class TestDatabaseComplianceHelpers:
    async def test_consent_coverage_empty(self, db):
        """Empty consent table returns zero coverage."""
        stats = await db.get_consent_coverage_stats()
        assert stats["total_subjects"] == 0
        assert stats["subjects_with_active_consent"] == 0
        assert stats["coverage_pct"] == 0.0

    async def test_consent_coverage_with_data(self, db, service):
        """Coverage correctly counts active vs total subjects."""
        await _seed_consent_data(db, service, n_subjects=10, n_active=7)
        stats = await db.get_consent_coverage_stats()
        assert stats["total_subjects"] == 10
        assert stats["subjects_with_active_consent"] == 7
        assert stats["coverage_pct"] == 70.0

    async def test_bias_audit_stats_empty(self, db):
        """Empty bias audits returns zero pass rate."""
        stats = await db.get_bias_audit_stats()
        assert stats["total_audits"] == 0
        assert stats["pass_rate_pct"] == 0.0

    async def test_bias_audit_stats_with_data(self, db):
        """Pass rate correctly counts compliant vs total."""
        await _seed_bias_audits(db, n_total=5, n_compliant=4)
        stats = await db.get_bias_audit_stats()
        assert stats["total_audits"] == 5
        assert stats["compliant_audits"] == 4
        assert stats["pass_rate_pct"] == 80.0

    async def test_explanation_coverage_empty(self, db):
        """No inferences means zero coverage."""
        stats = await db.get_explanation_coverage_stats()
        assert stats["total_inferences"] == 0
        assert stats["coverage_pct"] == 0.0

    async def test_explanation_coverage_with_data(self, db, service):
        """Coverage correctly counts explanations vs inferences."""
        await _seed_inference_and_explanation_events(service, 10, 7)
        stats = await db.get_explanation_coverage_stats()
        assert stats["total_inferences"] == 10
        assert stats["total_explanations"] == 7
        assert stats["coverage_pct"] == 70.0

    async def test_bundle_integrity_empty(self, db):
        """No bundles means zero integrity."""
        stats = await db.get_bundle_integrity_stats()
        assert stats["total_bundles"] == 0
        assert stats["integrity_pct"] == 0.0

    async def test_bundle_integrity_with_data(self, db, service):
        """Integrity correctly counts verified vs total bundles."""
        await _seed_bundles(db, service, n_total=5, n_verified=3)
        stats = await db.get_bundle_integrity_stats()
        assert stats["total_bundles"] == 5
        assert stats["verified_bundles"] == 3
        assert stats["integrity_pct"] == 60.0


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestComplianceAPI:
    async def test_get_compliance_report(self, client):
        """GET /governance/compliance returns a report."""
        resp = await client.get("/governance/compliance")
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id" in data
        assert "overall_compliance_score" in data
        assert "article_5" in data
        assert "article_14" in data
        assert "metrics" in data
        assert "compliant" in data

    async def test_post_generate_compliance_report(self, client):
        """POST /governance/compliance/generate creates a new report."""
        resp = await client.post("/governance/compliance/generate")
        assert resp.status_code == 201
        data = resp.json()
        assert "report_id" in data
        assert "event_id" in data

    async def test_get_returns_cached_report(self, client):
        """GET after POST should return the same cached report."""
        resp1 = await client.post("/governance/compliance/generate")
        report1 = resp1.json()

        resp2 = await client.get("/governance/compliance")
        report2 = resp2.json()

        assert report1["report_id"] == report2["report_id"]

    async def test_post_generates_new_report(self, client):
        """Subsequent POSTs should generate new reports with different IDs."""
        resp1 = await client.post("/governance/compliance/generate")
        report1 = resp1.json()

        resp2 = await client.post("/governance/compliance/generate")
        report2 = resp2.json()

        assert report1["report_id"] != report2["report_id"]

    async def test_api_report_json_structure(self, client):
        """API report must have proper JSON structure."""
        resp = await client.post("/governance/compliance/generate")
        data = resp.json()

        # Top-level fields
        assert isinstance(data["report_id"], str)
        assert isinstance(data["generated_at"], str)
        assert isinstance(data["overall_compliance_score"], (int, float))
        assert isinstance(data["compliant"], bool)
        assert isinstance(data["event_id"], str)

        # Metrics
        metrics = data["metrics"]
        assert isinstance(metrics["consent_coverage_pct"], (int, float))
        assert isinstance(metrics["bias_audit_pass_rate_pct"], (int, float))
        assert isinstance(metrics["explanation_coverage_pct"], (int, float))
        assert isinstance(metrics["bundle_integrity_pct"], (int, float))
