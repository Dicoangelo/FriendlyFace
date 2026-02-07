"""Tests for OSCAL and JSON-LD compliance export (US-052)."""

from __future__ import annotations

from friendlyface.governance.oscal import OSCALExporter


# ---------------------------------------------------------------------------
# Unit tests — OSCALExporter with mock DB stats
# ---------------------------------------------------------------------------


class FakeDB:
    """Minimal mock returning configurable compliance stats."""

    def __init__(
        self,
        consent_pct=100.0,
        bias_pct=95.0,
        explanation_pct=90.0,
        integrity_pct=99.0,
    ):
        self._consent = {
            "total_subjects": 100,
            "subjects_with_active_consent": int(consent_pct),
            "coverage_pct": consent_pct,
        }
        self._bias = {
            "total_audits": 50,
            "compliant_audits": int(50 * bias_pct / 100),
            "pass_rate_pct": bias_pct,
        }
        self._explanation = {
            "total_inferences": 200,
            "total_explanations": int(200 * explanation_pct / 100),
            "coverage_pct": explanation_pct,
        }
        self._bundle = {
            "total_bundles": 80,
            "verified_bundles": int(80 * integrity_pct / 100),
            "integrity_pct": integrity_pct,
        }

    async def get_consent_coverage_stats(self):
        return self._consent

    async def get_bias_audit_stats(self):
        return self._bias

    async def get_explanation_coverage_stats(self):
        return self._explanation

    async def get_bundle_integrity_stats(self):
        return self._bundle


class TestOSCALExport:
    async def test_oscal_structure(self):
        exporter = OSCALExporter(FakeDB())
        result = await exporter.export_oscal()
        ar = result["assessment-results"]
        assert "uuid" in ar
        assert ar["metadata"]["title"] == "FriendlyFace Compliance Assessment"
        assert ar["metadata"]["oscal-version"] == "1.0.4"
        assert len(ar["results"]) == 1
        assert len(ar["results"][0]["findings"]) == 4

    async def test_oscal_findings_all_satisfied(self):
        exporter = OSCALExporter(FakeDB())
        result = await exporter.export_oscal()
        findings = result["assessment-results"]["results"][0]["findings"]
        for f in findings:
            assert f["status"]["state"] == "satisfied"

    async def test_oscal_consent_not_satisfied(self):
        exporter = OSCALExporter(FakeDB(consent_pct=50.0))
        result = await exporter.export_oscal()
        findings = result["assessment-results"]["results"][0]["findings"]
        consent = [f for f in findings if "Consent" in f["title"]][0]
        assert consent["status"]["state"] == "not-satisfied"

    async def test_oscal_bias_not_satisfied(self):
        exporter = OSCALExporter(FakeDB(bias_pct=80.0))
        result = await exporter.export_oscal()
        findings = result["assessment-results"]["results"][0]["findings"]
        bias = [f for f in findings if "Bias" in f["title"]][0]
        assert bias["status"]["state"] == "not-satisfied"

    async def test_oscal_explainability_not_satisfied(self):
        exporter = OSCALExporter(FakeDB(explanation_pct=70.0))
        result = await exporter.export_oscal()
        findings = result["assessment-results"]["results"][0]["findings"]
        exp = [f for f in findings if "Explainability" in f["title"]][0]
        assert exp["status"]["state"] == "not-satisfied"

    async def test_oscal_integrity_not_satisfied(self):
        exporter = OSCALExporter(FakeDB(integrity_pct=90.0))
        result = await exporter.export_oscal()
        findings = result["assessment-results"]["results"][0]["findings"]
        integrity = [f for f in findings if "Integrity" in f["title"]][0]
        assert integrity["status"]["state"] == "not-satisfied"

    async def test_oscal_findings_have_uuids(self):
        exporter = OSCALExporter(FakeDB())
        result = await exporter.export_oscal()
        findings = result["assessment-results"]["results"][0]["findings"]
        uuids = {f["uuid"] for f in findings}
        assert len(uuids) == 4  # all unique

    async def test_oscal_observations_content(self):
        exporter = OSCALExporter(FakeDB(consent_pct=85.0))
        result = await exporter.export_oscal()
        findings = result["assessment-results"]["results"][0]["findings"]
        consent = [f for f in findings if "Consent" in f["title"]][0]
        obs = consent["observations"][0]
        assert "85.0%" in obs["description"]
        assert obs["total_subjects"] == 100


class TestJSONLDExport:
    async def test_json_ld_structure(self):
        exporter = OSCALExporter(FakeDB())
        result = await exporter.export_json_ld()
        assert result["@context"] == "https://friendlyface.dev/compliance/v1"
        assert result["@type"] == "ComplianceAssessment"
        assert "id" in result
        assert result["platform"] == "FriendlyFace"
        assert "assessments" in result

    async def test_json_ld_all_compliant(self):
        exporter = OSCALExporter(FakeDB())
        result = await exporter.export_json_ld()
        a = result["assessments"]
        assert a["consent"]["compliant"] is True
        assert a["bias_auditing"]["compliant"] is True
        assert a["explainability"]["compliant"] is True
        assert a["integrity"]["compliant"] is True

    async def test_json_ld_consent_not_compliant(self):
        exporter = OSCALExporter(FakeDB(consent_pct=79.9))
        result = await exporter.export_json_ld()
        assert result["assessments"]["consent"]["compliant"] is False

    async def test_json_ld_bias_not_compliant(self):
        exporter = OSCALExporter(FakeDB(bias_pct=89.9))
        result = await exporter.export_json_ld()
        assert result["assessments"]["bias_auditing"]["compliant"] is False

    async def test_json_ld_values(self):
        exporter = OSCALExporter(FakeDB(consent_pct=85.0, bias_pct=92.0))
        result = await exporter.export_json_ld()
        assert result["assessments"]["consent"]["coverage_pct"] == 85.0
        assert result["assessments"]["bias_auditing"]["pass_rate_pct"] == 92.0


# ---------------------------------------------------------------------------
# API endpoint tests (integration)
# ---------------------------------------------------------------------------


class TestOSCALEndpoint:
    async def test_export_oscal_default(self, client):
        resp = await client.get("/governance/compliance/export")
        assert resp.status_code == 200
        data = resp.json()
        assert "assessment-results" in data

    async def test_export_oscal_explicit(self, client):
        resp = await client.get("/governance/compliance/export?format=oscal")
        assert resp.status_code == 200
        assert "assessment-results" in resp.json()

    async def test_export_json_ld(self, client):
        resp = await client.get("/governance/compliance/export?format=json-ld")
        assert resp.status_code == 200
        data = resp.json()
        assert data["@type"] == "ComplianceAssessment"

    async def test_v1_export_oscal(self, client):
        resp = await client.get("/api/v1/governance/compliance/export")
        assert resp.status_code == 200
        assert "assessment-results" in resp.json()
