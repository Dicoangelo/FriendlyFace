"""OSCAL compliance export for auditor consumption.

Generates NIST OSCAL-compatible assessment results from FriendlyFace
compliance data (consent, bias audits, chain integrity, bundle verification).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from friendlyface.storage.database import Database


class OSCALExporter:
    """Export compliance data in OSCAL or JSON-LD format."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def export_oscal(self) -> dict[str, Any]:
        """Generate OSCAL assessment results document."""
        consent_stats = await self.db.get_consent_coverage_stats()
        bias_stats = await self.db.get_bias_audit_stats()
        explanation_stats = await self.db.get_explanation_coverage_stats()
        bundle_stats = await self.db.get_bundle_integrity_stats()

        now = datetime.now(timezone.utc).isoformat()

        return {
            "assessment-results": {
                "uuid": str(uuid4()),
                "metadata": {
                    "title": "FriendlyFace Compliance Assessment",
                    "last-modified": now,
                    "version": "1.0.0",
                    "oscal-version": "1.0.4",
                },
                "results": [
                    {
                        "uuid": str(uuid4()),
                        "title": "Automated Compliance Check",
                        "start": now,
                        "findings": [
                            self._consent_finding(consent_stats),
                            self._bias_finding(bias_stats),
                            self._explainability_finding(explanation_stats),
                            self._integrity_finding(bundle_stats),
                        ],
                    }
                ],
            }
        }

    async def export_json_ld(self) -> dict[str, Any]:
        """Generate JSON-LD compliance document."""
        consent_stats = await self.db.get_consent_coverage_stats()
        bias_stats = await self.db.get_bias_audit_stats()
        explanation_stats = await self.db.get_explanation_coverage_stats()
        bundle_stats = await self.db.get_bundle_integrity_stats()

        now = datetime.now(timezone.utc).isoformat()

        return {
            "@context": "https://friendlyface.dev/compliance/v1",
            "@type": "ComplianceAssessment",
            "id": str(uuid4()),
            "timestamp": now,
            "platform": "FriendlyFace",
            "version": "0.1.0",
            "assessments": {
                "consent": {
                    "coverage_pct": consent_stats["coverage_pct"],
                    "total_subjects": consent_stats["total_subjects"],
                    "compliant": consent_stats["coverage_pct"] >= 80.0,
                },
                "bias_auditing": {
                    "pass_rate_pct": bias_stats["pass_rate_pct"],
                    "total_audits": bias_stats["total_audits"],
                    "compliant": bias_stats["pass_rate_pct"] >= 90.0,
                },
                "explainability": {
                    "coverage_pct": explanation_stats["coverage_pct"],
                    "compliant": explanation_stats["coverage_pct"] >= 80.0,
                },
                "integrity": {
                    "integrity_pct": bundle_stats["integrity_pct"],
                    "compliant": bundle_stats["integrity_pct"] >= 95.0,
                },
            },
        }

    def _consent_finding(self, stats: dict) -> dict:
        compliant = stats["coverage_pct"] >= 80.0
        return {
            "uuid": str(uuid4()),
            "title": "Consent Coverage (GDPR Art 6/7)",
            "target": {"type": "control", "id": "consent-management"},
            "status": {"state": "satisfied" if compliant else "not-satisfied"},
            "observations": [
                {
                    "description": f"Consent coverage: {stats['coverage_pct']}%",
                    "total_subjects": stats["total_subjects"],
                    "active_consents": stats["subjects_with_active_consent"],
                }
            ],
        }

    def _bias_finding(self, stats: dict) -> dict:
        compliant = stats["pass_rate_pct"] >= 90.0
        return {
            "uuid": str(uuid4()),
            "title": "Bias Audit Compliance (EU AI Act Art 10)",
            "target": {"type": "control", "id": "bias-monitoring"},
            "status": {"state": "satisfied" if compliant else "not-satisfied"},
            "observations": [
                {
                    "description": f"Bias audit pass rate: {stats['pass_rate_pct']}%",
                    "total_audits": stats["total_audits"],
                    "compliant_audits": stats["compliant_audits"],
                }
            ],
        }

    def _explainability_finding(self, stats: dict) -> dict:
        compliant = stats["coverage_pct"] >= 80.0
        return {
            "uuid": str(uuid4()),
            "title": "Explainability Coverage (EU AI Act Art 13)",
            "target": {"type": "control", "id": "explainability"},
            "status": {"state": "satisfied" if compliant else "not-satisfied"},
            "observations": [
                {
                    "description": f"Explanation coverage: {stats['coverage_pct']}%",
                    "total_inferences": stats["total_inferences"],
                    "total_explanations": stats["total_explanations"],
                }
            ],
        }

    def _integrity_finding(self, stats: dict) -> dict:
        compliant = stats["integrity_pct"] >= 95.0
        return {
            "uuid": str(uuid4()),
            "title": "Forensic Chain Integrity (ICDF2C)",
            "target": {"type": "control", "id": "chain-integrity"},
            "status": {"state": "satisfied" if compliant else "not-satisfied"},
            "observations": [
                {
                    "description": f"Bundle verification rate: {stats['integrity_pct']}%",
                    "total_bundles": stats["total_bundles"],
                    "verified_bundles": stats["verified_bundles"],
                }
            ],
        }
