"""Governance compliance reporting for the FriendlyFace platform.

Generates structured compliance reports covering EU AI Act requirements:
  - Article 5: Prohibited practices check
  - Article 14: Human oversight requirements

Each report generation is logged as a ForensicEvent(COMPLIANCE_REPORT).

Metrics included:
  - Consent coverage %
  - Bias audit pass rate %
  - Explanation coverage %
  - Bundle integrity %
  - Overall compliance score (weighted average)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from friendlyface.core.models import EventType
from friendlyface.core.service import ForensicService
from friendlyface.storage.database import Database


class ComplianceReporter:
    """Generates EU AI Act compliance reports with forensic event logging.

    Reports cover:
      - Article 5 (prohibited practices): checks consent coverage and
        bias audit pass rate to ensure no prohibited discriminatory practices.
      - Article 14 (human oversight): checks explanation coverage and
        bundle integrity to ensure human-reviewable audit trails.

    Each report is logged as a COMPLIANCE_REPORT forensic event.
    """

    # Weights for overall compliance score
    CONSENT_WEIGHT = 0.30
    BIAS_AUDIT_WEIGHT = 0.25
    EXPLANATION_WEIGHT = 0.25
    BUNDLE_INTEGRITY_WEIGHT = 0.20

    def __init__(
        self,
        db: Database,
        forensic_service: ForensicService,
        *,
        actor: str = "compliance_reporter",
    ) -> None:
        self.db = db
        self.forensic_service = forensic_service
        self.actor = actor

    async def generate_report(self) -> dict[str, Any]:
        """Generate a full compliance report.

        Returns a structured JSON-serializable dict with:
          - report_id: unique identifier
          - generated_at: ISO timestamp
          - article_5: prohibited practices assessment
          - article_14: human oversight assessment
          - metrics: consent_coverage_pct, bias_audit_pass_rate_pct,
                     explanation_coverage_pct, bundle_integrity_pct
          - overall_compliance_score: weighted average in [0, 100]
          - compliant: bool (True if score >= 70)
          - event_id: forensic event ID for this report generation
        """
        # Gather all metrics from the database
        consent_stats = await self.db.get_consent_coverage_stats()
        bias_stats = await self.db.get_bias_audit_stats()
        explanation_stats = await self.db.get_explanation_coverage_stats()
        bundle_stats = await self.db.get_bundle_integrity_stats()

        # Extract percentages
        consent_pct = consent_stats["coverage_pct"]
        bias_pct = bias_stats["pass_rate_pct"]
        explanation_pct = explanation_stats["coverage_pct"]
        bundle_pct = bundle_stats["integrity_pct"]

        # Compute overall compliance score (weighted average)
        overall_score = round(
            consent_pct * self.CONSENT_WEIGHT
            + bias_pct * self.BIAS_AUDIT_WEIGHT
            + explanation_pct * self.EXPLANATION_WEIGHT
            + bundle_pct * self.BUNDLE_INTEGRITY_WEIGHT,
            2,
        )

        compliant = overall_score >= 70.0

        # Article 5: Prohibited practices assessment
        article_5 = {
            "title": "Article 5 \u2014 Prohibited Practices",
            "description": (
                "Checks that the system does not engage in prohibited "
                "discriminatory practices. Requires valid consent and "
                "bias audits passing fairness thresholds."
            ),
            "consent_coverage": consent_stats,
            "bias_audit": bias_stats,
            "status": "pass" if (consent_pct >= 70.0 and bias_pct >= 70.0) else "fail",
        }

        # Article 14: Human oversight requirements
        article_14 = {
            "title": "Article 14 \u2014 Human Oversight",
            "description": (
                "Checks that the system provides adequate transparency "
                "for human oversight. Requires explanations for inferences "
                "and verified forensic bundles."
            ),
            "explanation_coverage": explanation_stats,
            "bundle_integrity": bundle_stats,
            "status": ("pass" if (explanation_pct >= 70.0 and bundle_pct >= 70.0) else "fail"),
        }

        report_id = str(uuid4())
        generated_at = datetime.now(timezone.utc).isoformat()

        # Build the report
        report: dict[str, Any] = {
            "report_id": report_id,
            "generated_at": generated_at,
            "article_5": article_5,
            "article_14": article_14,
            "metrics": {
                "consent_coverage_pct": consent_pct,
                "bias_audit_pass_rate_pct": bias_pct,
                "explanation_coverage_pct": explanation_pct,
                "bundle_integrity_pct": bundle_pct,
            },
            "overall_compliance_score": overall_score,
            "compliant": compliant,
        }

        # Log this report generation as a forensic event
        event = await self.forensic_service.record_event(
            event_type=EventType.COMPLIANCE_REPORT,
            actor=self.actor,
            payload={
                "report_id": report_id,
                "overall_compliance_score": overall_score,
                "compliant": compliant,
                "metrics": report["metrics"],
            },
        )

        report["event_id"] = str(event.id)

        return report
