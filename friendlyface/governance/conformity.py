"""EU AI Act Annex IV Conformity Assessment Document Generator.

Generates technical documentation required for high-risk AI systems
under the EU AI Act (Article 11, Annex IV). Auto-populates sections
from FriendlyFace forensic data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from friendlyface.core.models import EventType
from friendlyface.core.service import ForensicService
from friendlyface.storage.database import Database


class ConformityAssessmentGenerator:
    """Generates EU AI Act Annex IV technical documentation.

    Each of the 10 Annex IV sections is auto-populated from forensic
    event data, bias audits, consent records, provenance nodes, and
    bundle integrity checks stored in the FriendlyFace database.
    """

    def __init__(
        self,
        db: Database,
        forensic_service: ForensicService,
        seal_service: Any | None = None,
    ) -> None:
        self.db = db
        self.service = forensic_service
        self.seal_service = seal_service

    async def generate(
        self,
        system_id: str = "friendlyface",
        system_name: str = "FriendlyFace",
    ) -> dict[str, Any]:
        """Generate a complete Annex IV conformity assessment document.

        Returns a structured document with 10 sections, each containing:
        - section_id: Annex IV section reference
        - title: Section title
        - status: 'complete' | 'partial' | 'missing'
        - content: Auto-populated content from forensic data
        - evidence: List of event_ids, bundle_ids, or seal_ids as evidence
        - gaps: List of identified gaps requiring action
        """
        sections = [
            await self._section_1_description(system_id, system_name),
            await self._section_2_intended_purpose(),
            await self._section_3_risk_classification(),
            await self._section_4_training_data(),
            await self._section_5_bias_testing(),
            await self._section_6_performance_metrics(),
            await self._section_7_human_oversight(),
            await self._section_8_cybersecurity(),
            await self._section_9_quality_management(),
            await self._section_10_post_market(),
        ]

        # Calculate overall completeness
        complete = sum(1 for s in sections if s["status"] == "complete")
        partial = sum(1 for s in sections if s["status"] == "partial")
        missing = sum(1 for s in sections if s["status"] == "missing")
        all_gaps: list[str] = []
        for s in sections:
            all_gaps.extend(s.get("gaps", []))

        # Collect seal references if seal_service is available
        seal_references: list[str] = []
        if self.seal_service is not None:
            try:
                seals, _ = await self.db.list_seals(
                    system_id=system_id, limit=10, offset=0
                )
                seal_references = [s["id"] for s in seals]
            except Exception:
                pass

        return {
            "document_id": str(uuid4()),
            "document_type": "eu_ai_act_annex_iv",
            "system_id": system_id,
            "system_name": system_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "regulation": "EU AI Act (Regulation 2024/1689)",
            "annex": "Annex IV \u2014 Technical Documentation",
            "risk_classification": "high-risk",
            "applicable_articles": [
                "Article 6",
                "Article 9",
                "Article 10",
                "Article 11",
                "Article 13",
                "Article 14",
                "Article 15",
            ],
            "completeness": {
                "total_sections": 10,
                "complete": complete,
                "partial": partial,
                "missing": missing,
                "score_pct": round((complete + partial * 0.5) / 10 * 100, 1),
            },
            "sections": sections,
            "gaps": all_gaps,
            "seal_references": seal_references,
            "enforcement_deadline": "2026-08-02",
        }

    # ------------------------------------------------------------------
    # Section 1: General Description of the AI System
    # ------------------------------------------------------------------

    async def _section_1_description(
        self, system_id: str, system_name: str
    ) -> dict[str, Any]:
        event_count = await self.db.get_event_count()
        bundle_count = await self.db.get_bundle_count()
        provenance_count = await self.db.get_provenance_count()
        events_by_type = await self.db.get_events_by_type()
        models = await self.db.list_models()

        evidence: list[str] = []
        gaps: list[str] = []

        content: dict[str, Any] = {
            "system_id": system_id,
            "system_name": system_name,
            "description": (
                f"{system_name} is a forensic-friendly AI facial recognition "
                "platform implementing a 6-layer architecture: Recognition, "
                "Federated Learning, Blockchain Forensic, Fairness, "
                "Explainability, and Consent & Governance."
            ),
            "total_forensic_events": event_count,
            "total_bundles": bundle_count,
            "total_provenance_nodes": provenance_count,
            "events_by_type": events_by_type,
            "registered_models": len(models),
        }

        if models:
            content["model_ids"] = [m["id"] for m in models]
            evidence.extend(m["id"] for m in models)

        has_data = event_count > 0 or bundle_count > 0 or len(models) > 0
        if not has_data:
            gaps.append(
                "No forensic events, bundles, or models found. "
                "Deploy and operate the system to populate Section 1."
            )
        if not models:
            gaps.append("No models registered in the model registry.")

        return {
            "section_id": "annex_iv_1",
            "title": "General Description of the AI System",
            "status": "complete" if has_data and models else ("partial" if has_data else "missing"),
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 2: Intended Purpose
    # ------------------------------------------------------------------

    async def _section_2_intended_purpose(self) -> dict[str, Any]:
        consent_stats = await self.db.get_consent_coverage_stats()
        inference_count = 0
        events_by_type = await self.db.get_events_by_type()
        inference_count = events_by_type.get(EventType.INFERENCE_RESULT.value, 0)

        evidence: list[str] = []
        gaps: list[str] = []

        content: dict[str, Any] = {
            "intended_purpose": (
                "Biometric identification and verification of individuals "
                "using facial recognition technology, with full forensic "
                "audit trails for law enforcement and identity verification "
                "use cases."
            ),
            "use_cases": [
                "Identity verification for access control",
                "Forensic facial comparison for law enforcement",
                "Multi-modal biometric fusion (face + voice)",
            ],
            "deployment_context": (
                "On-premises or cloud deployment with federated learning "
                "support for privacy-preserving model training."
            ),
            "total_inferences_processed": inference_count,
            "consent_subjects": consent_stats["total_subjects"],
        }

        # Section 2 is inherently documentary; mark complete if inferences exist
        if inference_count > 0:
            status = "complete"
        elif consent_stats["total_subjects"] > 0:
            status = "partial"
            gaps.append(
                "No inference events recorded. Run recognition inferences "
                "to demonstrate intended purpose in practice."
            )
        else:
            status = "missing"
            gaps.append(
                "No inference events or consent records found. "
                "Demonstrate intended purpose through operational data."
            )

        return {
            "section_id": "annex_iv_2",
            "title": "Intended Purpose and Foreseeable Misuse",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 3: Risk Classification
    # ------------------------------------------------------------------

    async def _section_3_risk_classification(self) -> dict[str, Any]:
        bias_stats = await self.db.get_bias_audit_stats()

        evidence: list[str] = []
        gaps: list[str] = []

        content: dict[str, Any] = {
            "classification": "high-risk",
            "annex_iii_category": "Category 1 — Biometric Identification",
            "applicable_articles": [
                "Article 6(2) — High-risk AI systems",
                "Article 9 — Risk management system",
                "Article 10 — Data and data governance",
            ],
            "justification": (
                "FriendlyFace performs real-time and post-facto biometric "
                "identification of natural persons, placing it in Annex III, "
                "point 1(a) of the EU AI Act."
            ),
            "risk_mitigations": {
                "bias_audits_conducted": bias_stats["total_audits"],
                "bias_audit_pass_rate_pct": bias_stats["pass_rate_pct"],
            },
        }

        if bias_stats["total_audits"] > 0:
            evidence.append(f"bias_audits:{bias_stats['total_audits']}")

        # Risk classification is mostly documentary, but bias audits are evidence
        if bias_stats["total_audits"] > 0 and bias_stats["pass_rate_pct"] >= 70.0:
            status = "complete"
        elif bias_stats["total_audits"] > 0:
            status = "partial"
            gaps.append(
                f"Bias audit pass rate is {bias_stats['pass_rate_pct']}% "
                "(below 70% threshold). Address failing audits."
            )
        else:
            status = "partial"
            gaps.append(
                "No bias audits conducted. Run fairness audits to "
                "demonstrate risk mitigation under Article 9."
            )

        return {
            "section_id": "annex_iv_3",
            "title": "Risk Classification and Management",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 4: Training Data
    # ------------------------------------------------------------------

    async def _section_4_training_data(self) -> dict[str, Any]:
        events_by_type = await self.db.get_events_by_type()
        training_starts = events_by_type.get(EventType.TRAINING_START.value, 0)
        training_completes = events_by_type.get(EventType.TRAINING_COMPLETE.value, 0)
        provenance_count = await self.db.get_provenance_count()

        evidence: list[str] = []
        gaps: list[str] = []

        # Gather training event details
        training_events: list[dict[str, Any]] = []
        if training_starts > 0 or training_completes > 0:
            all_events = await self.db.get_all_events()
            for ev in all_events:
                if ev.event_type in (
                    EventType.TRAINING_START,
                    EventType.TRAINING_COMPLETE,
                ):
                    training_events.append(
                        {
                            "event_id": str(ev.id),
                            "event_type": ev.event_type.value,
                            "timestamp": ev.timestamp.isoformat()
                            if isinstance(ev.timestamp, datetime)
                            else str(ev.timestamp),
                            "payload_keys": list(ev.payload.keys()),
                        }
                    )
                    evidence.append(str(ev.id))

        content: dict[str, Any] = {
            "training_events_started": training_starts,
            "training_events_completed": training_completes,
            "provenance_nodes": provenance_count,
            "training_event_details": training_events,
            "data_governance": (
                "Training data provenance is tracked via the Provenance DAG. "
                "Each dataset, model, and inference is linked in an immutable "
                "directed acyclic graph with SHA-256 hashing."
            ),
        }

        if training_starts > 0 and provenance_count > 0:
            status = "complete"
        elif training_starts > 0 or provenance_count > 0:
            status = "partial"
            if provenance_count == 0:
                gaps.append(
                    "No provenance nodes found. Link training data to "
                    "models via the Provenance DAG for Article 10 compliance."
                )
            if training_starts == 0:
                gaps.append("No training_start events recorded.")
        else:
            status = "missing"
            gaps.append(
                "No training data records found. Train a model and record "
                "dataset provenance for Annex IV Section 4."
            )

        return {
            "section_id": "annex_iv_4",
            "title": "Training, Validation, and Testing Data",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 5: Bias Testing and Fairness
    # ------------------------------------------------------------------

    async def _section_5_bias_testing(self) -> dict[str, Any]:
        bias_stats = await self.db.get_bias_audit_stats()
        all_audits = await self.db.get_all_bias_audits()

        evidence: list[str] = []
        gaps: list[str] = []

        # Extract demographic groups tested
        groups_tested: set[str] = set()
        audit_summaries: list[dict[str, Any]] = []
        for audit in all_audits:
            evidence.append(str(audit.id))
            groups_tested.update(audit.groups_evaluated)
            audit_summaries.append(
                {
                    "audit_id": str(audit.id),
                    "demographic_parity_gap": audit.demographic_parity_gap,
                    "equalized_odds_gap": audit.equalized_odds_gap,
                    "groups_evaluated": audit.groups_evaluated,
                    "compliant": audit.compliant,
                }
            )

        content: dict[str, Any] = {
            "total_audits": bias_stats["total_audits"],
            "compliant_audits": bias_stats["compliant_audits"],
            "pass_rate_pct": bias_stats["pass_rate_pct"],
            "demographic_groups_tested": sorted(groups_tested),
            "audit_summaries": audit_summaries,
            "methodology": (
                "Fairness testing uses demographic parity gap and equalized "
                "odds gap metrics per EU AI Act Article 10 requirements. "
                "Automated audits trigger after configurable inference intervals."
            ),
        }

        if bias_stats["total_audits"] > 0 and len(groups_tested) >= 2:
            status = "complete"
        elif bias_stats["total_audits"] > 0:
            status = "partial"
            if len(groups_tested) < 2:
                gaps.append(
                    "Fewer than 2 demographic groups tested. Expand bias "
                    "audits to cover more groups."
                )
        else:
            status = "missing"
            gaps.append(
                "No bias audits found. Conduct fairness audits across "
                "demographic groups for Annex IV Section 5."
            )

        return {
            "section_id": "annex_iv_5",
            "title": "Bias Testing and Fairness Assessment",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 6: Performance Metrics
    # ------------------------------------------------------------------

    async def _section_6_performance_metrics(self) -> dict[str, Any]:
        models = await self.db.list_models()
        fl_sims = await self.db.list_fl_simulations()
        events_by_type = await self.db.get_events_by_type()
        inference_count = events_by_type.get(EventType.INFERENCE_RESULT.value, 0)
        fl_round_count = events_by_type.get(EventType.FL_ROUND.value, 0)

        evidence: list[str] = []
        gaps: list[str] = []

        model_summaries: list[dict[str, Any]] = []
        for m in models:
            model_summaries.append(
                {
                    "model_id": m["id"],
                    "created_at": m.get("created_at", ""),
                    "accuracy": m.get("accuracy"),
                    "n_components": m.get("n_components"),
                    "n_subjects": m.get("n_subjects"),
                }
            )
            evidence.append(m["id"])

        fl_summaries: list[dict[str, Any]] = []
        for sim in fl_sims:
            fl_summaries.append(
                {
                    "sim_id": sim["id"],
                    "created_at": sim.get("created_at", ""),
                    "final_accuracy": sim.get("final_accuracy"),
                    "rounds": sim.get("rounds"),
                }
            )
            evidence.append(sim["id"])

        content: dict[str, Any] = {
            "total_models": len(models),
            "model_summaries": model_summaries,
            "total_inferences": inference_count,
            "fl_simulations": len(fl_sims),
            "fl_summaries": fl_summaries,
            "fl_rounds_recorded": fl_round_count,
        }

        if len(models) > 0 and inference_count > 0:
            status = "complete"
        elif len(models) > 0 or inference_count > 0 or len(fl_sims) > 0:
            status = "partial"
            if not models:
                gaps.append("No models in registry. Register trained models.")
            if inference_count == 0:
                gaps.append("No inference results recorded.")
        else:
            status = "missing"
            gaps.append(
                "No models, inferences, or FL simulations found. "
                "Train and evaluate models for Annex IV Section 6."
            )

        return {
            "section_id": "annex_iv_6",
            "title": "System Performance and Accuracy Metrics",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 7: Human Oversight Measures
    # ------------------------------------------------------------------

    async def _section_7_human_oversight(self) -> dict[str, Any]:
        consent_stats = await self.db.get_consent_coverage_stats()
        explanation_stats = await self.db.get_explanation_coverage_stats()

        evidence: list[str] = []
        gaps: list[str] = []

        content: dict[str, Any] = {
            "oversight_mechanisms": [
                "Consent management with grant/revoke tracking",
                "RBAC (role-based access control) on sensitive endpoints",
                "Manual review via forensic bundle inspection",
                "Explainability reports (LIME, SHAP, SDD) for each inference",
                "Real-time SSE event stream for monitoring",
            ],
            "consent_coverage": consent_stats,
            "explanation_coverage": explanation_stats,
            "article_14_compliance": (
                "Human oversight is ensured through mandatory consent collection, "
                "explainable AI outputs, and admin-gated operations. All actions "
                "are logged as immutable ForensicEvents."
            ),
        }

        has_consent = consent_stats["total_subjects"] > 0
        has_explanations = explanation_stats["total_explanations"] > 0

        if has_consent and has_explanations:
            status = "complete"
        elif has_consent or has_explanations:
            status = "partial"
            if not has_consent:
                gaps.append(
                    "No consent records found. Record subject consent "
                    "for Article 14 compliance."
                )
            if not has_explanations:
                gaps.append(
                    "No explanation events found. Generate explanations "
                    "for inferences to support human oversight."
                )
        else:
            status = "missing"
            gaps.append(
                "No consent records or explanations found. "
                "Implement human oversight measures for Article 14."
            )

        return {
            "section_id": "annex_iv_7",
            "title": "Human Oversight Measures",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 8: Cybersecurity Measures
    # ------------------------------------------------------------------

    async def _section_8_cybersecurity(self) -> dict[str, Any]:
        chain_integrity = await self.service.verify_chain_integrity()
        bundle_stats = await self.db.get_bundle_integrity_stats()
        events_by_type = await self.db.get_events_by_type()
        security_alerts = events_by_type.get(EventType.SECURITY_ALERT.value, 0)

        evidence: list[str] = []
        gaps: list[str] = []

        content: dict[str, Any] = {
            "security_measures": [
                "SHA-256 hash chaining on all forensic events",
                "Append-only Merkle tree with inclusion proofs",
                "Schnorr ZK proofs on forensic bundles (BioZero pattern)",
                "Ed25519 DID:key + W3C Verifiable Credentials",
                "Data poisoning detection in federated learning",
                "Rate limiting on sensitive endpoints",
            ],
            "hash_chain_integrity": {
                "valid": chain_integrity["valid"],
                "event_count": chain_integrity["count"],
                "errors": chain_integrity["errors"],
            },
            "bundle_integrity": bundle_stats,
            "security_alerts_recorded": security_alerts,
            "merkle_root": self.service.get_merkle_root() or "N/A",
        }

        if chain_integrity["valid"] and chain_integrity["count"] > 0:
            evidence.append(f"chain_integrity:{chain_integrity['count']}_events")
            status = "complete"
        elif chain_integrity["count"] > 0:
            status = "partial"
            gaps.append(
                f"Hash chain integrity check failed with "
                f"{len(chain_integrity['errors'])} error(s)."
            )
        else:
            status = "missing"
            gaps.append(
                "No forensic events in hash chain. Operate the system "
                "to build the immutable event chain for Article 15."
            )

        return {
            "section_id": "annex_iv_8",
            "title": "Cybersecurity and Data Integrity Measures",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 9: Quality Management System
    # ------------------------------------------------------------------

    async def _section_9_quality_management(self) -> dict[str, Any]:
        event_count = await self.db.get_event_count()
        bundle_count = await self.db.get_bundle_count()
        bundle_stats = await self.db.get_bundle_integrity_stats()
        compliance_reports = await self.db.list_compliance_reports()

        evidence: list[str] = []
        gaps: list[str] = []

        for report in compliance_reports:
            evidence.append(report.get("id", ""))

        content: dict[str, Any] = {
            "quality_processes": [
                "Automated bias audits at configurable intervals",
                "Forensic event logging for all operations",
                "Bundle integrity verification with Merkle proofs",
                "Compliance reporting with weighted scoring",
                "ForensicSeal issuance for certified compliance",
            ],
            "total_forensic_events": event_count,
            "total_bundles": bundle_count,
            "bundle_integrity_pct": bundle_stats["integrity_pct"],
            "compliance_reports_generated": len(compliance_reports),
            "testing_documentation": (
                "Automated test suite with 93% code coverage. "
                "CI/CD pipeline runs lint, security scan, and full test "
                "suite on every commit."
            ),
        }

        if event_count > 0 and bundle_count > 0 and len(compliance_reports) > 0:
            status = "complete"
        elif event_count > 0 or bundle_count > 0:
            status = "partial"
            if bundle_count == 0:
                gaps.append("No forensic bundles created. Create bundles to verify quality.")
            if not compliance_reports:
                gaps.append(
                    "No compliance reports generated. Run compliance reporting "
                    "to document quality management."
                )
        else:
            status = "missing"
            gaps.append(
                "No forensic events or bundles found. "
                "Operate the system to build quality management evidence."
            )

        return {
            "section_id": "annex_iv_9",
            "title": "Quality Management System",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # Section 10: Post-Market Monitoring
    # ------------------------------------------------------------------

    async def _section_10_post_market(self) -> dict[str, Any]:
        events_by_type = await self.db.get_events_by_type()
        compliance_reports = await self.db.list_compliance_reports()
        seal_count = 0
        try:
            seals, total = await self.db.list_seals(limit=1, offset=0)
            seal_count = total
        except Exception:
            pass

        evidence: list[str] = []
        gaps: list[str] = []

        content: dict[str, Any] = {
            "monitoring_mechanisms": [
                "Server-Sent Events (SSE) stream at /events/stream for real-time monitoring",
                "Compliance reporting with auto-generation endpoints",
                "ForensicSeal issuance with expiry tracking",
                "Automated bias audit triggers at configurable intervals",
                "Prometheus metrics endpoint at /metrics",
            ],
            "compliance_report_count": len(compliance_reports),
            "forensic_seal_count": seal_count,
            "event_types_tracked": list(events_by_type.keys()),
            "post_market_plan": (
                "Continuous monitoring via real-time event streams, "
                "periodic compliance reports, and ForensicSeal renewals. "
                "Bias audits are triggered automatically and can be "
                "reviewed through the governance dashboard."
            ),
        }

        has_reports = len(compliance_reports) > 0
        has_seals = seal_count > 0

        if has_reports and has_seals:
            status = "complete"
            evidence.append(f"seals:{seal_count}")
        elif has_reports or has_seals:
            status = "partial"
            if not has_reports:
                gaps.append("No compliance reports generated for post-market evidence.")
            if not has_seals:
                gaps.append(
                    "No ForensicSeals issued. Issue seals to certify "
                    "ongoing compliance for post-market monitoring."
                )
        else:
            status = "missing"
            gaps.append(
                "No compliance reports or seals found. Generate reports "
                "and issue seals for post-market monitoring under the EU AI Act."
            )

        return {
            "section_id": "annex_iv_10",
            "title": "Post-Market Monitoring Plan",
            "status": status,
            "content": content,
            "evidence": evidence,
            "gaps": gaps,
        }

    # ------------------------------------------------------------------
    # HTML Renderer
    # ------------------------------------------------------------------

    def render_html(self, document: dict[str, Any]) -> str:
        """Render the conformity assessment document as a professional HTML page."""
        status_colors = {
            "complete": "#16a34a",
            "partial": "#d97706",
            "missing": "#dc2626",
        }
        status_labels = {
            "complete": "Complete",
            "partial": "Partial",
            "missing": "Missing",
        }

        sections_html = ""
        for section in document.get("sections", []):
            status = section.get("status", "missing")
            color = status_colors.get(status, "#6b7280")
            label = status_labels.get(status, status)

            # Build evidence list
            evidence_items = ""
            for ev in section.get("evidence", []):
                evidence_items += f"<li><code>{_html_escape(str(ev))}</code></li>\n"
            evidence_block = (
                f"<h4>Evidence</h4><ul>{evidence_items}</ul>"
                if evidence_items
                else ""
            )

            # Build gaps list
            gaps_items = ""
            for gap in section.get("gaps", []):
                gaps_items += f"<li>{_html_escape(gap)}</li>\n"
            gaps_block = (
                f'<h4 style="color:#dc2626;">Gaps</h4><ul>{gaps_items}</ul>'
                if gaps_items
                else ""
            )

            # Content summary
            content = section.get("content", {})
            content_rows = ""
            for key, value in content.items():
                if isinstance(value, (dict, list)):
                    display = f"<pre>{_html_escape(str(value))}</pre>"
                else:
                    display = _html_escape(str(value))
                content_rows += (
                    f"<tr><td><strong>{_html_escape(key)}</strong></td>"
                    f"<td>{display}</td></tr>\n"
                )

            sections_html += f"""
            <div style="border:1px solid #e5e7eb; border-radius:8px; padding:20px;
                        margin-bottom:16px; background:#fff;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">{_html_escape(section.get('section_id', ''))}:
                        {_html_escape(section.get('title', ''))}</h3>
                    <span style="background:{color}; color:#fff; padding:4px 12px;
                                 border-radius:12px; font-size:0.85em; font-weight:600;">
                        {label}
                    </span>
                </div>
                <table style="width:100%; border-collapse:collapse; margin-top:12px;">
                    {content_rows}
                </table>
                {evidence_block}
                {gaps_block}
            </div>
            """

        completeness = document.get("completeness", {})
        score_pct = completeness.get("score_pct", 0)
        score_color = (
            "#16a34a" if score_pct >= 70 else "#d97706" if score_pct >= 40 else "#dc2626"
        )

        # Top-level gaps
        all_gaps_html = ""
        for gap in document.get("gaps", []):
            all_gaps_html += f"<li>{_html_escape(gap)}</li>\n"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EU AI Act Annex IV - {_html_escape(document.get('system_name', 'AI System'))}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 960px; margin: 0 auto; padding: 24px;
            background: #f9fafb; color: #1f2937;
        }}
        h1, h2, h3, h4 {{ margin-top: 0; }}
        table {{ font-size: 0.9em; }}
        table td {{ padding: 4px 8px; vertical-align: top; border-bottom: 1px solid #f3f4f6; }}
        pre {{ background: #f3f4f6; padding: 8px; border-radius: 4px;
               font-size: 0.8em; overflow-x: auto; white-space: pre-wrap; }}
        code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 3px; font-size: 0.85em; }}
        ul {{ padding-left: 20px; }}
        li {{ margin-bottom: 4px; }}
    </style>
</head>
<body>
    <div style="background:#fff; border:1px solid #e5e7eb; border-radius:8px;
                padding:24px; margin-bottom:24px;">
        <h1 style="margin-bottom:4px;">EU AI Act Annex IV</h1>
        <h2 style="color:#6b7280; font-weight:400; margin-top:0;">
            Technical Documentation &mdash; {_html_escape(document.get('system_name', ''))}
        </h2>
        <table>
            <tr><td><strong>Document ID</strong></td>
                <td><code>{_html_escape(document.get('document_id', ''))}</code></td></tr>
            <tr><td><strong>Generated</strong></td>
                <td>{_html_escape(document.get('generated_at', ''))}</td></tr>
            <tr><td><strong>Regulation</strong></td>
                <td>{_html_escape(document.get('regulation', ''))}</td></tr>
            <tr><td><strong>Risk Classification</strong></td>
                <td>{_html_escape(document.get('risk_classification', ''))}</td></tr>
            <tr><td><strong>Enforcement Deadline</strong></td>
                <td>{_html_escape(document.get('enforcement_deadline', ''))}</td></tr>
            <tr><td><strong>Completeness</strong></td>
                <td><span style="font-size:1.4em; font-weight:700; color:{score_color};">
                    {score_pct}%
                </span>
                ({completeness.get('complete', 0)} complete,
                 {completeness.get('partial', 0)} partial,
                 {completeness.get('missing', 0)} missing)</td></tr>
        </table>
    </div>

    <h2>Sections</h2>
    {sections_html}

    {"<h2>All Gaps Requiring Action</h2><ul>" + all_gaps_html + "</ul>" if all_gaps_html else ""}

    <div style="text-align:center; color:#9ca3af; font-size:0.8em; margin-top:32px;">
        Generated by FriendlyFace Conformity Assessment Engine
    </div>
</body>
</html>"""


def _html_escape(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
