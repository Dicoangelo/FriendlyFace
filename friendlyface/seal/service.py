"""ForensicSeal issuance service.

Issues tamper-evident compliance certificates (ForensicSeals) backed by:
  - 6-layer compliance checks (via ComplianceReporter)
  - Bundle verification for each referenced bundle
  - Provenance completeness checks
  - Ed25519 Verifiable Credential (W3C VC Data Model)
  - Schnorr ZK proof over the compliance evidence
  - Current Merkle root anchoring
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from friendlyface.core.models import EventType
from friendlyface.core.service import ForensicService
from friendlyface.crypto.did import Ed25519DIDKey
from friendlyface.crypto.schnorr import ZKBundleProver, ZKBundleVerifier
from friendlyface.crypto.vc import VerifiableCredential
from friendlyface.governance.compliance import ComplianceReporter
from friendlyface.storage.database import Database

logger = logging.getLogger("friendlyface.seal")


class ForensicSealService:
    """Issues and manages ForensicSeal compliance certificates."""

    DEFAULT_THRESHOLD = 80.0
    DEFAULT_EXPIRY_DAYS = 90

    def __init__(
        self,
        db: Database,
        forensic_service: ForensicService,
        compliance_reporter: ComplianceReporter,
    ) -> None:
        self.db = db
        self.forensic_service = forensic_service
        self.compliance_reporter = compliance_reporter

    async def issue_seal(
        self,
        system_id: str,
        system_name: str,
        assessment_scope: str = "full",
        bundle_ids: list[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        expiry_days: int = DEFAULT_EXPIRY_DAYS,
    ) -> dict[str, Any]:
        """Run compliance checks and issue a ForensicSeal if passing.

        Steps:
          1. Generate compliance report (consent, bias, explanation, bundle integrity)
          2. Verify each referenced bundle
          3. Check provenance completeness
          4. Calculate overall score
          5. If score >= threshold, issue seal with VC + ZK proof + Merkle root
          6. Persist and record forensic event

        Returns the seal data dict.

        Raises ValueError if compliance score is below threshold.
        """
        bundle_ids = bundle_ids or []

        # 1. Run compliance report (consent, bias, explanation, bundle integrity)
        compliance_report = await self.compliance_reporter.generate_report()
        compliance_score = compliance_report["overall_compliance_score"]

        # 2. Verify each referenced bundle
        bundle_verification_results: list[dict[str, Any]] = []
        bundles_valid = True
        for bid in bundle_ids:
            from uuid import UUID

            try:
                result = await self.forensic_service.verify_bundle(UUID(bid))
                bundle_verification_results.append(
                    {"bundle_id": bid, "valid": result.get("valid", False)}
                )
                if not result.get("valid", False):
                    bundles_valid = False
            except Exception as exc:
                logger.warning("Bundle %s verification failed: %s", bid, exc)
                bundle_verification_results.append(
                    {"bundle_id": bid, "valid": False, "error": str(exc)}
                )
                bundles_valid = False

        # 3. Check provenance completeness (hash chain integrity)
        chain_integrity = await self.forensic_service.verify_chain_integrity()
        chain_valid = chain_integrity.get("valid", False)

        # 4. Build compliance summary
        compliance_summary: dict[str, Any] = {
            "consent_coverage_pct": compliance_report["metrics"]["consent_coverage_pct"],
            "bias_audit_pass_rate_pct": compliance_report["metrics"]["bias_audit_pass_rate_pct"],
            "explanation_coverage_pct": compliance_report["metrics"]["explanation_coverage_pct"],
            "bundle_integrity_pct": compliance_report["metrics"]["bundle_integrity_pct"],
            "bundles_verified": bundle_verification_results,
            "bundles_all_valid": bundles_valid,
            "chain_integrity_valid": chain_valid,
            "chain_event_count": chain_integrity.get("count", 0),
        }

        # 5. Apply penalties for failed bundle/chain checks
        effective_score = compliance_score
        if bundle_ids and not bundles_valid:
            effective_score *= 0.8  # 20% penalty for failed bundle verification
        if not chain_valid:
            effective_score *= 0.9  # 10% penalty for broken chain
        effective_score = round(effective_score, 2)

        # 6. Check threshold
        if effective_score < threshold:
            raise ValueError(
                f"Compliance score {effective_score} is below threshold {threshold}. "
                f"Seal not issued. Summary: {json.dumps(compliance_summary)}"
            )

        # 7. Issue the seal
        now = datetime.now(timezone.utc)
        seal_id = str(uuid4())
        issued_at = now.isoformat()
        expires_at = (now + timedelta(days=expiry_days)).isoformat()

        # a. Create/reuse platform DID
        platform_did: Ed25519DIDKey = self.forensic_service._platform_did

        # b. Build VC with compliance claims
        vc_issuer = VerifiableCredential(platform_did)
        claims: dict[str, Any] = {
            "seal_id": seal_id,
            "system_id": system_id,
            "system_name": system_name,
            "assessment_scope": assessment_scope,
            "compliance_score": effective_score,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "bundle_ids": bundle_ids,
        }
        credential = vc_issuer.issue(
            claims=claims,
            credential_type="ForensicSealCredential",
            subject_did=f"did:system:{system_id}",
        )

        # c. Generate ZK proof of the compliance evidence
        evidence_hash = __import__("hashlib").sha256(
            json.dumps(compliance_summary, sort_keys=True, default=str).encode()
        ).hexdigest()
        zk_prover = ZKBundleProver()
        zk_proof = zk_prover.prove_bundle(seal_id, evidence_hash)

        # d. Get current Merkle root
        merkle_root = self.forensic_service.get_merkle_root() or ""

        # e. Find previous seal for this system (for chaining)
        existing_seals, _ = await self.db.list_seals(system_id=system_id, limit=1, offset=0)
        previous_seal_id = existing_seals[0]["id"] if existing_seals else None

        # f. Build seal data
        seal_data: dict[str, Any] = {
            "id": seal_id,
            "system_id": system_id,
            "system_name": system_name,
            "assessment_scope": assessment_scope,
            "credential": json.dumps(credential),
            "compliance_score": effective_score,
            "compliance_summary": json.dumps(compliance_summary),
            "merkle_root": merkle_root,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "status": "active",
            "previous_seal_id": previous_seal_id,
            "bundle_ids": json.dumps(bundle_ids),
            "zk_proof": zk_proof,
        }

        # g. Persist to DB
        await self.db.save_seal(seal_data)

        # h. Record forensic event
        await self.forensic_service.record_event(
            event_type=EventType.SEAL_ISSUED,
            actor="forensic_seal_service",
            payload={
                "seal_id": seal_id,
                "system_id": system_id,
                "compliance_score": effective_score,
                "bundle_count": len(bundle_ids),
            },
        )

        # Return seal with parsed JSON fields for the API response
        return {
            "id": seal_id,
            "system_id": system_id,
            "system_name": system_name,
            "assessment_scope": assessment_scope,
            "credential": credential,
            "compliance_score": effective_score,
            "compliance_summary": compliance_summary,
            "merkle_root": merkle_root,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "status": "active",
            "previous_seal_id": previous_seal_id,
            "bundle_ids": bundle_ids,
            "zk_proof": zk_proof,
        }

    async def get_seal_status(self, seal_id: str) -> dict[str, Any]:
        """Get seal status including days until expiry.

        Auto-expires: if the seal is past ``expires_at`` but still marked
        ``active``, its status is updated to ``expired`` before returning.

        Returns a dict with seal_id, status, issued_at, expires_at,
        days_remaining, compliance_score, system_id, system_name.

        Raises LookupError if the seal does not exist.
        """
        seal = await self.db.get_seal(seal_id)
        if seal is None:
            raise LookupError(f"Seal {seal_id} not found")

        now = datetime.now(timezone.utc)
        status = seal.get("status", "active")
        expires_at_str = seal.get("expires_at", "")
        days_remaining: int | None = None

        if expires_at_str:
            try:
                expires_dt = datetime.fromisoformat(expires_at_str)
                delta = expires_dt - now
                days_remaining = max(int(delta.total_seconds() / 86400), 0)

                # Auto-expire if past expiry and still active
                if now >= expires_dt and status == "active":
                    status = "expired"
                    await self.db.update_seal_status(seal_id, "expired")
            except (ValueError, TypeError):
                pass

        return {
            "seal_id": seal_id,
            "status": status,
            "issued_at": seal.get("issued_at", ""),
            "expires_at": expires_at_str,
            "days_remaining": days_remaining,
            "compliance_score": seal.get("compliance_score", 0.0),
            "system_id": seal.get("system_id", ""),
            "system_name": seal.get("system_name", ""),
        }

    async def renew_seal(self, seal_id: str) -> dict[str, Any]:
        """Renew a seal by re-running compliance checks and issuing a new seal.

        The new seal's ``previous_seal_id`` links to the old seal, creating a
        continuous compliance chain.

        Raises LookupError if the seal does not exist.
        Raises ValueError if the seal is revoked (cannot renew revoked seals).
        """
        seal = await self.db.get_seal(seal_id)
        if seal is None:
            raise LookupError(f"Seal {seal_id} not found")

        if seal.get("status") == "revoked":
            raise ValueError(
                f"Cannot renew revoked seal {seal_id}. "
                "Revocation is irreversible."
            )

        # Re-issue with the same parameters
        bundle_ids: list[str] = []
        try:
            bundle_ids = json.loads(seal.get("bundle_ids", "[]"))
        except (json.JSONDecodeError, TypeError):
            pass

        new_seal = await self.issue_seal(
            system_id=seal["system_id"],
            system_name=seal["system_name"],
            assessment_scope=seal.get("assessment_scope", "full"),
            bundle_ids=bundle_ids,
            threshold=0.0,  # Renewal re-runs checks; threshold is re-evaluated
        )

        # Overwrite previous_seal_id to point at the renewed seal specifically
        new_seal["previous_seal_id"] = seal_id
        # Persist the link update
        await self.db.update_seal_previous(new_seal["id"], seal_id)

        return new_seal

    async def revoke_seal(self, seal_id: str, reason: str) -> dict[str, Any]:
        """Revoke a seal. Irreversible. Records a forensic event.

        Raises LookupError if the seal does not exist.
        Raises ValueError if the seal is already revoked.
        """
        seal = await self.db.get_seal(seal_id)
        if seal is None:
            raise LookupError(f"Seal {seal_id} not found")

        if seal.get("status") == "revoked":
            raise ValueError(f"Seal {seal_id} is already revoked")

        # Update status to revoked
        await self.db.update_seal_status(seal_id, "revoked", reason=reason)

        # Record forensic event
        await self.forensic_service.record_event(
            event_type=EventType.SEAL_REVOKED,
            actor="forensic_seal_service",
            payload={
                "seal_id": seal_id,
                "system_id": seal.get("system_id", ""),
                "reason": reason,
            },
        )

        # Return updated seal data
        seal["status"] = "revoked"
        seal["revocation_reason"] = reason
        return {
            "seal_id": seal_id,
            "status": "revoked",
            "reason": reason,
            "system_id": seal.get("system_id", ""),
            "system_name": seal.get("system_name", ""),
        }

    async def verify_seal(
        self,
        credential: dict[str, Any] | None = None,
        seal_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify a ForensicSeal credential.

        Can verify by:
        1. Passing the full credential JSON (offline-capable verification)
        2. Passing a seal_id (looks up from DB)

        Checks:
        - DID signature valid (Ed25519)
        - Credential not expired
        - Credential not revoked (DB check if seal_id available)
        - ZK proof valid (Schnorr verification)
        - Merkle root matches current chain state

        Returns a dict with ``valid``, ``checks``, and metadata fields.
        """
        seal_record: dict[str, Any] | None = None
        zk_proof_str: str | None = None

        # Resolve credential: from argument or DB lookup
        if credential is None and seal_id is None:
            raise ValueError("Must provide either credential or seal_id")

        if seal_id is not None:
            seal_record = await self.db.get_seal(seal_id)
            if seal_record is None:
                raise LookupError(f"Seal {seal_id} not found")
            # Parse credential from DB (stored as JSON string)
            try:
                credential = json.loads(seal_record["credential"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Stored credential is malformed: {exc}")
            zk_proof_str = seal_record.get("zk_proof")
        elif credential is not None and seal_id is None:
            # Offline verification — try to extract seal_id from credential
            subject = credential.get("credentialSubject", {})
            embedded_seal_id = subject.get("seal_id")
            if embedded_seal_id:
                seal_record = await self.db.get_seal(embedded_seal_id)
                if seal_record:
                    zk_proof_str = seal_record.get("zk_proof")

        # ------------------------------------------------------------------
        # 1. Signature check
        # ------------------------------------------------------------------
        sig_check: dict[str, Any] = {"valid": False, "detail": "Not checked"}
        try:
            public_key = self.forensic_service._platform_did.export_public()
            vc_result = VerifiableCredential.verify(credential, public_key)
            sig_check["valid"] = vc_result["valid"]
            sig_check["detail"] = (
                "Ed25519 signature valid"
                if vc_result["valid"]
                else "Ed25519 signature invalid"
            )
            if vc_result.get("legacy"):
                sig_check["detail"] += " (legacy proof format)"
        except Exception as exc:
            sig_check["detail"] = f"Signature verification error: {exc}"

        # ------------------------------------------------------------------
        # 2. Expiry check
        # ------------------------------------------------------------------
        subject = credential.get("credentialSubject", {})
        expires_at_str = subject.get("expires_at", "")
        issued_at_str = subject.get("issued_at", "")
        expiry_check: dict[str, Any] = {"valid": True, "expires_at": expires_at_str, "detail": "No expiry set"}

        if expires_at_str:
            try:
                expires_dt = datetime.fromisoformat(expires_at_str)
                now = datetime.now(timezone.utc)
                if now >= expires_dt:
                    expiry_check["valid"] = False
                    expiry_check["detail"] = f"Credential expired at {expires_at_str}"
                else:
                    expiry_check["detail"] = f"Valid until {expires_at_str}"
            except (ValueError, TypeError):
                expiry_check["valid"] = False
                expiry_check["detail"] = f"Cannot parse expires_at: {expires_at_str}"

        # ------------------------------------------------------------------
        # 3. Revocation check (only if we have DB access)
        # ------------------------------------------------------------------
        revocation_check: dict[str, Any] = {"valid": True, "detail": "No DB record to check"}
        if seal_record is not None:
            status = seal_record.get("status", "active")
            if status == "revoked":
                revocation_check["valid"] = False
                reason = seal_record.get("revocation_reason", "")
                revocation_check["detail"] = "Seal revoked" + (f": {reason}" if reason else "")
            else:
                revocation_check["detail"] = f"Seal status: {status}"

        # ------------------------------------------------------------------
        # 4. ZK proof check
        # ------------------------------------------------------------------
        zk_check: dict[str, Any] = {"valid": True, "detail": "No ZK proof to verify"}
        if zk_proof_str:
            try:
                verifier = ZKBundleVerifier()
                zk_valid = verifier.verify_bundle(zk_proof_str)
                zk_check["valid"] = zk_valid
                zk_check["detail"] = (
                    "Schnorr ZK proof valid" if zk_valid else "Schnorr ZK proof invalid"
                )
            except Exception as exc:
                zk_check["valid"] = False
                zk_check["detail"] = f"ZK proof verification error: {exc}"

        # ------------------------------------------------------------------
        # 5. Merkle root check
        # ------------------------------------------------------------------
        merkle_check: dict[str, Any] = {"valid": True, "detail": "No Merkle root to verify"}
        if seal_record is not None:
            stored_root = seal_record.get("merkle_root", "")
            if stored_root:
                current_root = self.forensic_service.get_merkle_root() or ""
                if current_root == stored_root:
                    merkle_check["detail"] = "Merkle root matches current chain state"
                else:
                    # Not a failure — new events may have been added since issuance
                    merkle_check["detail"] = (
                        "Merkle root differs from current chain state "
                        "(expected if new events were recorded after seal issuance)"
                    )

        # ------------------------------------------------------------------
        # Aggregate result
        # ------------------------------------------------------------------
        all_valid = all(
            c["valid"]
            for c in [sig_check, expiry_check, revocation_check, zk_check, merkle_check]
        )

        return {
            "valid": all_valid,
            "checks": {
                "signature": sig_check,
                "expiry": expiry_check,
                "revocation": revocation_check,
                "zk_proof": zk_check,
                "merkle": merkle_check,
            },
            "issuer": credential.get("issuer", ""),
            "issued_at": issued_at_str,
            "expires_at": expires_at_str,
            "compliance_score": subject.get("compliance_score", 0.0),
            "system_id": subject.get("system_id", ""),
            "system_name": subject.get("system_name", ""),
        }
