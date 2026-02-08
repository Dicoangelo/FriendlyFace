"""Data retention policy engine (US-051).

Manages retention policies that automatically trigger cryptographic erasure
when subject data exceeds configured retention periods.  Supports CRUD
operations on policies and on-demand or scheduled evaluation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from friendlyface.governance.erasure import ErasureManager
from friendlyface.storage.database import Database

logger = logging.getLogger("friendlyface.governance.retention")


class RetentionEngine:
    """Evaluate and enforce data retention policies."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.erasure = ErasureManager(db)

    # ------------------------------------------------------------------
    # Policy CRUD
    # ------------------------------------------------------------------

    async def create_policy(
        self,
        name: str,
        entity_type: str,
        retention_days: int,
        action: str = "erase",
        enabled: bool = True,
    ) -> dict:
        """Create a new retention policy."""
        now = datetime.now(timezone.utc).isoformat()
        policy = {
            "id": str(uuid4()),
            "name": name,
            "entity_type": entity_type,
            "retention_days": retention_days,
            "action": action,
            "enabled": enabled,
            "created_at": now,
            "updated_at": now,
        }
        await self.db.insert_retention_policy(policy)
        logger.info("Created retention policy %s: %s (%d days)", policy["id"], name, retention_days)
        return policy

    async def list_policies(self, enabled_only: bool = False) -> list[dict]:
        """List all retention policies."""
        return await self.db.list_retention_policies(enabled_only=enabled_only)

    async def get_policy(self, policy_id: str) -> dict | None:
        """Get a specific policy by ID."""
        return await self.db.get_retention_policy(policy_id)

    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a retention policy."""
        deleted = await self.db.delete_retention_policy(policy_id)
        if deleted:
            logger.info("Deleted retention policy %s", policy_id)
        return deleted

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def evaluate(self) -> dict:
        """Evaluate all enabled policies and erase expired subjects.

        Returns a summary of actions taken per policy.
        """
        policies = await self.db.list_retention_policies(enabled_only=True)
        results: list[dict] = []
        total_erased = 0

        for policy in policies:
            subjects = await self.db.get_subjects_exceeding_retention(
                policy["entity_type"], policy["retention_days"]
            )

            erased: list[str] = []
            for subject_id in subjects:
                if policy["action"] == "erase":
                    result = await self.erasure.erase_subject(subject_id)
                    if result["status"] in ("completed", "no_data"):
                        erased.append(subject_id)

            total_erased += len(erased)
            results.append(
                {
                    "policy_id": policy["id"],
                    "policy_name": policy["name"],
                    "entity_type": policy["entity_type"],
                    "retention_days": policy["retention_days"],
                    "subjects_found": len(subjects),
                    "subjects_erased": len(erased),
                    "erased_subjects": erased,
                }
            )

            if erased:
                logger.info(
                    "Policy %s: erased %d subjects (retention: %d days)",
                    policy["name"],
                    len(erased),
                    policy["retention_days"],
                )

        return {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "policies_evaluated": len(policies),
            "total_subjects_erased": total_erased,
            "results": results,
        }
