"""Bias audit engine for the FriendlyFace platform.

Computes demographic parity and equalized odds across demographic groups,
producing a BiasAuditRecord and logging it as a ForensicEvent(BIAS_AUDIT).

References:
  - EU AI Act Article 5/14 compliance
  - arXiv:2505.14320 - fairness metrics for facial recognition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from friendlyface.core.models import BiasAuditRecord, EventType
from friendlyface.core.service import ForensicService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input data structures
# ---------------------------------------------------------------------------


@dataclass
class GroupResult:
    """Recognition results for a single demographic group.

    Attributes:
        group_name: Identifier for the demographic group (e.g. "age_18_30").
        true_positives: Number of correct positive predictions.
        false_positives: Number of incorrect positive predictions.
        true_negatives: Number of correct negative predictions.
        false_negatives: Number of missed positive predictions.
    """

    group_name: str
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def total_positive_predictions(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def total_actual_positives(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def total_actual_negatives(self) -> int:
        return self.true_negatives + self.false_positives

    @property
    def total(self) -> int:
        return (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )

    @property
    def positive_prediction_rate(self) -> float:
        """Fraction of all samples predicted positive (for demographic parity)."""
        if self.total == 0:
            return 0.0
        return self.total_positive_predictions / self.total

    @property
    def true_positive_rate(self) -> float:
        """TPR = TP / (TP + FN). Also called recall / sensitivity."""
        if self.total_actual_positives == 0:
            return 0.0
        return self.true_positives / self.total_actual_positives

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN)."""
        if self.total_actual_negatives == 0:
            return 0.0
        return self.false_positives / self.total_actual_negatives


@dataclass
class FairnessThresholds:
    """Configurable thresholds for bias alerting.

    Attributes:
        demographic_parity_threshold: Maximum acceptable gap in positive
            prediction rates across groups. Default 0.1 (10%).
        equalized_odds_threshold: Maximum acceptable gap in TPR or FPR
            across groups. Default 0.1 (10%).
    """

    demographic_parity_threshold: float = 0.1
    equalized_odds_threshold: float = 0.1


@dataclass
class BiasAlert:
    """Alert emitted when a fairness threshold is breached."""

    metric: str
    gap: float
    threshold: float
    groups: list[str] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Core metric computation (pure functions, no I/O)
# ---------------------------------------------------------------------------


@dataclass
class GroupMetrics:
    """Computed per-group fairness metrics."""

    group_name: str
    positive_prediction_rate: float
    true_positive_rate: float
    false_positive_rate: float
    sample_count: int


def compute_group_metrics(group: GroupResult) -> GroupMetrics:
    """Compute fairness-relevant metrics for a single group."""
    return GroupMetrics(
        group_name=group.group_name,
        positive_prediction_rate=group.positive_prediction_rate,
        true_positive_rate=group.true_positive_rate,
        false_positive_rate=group.false_positive_rate,
        sample_count=group.total,
    )


def compute_demographic_parity_gap(metrics: list[GroupMetrics]) -> float:
    """Compute the max gap in positive prediction rates across groups.

    Demographic parity requires all groups to have the same positive
    prediction rate. The gap is max(rate) - min(rate).
    """
    if len(metrics) < 2:
        return 0.0
    rates = [m.positive_prediction_rate for m in metrics]
    return max(rates) - min(rates)


def compute_equalized_odds_gap(metrics: list[GroupMetrics]) -> float:
    """Compute the max gap in TPR and FPR across groups.

    Equalized odds requires equal TPR and FPR across groups.
    Returns the maximum of (max_tpr - min_tpr, max_fpr - min_fpr).
    """
    if len(metrics) < 2:
        return 0.0
    tpr_values = [m.true_positive_rate for m in metrics]
    fpr_values = [m.false_positive_rate for m in metrics]
    tpr_gap = max(tpr_values) - min(tpr_values)
    fpr_gap = max(fpr_values) - min(fpr_values)
    return max(tpr_gap, fpr_gap)


def compute_overall_fairness_score(
    dp_gap: float,
    eo_gap: float,
    thresholds: FairnessThresholds,
) -> float:
    """Compute an overall fairness score in [0, 1].

    1.0 = perfectly fair, 0.0 = maximally unfair.
    Scores each metric relative to its threshold and averages them.
    A metric at or below threshold scores 1.0; at 2x threshold scores 0.0.
    """

    def _metric_score(gap: float, threshold: float) -> float:
        if threshold <= 0:
            return 0.0 if gap > 0 else 1.0
        ratio = gap / threshold
        # Linear decay: 0 gap -> 1.0, at threshold -> 0.5, at 2x threshold -> 0.0
        return max(0.0, 1.0 - ratio * 0.5)

    dp_score = _metric_score(dp_gap, thresholds.demographic_parity_threshold)
    eo_score = _metric_score(eo_gap, thresholds.equalized_odds_threshold)
    return (dp_score + eo_score) / 2.0


def check_alerts(
    dp_gap: float,
    eo_gap: float,
    thresholds: FairnessThresholds,
    group_names: list[str],
) -> list[BiasAlert]:
    """Check whether fairness thresholds are breached and return alerts."""
    alerts: list[BiasAlert] = []

    if dp_gap > thresholds.demographic_parity_threshold:
        alerts.append(
            BiasAlert(
                metric="demographic_parity",
                gap=dp_gap,
                threshold=thresholds.demographic_parity_threshold,
                groups=group_names,
                message=(
                    f"Demographic parity gap {dp_gap:.4f} exceeds "
                    f"threshold {thresholds.demographic_parity_threshold:.4f}"
                ),
            )
        )

    if eo_gap > thresholds.equalized_odds_threshold:
        alerts.append(
            BiasAlert(
                metric="equalized_odds",
                gap=eo_gap,
                threshold=thresholds.equalized_odds_threshold,
                groups=group_names,
                message=(
                    f"Equalized odds gap {eo_gap:.4f} exceeds "
                    f"threshold {thresholds.equalized_odds_threshold:.4f}"
                ),
            )
        )

    return alerts


# ---------------------------------------------------------------------------
# BiasAuditor - orchestrates computation, forensic logging, alerting
# ---------------------------------------------------------------------------


class BiasAuditor:
    """Bias audit engine for facial recognition results.

    Accepts recognition results grouped by demographic attribute, computes
    demographic parity and equalized odds metrics, and logs the audit as
    a ForensicEvent(BIAS_AUDIT) in the forensic chain.

    Usage::

        auditor = BiasAuditor(service, thresholds=FairnessThresholds(0.08, 0.08))
        record, alerts = await auditor.audit(group_results)
    """

    def __init__(
        self,
        service: ForensicService,
        *,
        thresholds: FairnessThresholds | None = None,
        actor: str = "bias_auditor",
    ) -> None:
        self.service = service
        self.thresholds = thresholds or FairnessThresholds()
        self.actor = actor

    async def audit(
        self,
        group_results: list[GroupResult],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[BiasAuditRecord, list[BiasAlert]]:
        """Run a full bias audit and log to the forensic chain.

        Args:
            group_results: Recognition results for each demographic group.
            metadata: Optional extra metadata to include in the forensic event.

        Returns:
            Tuple of (BiasAuditRecord, list of BiasAlert).

        Raises:
            ValueError: If fewer than 2 groups are provided.
        """
        if len(group_results) < 2:
            raise ValueError("At least 2 demographic groups are required for a bias audit")

        # 1. Compute per-group metrics
        per_group = [compute_group_metrics(g) for g in group_results]

        # 2. Compute aggregate metrics
        dp_gap = compute_demographic_parity_gap(per_group)
        eo_gap = compute_equalized_odds_gap(per_group)
        fairness_score = compute_overall_fairness_score(dp_gap, eo_gap, self.thresholds)

        # 3. Check compliance
        group_names = [g.group_name for g in group_results]
        alerts = check_alerts(dp_gap, eo_gap, self.thresholds, group_names)
        compliant = len(alerts) == 0

        # 4. Build details dict
        details: dict[str, Any] = {
            "per_group_metrics": [
                {
                    "group": m.group_name,
                    "positive_prediction_rate": m.positive_prediction_rate,
                    "true_positive_rate": m.true_positive_rate,
                    "false_positive_rate": m.false_positive_rate,
                    "sample_count": m.sample_count,
                }
                for m in per_group
            ],
            "fairness_score": fairness_score,
            "thresholds": {
                "demographic_parity": self.thresholds.demographic_parity_threshold,
                "equalized_odds": self.thresholds.equalized_odds_threshold,
            },
            "alerts": [
                {
                    "metric": a.metric,
                    "gap": a.gap,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in alerts
            ],
        }
        if metadata:
            details["metadata"] = metadata

        # 5. Log forensic event
        event_payload: dict[str, Any] = {
            "demographic_parity_gap": dp_gap,
            "equalized_odds_gap": eo_gap,
            "fairness_score": fairness_score,
            "groups_evaluated": group_names,
            "compliant": compliant,
            "n_alerts": len(alerts),
        }
        if metadata:
            event_payload["metadata"] = metadata

        event = await self.service.record_event(
            event_type=EventType.BIAS_AUDIT,
            actor=self.actor,
            payload=event_payload,
        )

        # 6. Build BiasAuditRecord
        record = BiasAuditRecord(
            event_id=event.id,
            demographic_parity_gap=dp_gap,
            equalized_odds_gap=eo_gap,
            groups_evaluated=group_names,
            compliant=compliant,
            details=details,
        )

        # 7. Persist bias audit record
        await self.service.db.insert_bias_audit(record)

        # 8. Log alerts
        if alerts:
            logger.warning(
                "Bias audit %s: %d alert(s) triggered",
                record.id,
                len(alerts),
            )
            for alert in alerts:
                await self.service.record_event(
                    event_type=EventType.SECURITY_ALERT,
                    actor=self.actor,
                    payload={
                        "alert_type": "bias_threshold_breach",
                        "metric": alert.metric,
                        "gap": alert.gap,
                        "threshold": alert.threshold,
                        "groups": alert.groups,
                        "bias_audit_id": str(record.id),
                    },
                )

        return record, alerts
