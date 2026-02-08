"""Confidence calibration via Platt scaling and quality weighting (US-048).

Maps raw model scores to well-calibrated probabilities so that a
reported 80% confidence actually corresponds to ~80% accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("friendlyface.recognition.calibration")


@dataclass
class CalibratedScore:
    """A calibrated confidence score with provenance."""

    raw_score: float
    calibrated_score: float
    quality_weight: float
    final_score: float
    method: str


class PlattCalibrator:
    """Platt scaling (logistic regression) for score calibration.

    Fits sigmoid parameters A, B such that:
        P(y=1 | s) = 1 / (1 + exp(A*s + B))

    where s is the raw score from the model.
    """

    def __init__(self) -> None:
        self.a: float = -1.0  # Default: identity-like mapping
        self.b: float = 0.0
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, scores: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> None:
        """Fit Platt scaling parameters via Newton's method.

        Args:
            scores: Raw model scores, shape (N,).
            labels: Binary labels (0/1), shape (N,).
            max_iter: Maximum optimization iterations.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        n = len(scores)
        if n < 2:
            logger.warning("Need at least 2 samples for Platt scaling; using defaults")
            return

        # Target probabilities with label smoothing (Platt's method)
        n_pos = np.sum(labels)
        n_neg = n - n_pos
        t = np.where(labels > 0.5, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))

        a, b = 0.0, np.log((n_neg + 1) / (n_pos + 1))
        lr = 1.0

        for _ in range(max_iter):
            # Forward pass
            z = a * scores + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

            # Gradient
            diff = p - t
            da = float(np.dot(diff, scores))
            db = float(np.sum(diff))

            # Hessian diagonal (approximate)
            w = p * (1 - p) + 1e-12
            haa = float(np.dot(w, scores**2))
            hbb = float(np.sum(w))

            if haa == 0 or hbb == 0:
                break

            a -= lr * da / haa
            b -= lr * db / hbb

        self.a = a
        self.b = b
        self._fitted = True
        logger.info("Platt calibration fitted: A=%.4f, B=%.4f", a, b)

    def calibrate(self, score: float) -> float:
        """Apply Platt scaling to a raw score.

        Returns a calibrated probability in [0, 1].
        """
        z = self.a * score + self.b
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))

    def calibrate_batch(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to an array of raw scores."""
        z = self.a * scores + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def quality_weight(quality_score: float, steepness: float = 5.0) -> float:
    """Compute a quality-based weight for score adjustment.

    Uses a sigmoid-like curve centered at quality=0.5:
    - High quality (>0.7) → weight ≈ 1.0
    - Low quality (<0.3) → weight ≈ 0.5 (downweights unreliable detections)

    Args:
        quality_score: Face quality in [0, 1].
        steepness: Controls the transition steepness.

    Returns:
        Weight in [0.5, 1.0].
    """
    # Map quality [0, 1] to weight [0.5, 1.0]
    sigmoid = 1.0 / (1.0 + np.exp(-steepness * (quality_score - 0.5)))
    return float(0.5 + 0.5 * sigmoid)


def calibrate_with_quality(
    raw_score: float,
    quality_score: float,
    calibrator: PlattCalibrator | None = None,
) -> CalibratedScore:
    """Calibrate a raw model score with Platt scaling and quality weighting.

    Args:
        raw_score: Raw confidence from the recognition model.
        quality_score: Face image quality in [0, 1].
        calibrator: Optional fitted PlattCalibrator. If None, uses identity.

    Returns:
        CalibratedScore with raw, calibrated, and quality-adjusted scores.
    """
    if calibrator is not None and calibrator.is_fitted:
        cal = calibrator.calibrate(raw_score)
        method = "platt+quality"
    else:
        cal = raw_score
        method = "quality-only"

    qw = quality_weight(quality_score)
    final = cal * qw

    return CalibratedScore(
        raw_score=round(raw_score, 6),
        calibrated_score=round(cal, 6),
        quality_weight=round(qw, 6),
        final_score=round(final, 6),
        method=method,
    )
