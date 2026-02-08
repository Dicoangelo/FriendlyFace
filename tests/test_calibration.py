"""Tests for confidence calibration via Platt scaling (US-048)."""

from __future__ import annotations

import numpy as np
import pytest

from friendlyface.recognition.calibration import (
    CalibratedScore,
    PlattCalibrator,
    calibrate_with_quality,
    quality_weight,
)


class TestPlattCalibrator:
    def test_default_not_fitted(self):
        cal = PlattCalibrator()
        assert cal.is_fitted is False

    def test_fit_marks_fitted(self):
        cal = PlattCalibrator()
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        cal.fit(scores, labels)
        assert cal.is_fitted is True

    def test_fit_too_few_samples(self):
        cal = PlattCalibrator()
        cal.fit(np.array([0.5]), np.array([1]))
        assert cal.is_fitted is False  # Needs >= 2

    def test_calibrate_output_range(self):
        cal = PlattCalibrator()
        scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        cal.fit(scores, labels)
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = cal.calibrate(s)
            assert 0.0 <= p <= 1.0

    def test_calibrate_monotonic(self):
        """Higher raw scores should produce higher calibrated scores."""
        cal = PlattCalibrator()
        scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1, 1])
        cal.fit(scores, labels)
        vals = [cal.calibrate(s) for s in np.linspace(0, 1, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-10

    def test_calibrate_batch_matches_single(self):
        cal = PlattCalibrator()
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        cal.fit(scores, labels)
        test_scores = np.array([0.2, 0.5, 0.8])
        batch = cal.calibrate_batch(test_scores)
        singles = [cal.calibrate(s) for s in test_scores]
        np.testing.assert_allclose(batch, singles, atol=1e-10)

    def test_default_calibrator_identity_like(self):
        """Unfitted calibrator should still return values in [0, 1]."""
        cal = PlattCalibrator()
        for s in [0.0, 0.3, 0.5, 0.7, 1.0]:
            p = cal.calibrate(s)
            assert 0.0 <= p <= 1.0

    def test_extreme_scores(self):
        cal = PlattCalibrator()
        scores = np.array([0.1, 0.9])
        labels = np.array([0, 1])
        cal.fit(scores, labels)
        assert 0.0 <= cal.calibrate(-100.0) <= 1.0
        assert 0.0 <= cal.calibrate(100.0) <= 1.0


class TestQualityWeight:
    def test_high_quality_near_one(self):
        w = quality_weight(0.9)
        assert w > 0.9

    def test_low_quality_near_half(self):
        w = quality_weight(0.1)
        assert w < 0.6

    def test_range(self):
        for q in np.linspace(0, 1, 20):
            w = quality_weight(float(q))
            assert 0.5 <= w <= 1.0

    def test_monotonic(self):
        weights = [quality_weight(q) for q in np.linspace(0, 1, 50)]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1] + 1e-10

    def test_steepness_parameter(self):
        # Higher steepness = sharper transition
        gentle = quality_weight(0.3, steepness=2.0)
        steep = quality_weight(0.3, steepness=20.0)
        # Both should be in range but differ
        assert 0.5 <= gentle <= 1.0
        assert 0.5 <= steep <= 1.0


class TestCalibrateWithQuality:
    def test_without_calibrator(self):
        result = calibrate_with_quality(0.8, 0.9)
        assert isinstance(result, CalibratedScore)
        assert result.method == "quality-only"
        assert result.calibrated_score == 0.8  # identity
        assert result.final_score <= result.calibrated_score

    def test_with_fitted_calibrator(self):
        cal = PlattCalibrator()
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        cal.fit(scores, labels)
        result = calibrate_with_quality(0.7, 0.85, calibrator=cal)
        assert result.method == "platt+quality"
        assert result.raw_score == pytest.approx(0.7, abs=1e-5)

    def test_with_unfitted_calibrator(self):
        cal = PlattCalibrator()  # Not fitted
        result = calibrate_with_quality(0.6, 0.7, calibrator=cal)
        assert result.method == "quality-only"

    def test_fields_rounded(self):
        result = calibrate_with_quality(0.123456789, 0.987654321)
        assert result.raw_score == round(0.123456789, 6)
        assert result.quality_weight == round(result.quality_weight, 6)
        assert result.final_score == round(result.final_score, 6)

    def test_final_score_is_product(self):
        result = calibrate_with_quality(0.8, 0.9)
        expected = round(result.calibrated_score * result.quality_weight, 6)
        assert result.final_score == expected

    def test_low_quality_reduces_score(self):
        high_q = calibrate_with_quality(0.8, 0.95)
        low_q = calibrate_with_quality(0.8, 0.1)
        assert high_q.final_score > low_q.final_score
