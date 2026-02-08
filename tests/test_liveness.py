"""Tests for passive liveness detection (US-047)."""

from __future__ import annotations

import numpy as np
import pytest

from friendlyface.recognition.liveness import (
    LivenessResult,
    _color_score,
    _frequency_score,
    _moire_score,
    check_liveness,
)


def _natural_face(h: int = 112, w: int = 112) -> np.ndarray:
    """Simulate a natural face image with skin-like colors and smooth gradients."""
    rng = np.random.default_rng(42)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Skin-tone base (R > G > B)
    img[:, :, 0] = 180  # R
    img[:, :, 1] = 140  # G
    img[:, :, 2] = 110  # B
    # Add natural variation
    noise = rng.integers(-15, 16, size=(h, w, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _flat_image(h: int = 112, w: int = 112) -> np.ndarray:
    """A completely flat (single color) image — suspicious for liveness."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


class TestMoireScore:
    def test_returns_float_in_range(self):
        gray = np.random.default_rng(1).integers(0, 256, (112, 112), dtype=np.uint8)
        score = _moire_score(gray)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_smooth_image_high_score(self):
        """Smooth gradients should produce low second derivatives → high score."""
        gray = np.tile(np.linspace(50, 200, 112, dtype=np.uint8), (112, 1))
        score = _moire_score(gray)
        assert score > 0.5

    def test_high_frequency_low_score(self):
        """Alternating pattern should produce high second derivatives → lower score."""
        gray = np.zeros((112, 112), dtype=np.uint8)
        gray[::2, :] = 255
        score = _moire_score(gray)
        assert score < 0.8


class TestFrequencyScore:
    def test_returns_float_in_range(self):
        gray = np.random.default_rng(3).integers(0, 256, (112, 112), dtype=np.uint8)
        score = _frequency_score(gray)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_constant_image(self):
        gray = np.full((112, 112), 128, dtype=np.uint8)
        score = _frequency_score(gray)
        # Constant image has zero low-freq energy (after DC) → fallback
        assert 0.0 <= score <= 1.0


class TestColorScore:
    def test_returns_float_in_range(self):
        img = _natural_face()
        score = _color_score(img)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_grayscale_returns_default(self):
        gray = np.random.default_rng(5).integers(0, 256, (112, 112), dtype=np.uint8)
        score = _color_score(gray)
        assert score == 0.5

    def test_skin_tone_scores_high(self):
        img = _natural_face()
        score = _color_score(img)
        assert score > 0.4

    def test_blue_image_lower_score(self):
        """An image with B > R should have lower skin score."""
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        img[:, :, 0] = 50  # Low R
        img[:, :, 1] = 50  # Low G
        img[:, :, 2] = 200  # High B
        score = _color_score(img)
        assert score < _color_score(_natural_face())


class TestCheckLiveness:
    def test_returns_liveness_result(self):
        img = _natural_face()
        result = check_liveness(img)
        assert isinstance(result, LivenessResult)
        assert isinstance(result.is_live, bool)
        assert 0.0 <= result.score <= 1.0
        assert "moire" in result.checks
        assert "frequency" in result.checks
        assert "color" in result.checks

    def test_details_live(self):
        img = _natural_face()
        result = check_liveness(img, threshold=0.0)  # Very low threshold
        assert result.is_live is True
        assert result.details == "LIVE"

    def test_details_spoof(self):
        img = _natural_face()
        result = check_liveness(img, threshold=1.0)  # Impossible threshold
        assert result.is_live is False
        assert result.details.startswith("SPOOF")

    def test_grayscale_input(self):
        gray = np.random.default_rng(9).integers(0, 256, (112, 112), dtype=np.uint8)
        result = check_liveness(gray)
        assert isinstance(result, LivenessResult)

    def test_check_scores_rounded(self):
        result = check_liveness(_natural_face())
        for v in result.checks.values():
            # All check values should have at most 4 decimal places
            assert v == round(v, 4)

    def test_combined_score_is_weighted_sum(self):
        result = check_liveness(_natural_face())
        expected = (
            0.35 * result.checks["moire"]
            + 0.35 * result.checks["frequency"]
            + 0.30 * result.checks["color"]
        )
        assert result.score == pytest.approx(round(expected, 4), abs=1e-4)

    def test_threshold_below_score_is_live(self):
        img = _natural_face()
        result = check_liveness(img)
        # Re-run with threshold slightly below score
        easier = check_liveness(img, threshold=result.score - 0.01)
        assert easier.is_live is True

    def test_spoof_shows_worst_check(self):
        result = check_liveness(_natural_face(), threshold=1.0)
        assert "lowest:" in result.details
