"""Passive liveness detection for anti-spoofing (US-047).

Analyzes face images for presentation attacks (printed photos, screen
replay) using texture, frequency, and color-space features — all
numpy-only, no external ML models required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("friendlyface.recognition.liveness")


@dataclass
class LivenessResult:
    """Result of a liveness check."""

    is_live: bool
    score: float  # 0.0 (spoof) – 1.0 (live)
    checks: dict[str, float]  # Individual check scores
    details: str


def _moire_score(gray: np.ndarray) -> float:
    """Detect moiré patterns indicative of screen replay attacks.

    Moiré patterns create periodic high-frequency noise. We measure
    the ratio of high-frequency energy to total energy using simple
    finite-difference approximations.
    """
    # Horizontal and vertical gradients
    dx = np.diff(gray.astype(np.float64), axis=1)
    dy = np.diff(gray.astype(np.float64), axis=0)

    # Second derivatives (approximation of high-frequency content)
    d2x = np.diff(dx, axis=1)
    d2y = np.diff(dy, axis=0)

    hf_energy = float(np.mean(d2x**2) + np.mean(d2y**2))
    total_energy = float(np.mean(dx**2) + np.mean(dy**2)) + 1e-10

    # High ratio of second derivative to first = moiré-like patterns
    ratio = hf_energy / total_energy
    # Live faces have ratio < 1.5, screen replays > 2.0 typically
    return float(np.clip(1.0 - (ratio - 1.0) / 2.0, 0.0, 1.0))


def _frequency_score(gray: np.ndarray) -> float:
    """Analyze frequency distribution for naturalness.

    Real faces have a characteristic 1/f frequency falloff.
    Printed or screen-displayed faces have different spectral profiles.
    """
    # Use FFT for frequency analysis
    f_transform = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Compute energy in low, mid, high frequency bands
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    low = float(np.mean(magnitude[dist < min(h, w) * 0.1]))
    mid = float(np.mean(magnitude[(dist >= min(h, w) * 0.1) & (dist < min(h, w) * 0.3)]))
    high = float(np.mean(magnitude[dist >= min(h, w) * 0.3]))

    if low == 0:
        return 0.5

    # Natural faces: smooth falloff. Spoofs: abnormal mid/high ratios
    mid_ratio = mid / (low + 1e-10)
    high_ratio = high / (low + 1e-10)

    # Ideal: mid_ratio ~ 0.1-0.3, high_ratio ~ 0.01-0.05
    mid_score = float(np.clip(1.0 - abs(mid_ratio - 0.2) / 0.3, 0.0, 1.0))
    high_score = float(np.clip(1.0 - abs(high_ratio - 0.03) / 0.1, 0.0, 1.0))

    return (mid_score + high_score) / 2.0


def _color_score(image: np.ndarray) -> float:
    """Analyze color-space properties for liveness cues.

    Live faces have characteristic skin-tone distributions and
    color variance that differ from printed/screen images.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return 0.5  # Can't analyze color on grayscale

    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)

    # Color variance — live faces have natural variance
    color_var = float(np.std(r) + np.std(g) + np.std(b)) / 3.0
    var_score = float(np.clip(color_var / 40.0, 0.0, 1.0))

    # Inter-channel correlation — skin has characteristic R > G > B
    total = r + g + b + 1e-10
    r_ratio = np.mean(r / total)
    g_ratio = np.mean(g / total)

    # Skin typically: R > 0.36, G ~ 0.28-0.34
    skin_score = 1.0
    if r_ratio < 0.33 or g_ratio > 0.38:
        skin_score = 0.5

    return float(0.6 * var_score + 0.4 * skin_score)


def check_liveness(
    image: np.ndarray,
    threshold: float = 0.5,
) -> LivenessResult:
    """Run passive liveness detection on a face image.

    Args:
        image: Face crop as (H, W, 3) RGB or (H, W) grayscale uint8.
        threshold: Minimum combined score to classify as live.

    Returns:
        LivenessResult with is_live flag, combined score, and per-check details.
    """
    gray = image
    if image.ndim == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)

    moire = _moire_score(gray)
    freq = _frequency_score(gray)
    color = _color_score(image)

    # Weighted combination
    combined = 0.35 * moire + 0.35 * freq + 0.30 * color
    is_live = combined >= threshold

    checks = {
        "moire": round(moire, 4),
        "frequency": round(freq, 4),
        "color": round(color, 4),
    }

    details = "LIVE" if is_live else "SPOOF"
    if not is_live:
        worst = min(checks, key=checks.get)  # type: ignore[arg-type]
        details = f"SPOOF (lowest: {worst}={checks[worst]:.3f})"

    return LivenessResult(
        is_live=is_live,
        score=round(combined, 4),
        checks=checks,
        details=details,
    )
