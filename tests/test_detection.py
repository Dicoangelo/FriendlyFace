"""Tests for face detection, landmarks, alignment, and quality scoring (US-045)."""

from __future__ import annotations

import numpy as np

from friendlyface.recognition.detection import (
    ALIGNED_SIZE,
    DetectedFace,
    FaceDetector,
    FaceLandmarks,
    _align_face,
    _compute_quality,
)


def _synthetic_face(h: int = 224, w: int = 224) -> np.ndarray:
    """Generate a synthetic RGB face-like image."""
    rng = np.random.default_rng(42)
    img = rng.integers(80, 200, size=(h, w, 3), dtype=np.uint8)
    # Simulate skin-tone in center region
    cy, cx = h // 2, w // 2
    img[cy - 30 : cy + 30, cx - 25 : cx + 25, 0] = 180  # R channel higher
    img[cy - 30 : cy + 30, cx - 25 : cx + 25, 1] = 140  # G
    img[cy - 30 : cy + 30, cx - 25 : cx + 25, 2] = 110  # B
    return img


class TestComputeQuality:
    def test_returns_float_in_unit_range(self):
        img = _synthetic_face()
        q = _compute_quality(img)
        assert isinstance(q, float)
        assert 0.0 <= q <= 1.0

    def test_grayscale_input(self):
        gray = np.random.default_rng(7).integers(0, 256, (112, 112), dtype=np.uint8)
        q = _compute_quality(gray)
        assert 0.0 <= q <= 1.0

    def test_sharp_image_scores_higher(self):
        """A high-contrast checkerboard should outscore a flat image."""
        flat = np.full((112, 112), 127, dtype=np.uint8)
        checkerboard = np.zeros((112, 112), dtype=np.uint8)
        checkerboard[::2, ::2] = 255
        checkerboard[1::2, 1::2] = 255
        assert _compute_quality(checkerboard) > _compute_quality(flat)

    def test_dark_image_lower_brightness_score(self):
        dark = np.full((112, 112, 3), 10, dtype=np.uint8)
        mid = np.full((112, 112, 3), 127, dtype=np.uint8)
        # Mid-brightness should score better on brightness component
        assert _compute_quality(mid) >= _compute_quality(dark)


class TestFaceLandmarks:
    def test_fields(self):
        lm = FaceLandmarks(
            left_eye=(30.0, 40.0),
            right_eye=(80.0, 40.0),
            nose_tip=(55.0, 65.0),
            mouth_left=(35.0, 85.0),
            mouth_right=(75.0, 85.0),
        )
        assert lm.left_eye == (30.0, 40.0)
        assert lm.nose_tip == (55.0, 65.0)


class TestAlignFace:
    def test_output_shape(self):
        img = _synthetic_face()
        lm = FaceLandmarks(
            left_eye=(80.0, 90.0),
            right_eye=(140.0, 90.0),
            nose_tip=(110.0, 120.0),
            mouth_left=(85.0, 150.0),
            mouth_right=(135.0, 150.0),
        )
        aligned = _align_face(img, lm)
        assert aligned.shape[:2] == ALIGNED_SIZE

    def test_custom_output_size(self):
        img = _synthetic_face(300, 300)
        lm = FaceLandmarks(
            left_eye=(100.0, 120.0),
            right_eye=(200.0, 120.0),
            nose_tip=(150.0, 160.0),
            mouth_left=(110.0, 200.0),
            mouth_right=(190.0, 200.0),
        )
        aligned = _align_face(img, lm, output_size=(64, 64))
        assert aligned.shape[:2] == (64, 64)


class TestFaceDetector:
    def test_backend_heuristic_without_mediapipe(self):
        det = FaceDetector(min_confidence=0.5)
        # In test env without mediapipe, fallback to heuristic
        assert det.backend in ("heuristic", "mediapipe")

    def test_detect_returns_list(self):
        det = FaceDetector()
        img = _synthetic_face()
        faces = det.detect(img)
        assert isinstance(faces, list)
        assert len(faces) >= 1

    def test_detected_face_fields(self):
        det = FaceDetector()
        img = _synthetic_face()
        faces = det.detect(img)
        f = faces[0]
        assert isinstance(f, DetectedFace)
        assert len(f.bbox) == 4
        assert f.landmarks is not None
        assert 0.0 <= f.quality_score <= 1.0
        assert 0.0 <= f.confidence <= 1.0

    def test_aligned_crop_generated(self):
        det = FaceDetector()
        img = _synthetic_face()
        faces = det.detect(img)
        assert faces[0].aligned is not None
        assert faces[0].aligned.shape[:2] == ALIGNED_SIZE

    def test_heuristic_bbox_within_bounds(self):
        det = FaceDetector()
        img = _synthetic_face(200, 300)
        faces = det.detect(img)
        x, y, w, h = faces[0].bbox
        assert x >= 0
        assert y >= 0
        assert x + w <= 300
        assert y + h <= 200

    def test_min_confidence_default(self):
        det = FaceDetector()
        assert det.min_confidence == 0.5

    def test_min_confidence_custom(self):
        det = FaceDetector(min_confidence=0.8)
        assert det.min_confidence == 0.8
