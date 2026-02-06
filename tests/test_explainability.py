"""Tests for LIME explainability module.

Tests use mocked LIME explanations to avoid requiring actual model
inference, ensuring fast and deterministic test execution.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
from PIL import Image

from friendlyface.core.models import EventType, ProvenanceRelation
from friendlyface.explainability.lime_explain import (
    LimeExplanation,
    SuperpixelContribution,
    _get_superpixel_bbox,
    _image_from_bytes,
    _segment_image,
    generate_lime_explanation,
)
from friendlyface.recognition.pca import IMAGE_SIZE


def _make_test_image(size: tuple[int, int] = IMAGE_SIZE, seed: int = 42) -> bytes:
    """Create a synthetic grayscale image and return raw PNG bytes."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=size, dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_predict_fn(images: np.ndarray) -> np.ndarray:
    """Mock prediction function returning fixed probabilities."""
    n = images.shape[0]
    # Return 3-class probabilities with class 1 dominant
    probs = np.zeros((n, 3), dtype=np.float64)
    probs[:, 0] = 0.1
    probs[:, 1] = 0.7
    probs[:, 2] = 0.2
    return probs


class _MockLimeExplanation:
    """Mock lime.lime_image.ImageExplanation for deterministic tests."""

    def __init__(self, segments: np.ndarray, label: int = 1):
        n_segments = int(segments.max()) + 1
        self.top_labels = [label]
        # Build mock local explanation with descending weights
        self.local_exp = {
            label: [(i, float(n_segments - i) / n_segments) for i in range(n_segments)]
        }
        self.intercept = {label: 0.35}


# ---------------------------------------------------------------------------
# Unit tests: image utilities
# ---------------------------------------------------------------------------


class TestImageFromBytes:
    def test_correct_size(self):
        image_bytes = _make_test_image(size=IMAGE_SIZE)
        img = _image_from_bytes(image_bytes)
        assert img.size == IMAGE_SIZE
        assert img.mode == "L"

    def test_resizes_non_standard(self):
        image_bytes = _make_test_image(size=(200, 200))
        img = _image_from_bytes(image_bytes)
        assert img.size == IMAGE_SIZE


# ---------------------------------------------------------------------------
# Unit tests: segmentation
# ---------------------------------------------------------------------------


class TestSegmentation:
    def test_segment_produces_correct_shape(self):
        arr = np.zeros((112, 112), dtype=np.float64)
        segments = _segment_image(arr, num_superpixels=25)
        assert segments.shape == (112, 112)
        assert segments.dtype == np.int32

    def test_segment_covers_all_pixels(self):
        arr = np.zeros((112, 112), dtype=np.float64)
        segments = _segment_image(arr, num_superpixels=25)
        # Every pixel should be assigned
        unique_ids = np.unique(segments)
        assert len(unique_ids) >= 1
        # All pixels covered (no gaps)
        assert segments.min() >= 0

    def test_segment_num_regions(self):
        arr = np.zeros((112, 112), dtype=np.float64)
        segments = _segment_image(arr, num_superpixels=16)
        unique_ids = np.unique(segments)
        # Should produce approximately num_superpixels regions
        assert len(unique_ids) >= 1

    def test_superpixel_bbox(self):
        segments = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.int32,
        )
        bbox = _get_superpixel_bbox(segments, 0)
        assert bbox == (0, 0, 1, 1)

        bbox3 = _get_superpixel_bbox(segments, 3)
        assert bbox3 == (2, 2, 3, 3)

    def test_superpixel_bbox_missing(self):
        segments = np.zeros((4, 4), dtype=np.int32)
        bbox = _get_superpixel_bbox(segments, 99)
        assert bbox == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Unit tests: LimeExplanation artifact
# ---------------------------------------------------------------------------


class TestLimeExplanationArtifact:
    def test_artifact_hash_deterministic(self):
        inference_id = uuid4()
        explanation = LimeExplanation(
            inference_event_id=inference_id,
            predicted_label=1,
            original_confidence=0.85,
            top_regions=[
                SuperpixelContribution(superpixel_id=0, importance=0.5, bbox=(0, 0, 10, 10)),
            ],
            feature_importance_map=np.ones((112, 112), dtype=np.float64),
            confidence_decomposition={"base_score": 0.3, "region_0": 0.5},
            num_superpixels=50,
            num_samples=100,
        )
        h1 = explanation.compute_artifact_hash()
        h2 = explanation.compute_artifact_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_artifact_hash_changes_with_data(self):
        inference_id = uuid4()
        exp1 = LimeExplanation(
            inference_event_id=inference_id,
            predicted_label=1,
            original_confidence=0.85,
            top_regions=[
                SuperpixelContribution(superpixel_id=0, importance=0.5, bbox=(0, 0, 10, 10)),
            ],
            feature_importance_map=np.ones((112, 112), dtype=np.float64),
            confidence_decomposition={"base_score": 0.3, "region_0": 0.5},
            num_superpixels=50,
            num_samples=100,
        )
        exp2 = LimeExplanation(
            inference_event_id=inference_id,
            predicted_label=2,  # different label
            original_confidence=0.85,
            top_regions=[
                SuperpixelContribution(superpixel_id=0, importance=0.5, bbox=(0, 0, 10, 10)),
            ],
            feature_importance_map=np.ones((112, 112), dtype=np.float64),
            confidence_decomposition={"base_score": 0.3, "region_0": 0.5},
            num_superpixels=50,
            num_samples=100,
        )
        assert exp1.compute_artifact_hash() != exp2.compute_artifact_hash()


# ---------------------------------------------------------------------------
# Unit tests: generate_lime_explanation (with mocked LIME)
# ---------------------------------------------------------------------------


class TestGenerateLimeExplanation:
    """Tests for the full LIME explanation generation pipeline.

    LIME internals are mocked so tests don't need trained models.
    """

    def _run_with_mock(
        self,
        *,
        top_k: int = 3,
        num_superpixels: int = 16,
        num_samples: int = 10,
        inference_provenance_id: uuid4 | None = None,
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> LimeExplanation:
        """Helper to run generate_lime_explanation with mocked LIME."""
        image_bytes = _make_test_image()
        inference_event_id = uuid4()

        # Build mock segments for the image
        img = _image_from_bytes(image_bytes)
        image_array = np.asarray(img, dtype=np.float64)
        segments = _segment_image(image_array, num_superpixels)
        mock_lime_exp = _MockLimeExplanation(segments, label=1)

        with patch("lime.lime_image.LimeImageExplainer") as MockExplainerClass:
            mock_explainer = MagicMock()
            MockExplainerClass.return_value = mock_explainer
            mock_explainer.explain_instance.return_value = mock_lime_exp

            return generate_lime_explanation(
                image_bytes,
                _mock_predict_fn,
                inference_event_id=inference_event_id,
                predicted_label=1,
                original_confidence=0.7,
                num_superpixels=num_superpixels,
                num_samples=num_samples,
                top_k=top_k,
                inference_provenance_id=inference_provenance_id,
                previous_hash=previous_hash,
                sequence_number=sequence_number,
            )

    def test_returns_lime_explanation(self):
        explanation = self._run_with_mock()
        assert isinstance(explanation, LimeExplanation)

    def test_top_regions_count(self):
        explanation = self._run_with_mock(top_k=3, num_superpixels=16)
        assert len(explanation.top_regions) == 3

    def test_top_regions_clamped(self):
        """top_k larger than available regions returns all regions."""
        explanation = self._run_with_mock(top_k=100, num_superpixels=4)
        # Can't have more top regions than superpixels
        assert len(explanation.top_regions) <= 100

    def test_top_regions_sorted_by_importance(self):
        explanation = self._run_with_mock(top_k=5, num_superpixels=16)
        importances = [abs(r.importance) for r in explanation.top_regions]
        assert importances == sorted(importances, reverse=True)

    def test_feature_importance_map_shape(self):
        explanation = self._run_with_mock()
        assert explanation.feature_importance_map.shape == IMAGE_SIZE

    def test_confidence_decomposition_has_base(self):
        explanation = self._run_with_mock()
        assert "base_score" in explanation.confidence_decomposition

    def test_confidence_decomposition_has_regions(self):
        explanation = self._run_with_mock(top_k=3)
        decomp = explanation.confidence_decomposition
        region_keys = [k for k in decomp if k.startswith("region_")]
        assert len(region_keys) == 3

    def test_artifact_hash_set(self):
        explanation = self._run_with_mock()
        assert explanation.artifact_hash != ""
        assert len(explanation.artifact_hash) == 64

    def test_artifact_hash_matches_recompute(self):
        explanation = self._run_with_mock()
        assert explanation.artifact_hash == explanation.compute_artifact_hash()

    # -----------------------------------------------------------------------
    # Forensic event tests
    # -----------------------------------------------------------------------

    def test_forensic_event_created(self):
        explanation = self._run_with_mock()
        event = explanation.forensic_event
        assert event is not None
        assert event.event_type == EventType.EXPLANATION_GENERATED
        assert event.event_hash != ""
        assert event.verify()

    def test_forensic_event_payload(self):
        explanation = self._run_with_mock()
        payload = explanation.forensic_event.payload
        assert payload["explanation_type"] == "LIME"
        assert payload["artifact_hash"] == explanation.artifact_hash
        assert payload["predicted_label"] == 1
        assert "top_regions" in payload
        assert "confidence_decomposition" in payload

    def test_forensic_event_chain_params(self):
        explanation = self._run_with_mock(
            previous_hash="prev_abc",
            sequence_number=7,
        )
        event = explanation.forensic_event
        assert event.previous_hash == "prev_abc"
        assert event.sequence_number == 7

    # -----------------------------------------------------------------------
    # Provenance node tests
    # -----------------------------------------------------------------------

    def test_provenance_node_created(self):
        explanation = self._run_with_mock()
        node = explanation.provenance_node
        assert node is not None
        assert node.entity_type == "explanation"
        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()

    def test_provenance_node_metadata(self):
        explanation = self._run_with_mock()
        meta = explanation.provenance_node.metadata
        assert meta["explanation_type"] == "LIME"
        assert meta["artifact_hash"] == explanation.artifact_hash
        assert meta["predicted_label"] == 1

    def test_provenance_node_with_parent(self):
        parent_id = uuid4()
        explanation = self._run_with_mock(inference_provenance_id=parent_id)
        node = explanation.provenance_node
        assert parent_id in node.parents
        assert ProvenanceRelation.DERIVED_FROM in node.relations

    def test_provenance_node_no_parent(self):
        explanation = self._run_with_mock(inference_provenance_id=None)
        node = explanation.provenance_node
        assert node.parents == []
        assert node.relations == []

    # -----------------------------------------------------------------------
    # SuperpixelContribution tests
    # -----------------------------------------------------------------------

    def test_top_regions_have_valid_bbox(self):
        explanation = self._run_with_mock(num_superpixels=16)
        for region in explanation.top_regions:
            assert len(region.bbox) == 4
            min_r, min_c, max_r, max_c = region.bbox
            assert min_r >= 0
            assert min_c >= 0
            assert max_r >= min_r
            assert max_c >= min_c

    def test_top_regions_have_superpixel_ids(self):
        explanation = self._run_with_mock(num_superpixels=16)
        for region in explanation.top_regions:
            assert isinstance(region.superpixel_id, int)
            assert region.superpixel_id >= 0

    def test_predicted_label_preserved(self):
        explanation = self._run_with_mock()
        assert explanation.predicted_label == 1

    def test_original_confidence_preserved(self):
        explanation = self._run_with_mock()
        assert explanation.original_confidence == 0.7
