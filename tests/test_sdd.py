"""Tests for SDD saliency explainability module."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from friendlyface.core.models import EventType, ProvenanceRelation
from friendlyface.explainability.sdd_explain import (
    FacialRegion,
    SDDExplanation,
    _CANONICAL_REGIONS,
    _normalize_saliency,
    _score_regions,
    generate_sdd_explanation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grayscale_bytes(seed: int = 42) -> bytes:
    """Generate a deterministic 112x112 grayscale PNG as bytes."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(112, 112), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _simple_predict_fn(flat_image: np.ndarray) -> tuple[str, float]:
    """A synthetic predict_fn that returns deterministic results.

    Confidence is the mean pixel value (so perturbations produce
    non-zero gradients).
    """
    confidence = float(np.mean(flat_image))
    return ("person_A", confidence)


def _constant_predict_fn(flat_image: np.ndarray) -> tuple[str, float]:
    """Predict function that always returns the same confidence."""
    return ("person_B", 0.85)


# ---------------------------------------------------------------------------
# SDDExplanation dataclass
# ---------------------------------------------------------------------------


class TestSDDExplanation:
    def test_compute_artifact_hash_deterministic(self):
        """Same inputs must produce the same artifact hash."""
        regions = [
            FacialRegion(name="nose", bbox=(51, 28, 75, 83), importance=1.0, pixel_count=1400),
            FacialRegion(name="mouth", bbox=(76, 20, 95, 91), importance=0.8, pixel_count=1440),
        ]
        saliency = [[0.0] * 112 for _ in range(112)]

        exp_a = SDDExplanation(
            inference_event_id="evt-001",
            predicted_label="person_A",
            original_confidence=0.95,
            saliency_map=saliency,
            regions=regions,
            dominant_region="nose",
        )
        exp_b = SDDExplanation(
            inference_event_id="evt-001",
            predicted_label="person_A",
            original_confidence=0.95,
            saliency_map=saliency,
            regions=regions,
            dominant_region="nose",
        )

        assert exp_a.compute_artifact_hash() == exp_b.compute_artifact_hash()
        assert len(exp_a.compute_artifact_hash()) == 64  # SHA-256 hex

    def test_compute_artifact_hash_changes_with_input(self):
        """Different inputs must produce different hashes."""
        regions = [
            FacialRegion(name="nose", bbox=(51, 28, 75, 83), importance=1.0, pixel_count=1400),
        ]
        saliency = [[0.0] * 112 for _ in range(112)]

        exp_a = SDDExplanation(
            inference_event_id="evt-001",
            predicted_label="person_A",
            original_confidence=0.95,
            saliency_map=saliency,
            regions=regions,
            dominant_region="nose",
        )
        exp_b = SDDExplanation(
            inference_event_id="evt-002",
            predicted_label="person_A",
            original_confidence=0.95,
            saliency_map=saliency,
            regions=regions,
            dominant_region="nose",
        )

        assert exp_a.compute_artifact_hash() != exp_b.compute_artifact_hash()


# ---------------------------------------------------------------------------
# Region scoring internals
# ---------------------------------------------------------------------------


class TestScoreRegions:
    def test_returns_all_seven_regions(self):
        """_score_regions must return all 7 canonical regions."""
        gradient_map = np.random.default_rng(0).standard_normal((112, 112))
        regions = _score_regions(gradient_map)
        assert len(regions) == 7
        names = {r.name for r in regions}
        assert names == set(_CANONICAL_REGIONS.keys())

    def test_sorted_by_importance_descending(self):
        """Regions must be sorted by importance (highest first)."""
        gradient_map = np.random.default_rng(1).standard_normal((112, 112))
        regions = _score_regions(gradient_map)
        importances = [r.importance for r in regions]
        assert importances == sorted(importances, reverse=True)

    def test_importance_normalized_zero_to_one(self):
        """All importance values must be in [0, 1]."""
        gradient_map = np.random.default_rng(2).standard_normal((112, 112))
        regions = _score_regions(gradient_map)
        for r in regions:
            assert 0.0 <= r.importance <= 1.0

    def test_max_importance_is_one(self):
        """The top region must have importance exactly 1.0."""
        gradient_map = np.random.default_rng(3).standard_normal((112, 112))
        regions = _score_regions(gradient_map)
        assert regions[0].importance == 1.0

    def test_zero_gradient_map(self):
        """All-zero gradient map must not cause division errors."""
        gradient_map = np.zeros((112, 112))
        regions = _score_regions(gradient_map)
        assert len(regions) == 7
        for r in regions:
            assert r.importance == 0.0

    def test_pixel_counts_positive(self):
        """Each region must report a positive pixel count."""
        gradient_map = np.ones((112, 112))
        regions = _score_regions(gradient_map)
        for r in regions:
            assert r.pixel_count > 0


# ---------------------------------------------------------------------------
# Saliency normalization
# ---------------------------------------------------------------------------


class TestNormalizeSaliency:
    def test_shape_preserved(self):
        grad = np.random.default_rng(10).standard_normal((112, 112))
        sal = _normalize_saliency(grad)
        assert len(sal) == 112
        assert len(sal[0]) == 112

    def test_values_in_zero_one(self):
        grad = np.random.default_rng(11).standard_normal((112, 112))
        sal = _normalize_saliency(grad)
        flat = [v for row in sal for v in row]
        assert min(flat) >= 0.0
        assert max(flat) <= 1.0

    def test_max_value_is_one(self):
        grad = np.random.default_rng(12).standard_normal((112, 112))
        sal = _normalize_saliency(grad)
        flat = [v for row in sal for v in row]
        assert max(flat) == pytest.approx(1.0)

    def test_zero_gradient_map(self):
        grad = np.zeros((112, 112))
        sal = _normalize_saliency(grad)
        flat = [v for row in sal for v in row]
        assert all(v == 0.0 for v in flat)


# ---------------------------------------------------------------------------
# Full generate_sdd_explanation
# ---------------------------------------------------------------------------


class TestGenerateSDDExplanation:
    """Integration tests for the public generate_sdd_explanation function.

    These use a tiny synthetic predict_fn so gradient computation
    is fast (no real model needed).
    """

    @pytest.fixture
    def image_bytes(self) -> bytes:
        return _make_grayscale_bytes(seed=42)

    @pytest.fixture
    def features(self) -> np.ndarray:
        return np.random.default_rng(42).standard_normal(50)

    def test_saliency_map_shape(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-1"
        )
        assert len(exp.saliency_map) == 112
        assert all(len(row) == 112 for row in exp.saliency_map)

    def test_all_seven_regions_present(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-2"
        )
        names = {r.name for r in exp.regions}
        assert names == set(_CANONICAL_REGIONS.keys())
        assert len(exp.regions) == 7

    def test_regions_sorted_by_importance(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-3"
        )
        importances = [r.importance for r in exp.regions]
        assert importances == sorted(importances, reverse=True)

    def test_dominant_region_is_first(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-4"
        )
        assert exp.dominant_region == exp.regions[0].name

    def test_region_importance_normalized(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-5"
        )
        for r in exp.regions:
            assert 0.0 <= r.importance <= 1.0

    def test_artifact_hash_deterministic(self):
        """Same image + same predict_fn must yield the same artifact hash."""
        img = _make_grayscale_bytes(seed=99)
        feats = np.zeros(50)

        exp_a = generate_sdd_explanation(
            img, _simple_predict_fn, feats, inference_event_id="evt-det"
        )
        exp_b = generate_sdd_explanation(
            img, _simple_predict_fn, feats, inference_event_id="evt-det"
        )
        assert exp_a.artifact_hash == exp_b.artifact_hash
        assert len(exp_a.artifact_hash) == 64

    def test_forensic_event_created(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes,
            _simple_predict_fn,
            features,
            inference_event_id="evt-6",
            actor="test_actor",
        )
        fe = exp.forensic_event
        assert fe is not None
        assert fe.event_type == EventType.EXPLANATION_GENERATED
        assert fe.actor == "test_actor"
        assert fe.event_hash != ""
        assert fe.verify()

    def test_forensic_event_payload(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-7"
        )
        payload = exp.forensic_event.payload
        assert payload["explanation_type"] == "SDD"
        assert payload["method"] == "spatial_directional_decomposition"
        assert payload["inference_event_id"] == "evt-7"
        assert payload["artifact_hash"] == exp.artifact_hash
        assert payload["dominant_region"] == exp.dominant_region
        assert payload["num_regions"] == 7
        assert len(payload["region_scores"]) == 7

    def test_provenance_node_created(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-8"
        )
        pn = exp.provenance_node
        assert pn is not None
        assert pn.entity_type == "explanation"
        assert pn.node_hash != ""
        assert pn.node_hash == pn.compute_hash()

    def test_provenance_node_metadata(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-9"
        )
        meta = exp.provenance_node.metadata
        assert meta["explanation_type"] == "SDD"
        assert meta["inference_event_id"] == "evt-9"
        assert meta["artifact_hash"] == exp.artifact_hash
        assert meta["dominant_region"] == exp.dominant_region

    def test_provenance_linked_to_parent(self, image_bytes, features):
        parent_id = str(uuid4())
        exp = generate_sdd_explanation(
            image_bytes,
            _simple_predict_fn,
            features,
            inference_event_id="evt-10",
            parent_provenance_id=parent_id,
        )
        pn = exp.provenance_node
        assert len(pn.parents) == 1
        assert str(pn.parents[0]) == parent_id
        assert pn.relations == [ProvenanceRelation.DERIVED_FROM]

    def test_provenance_no_parent(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-11"
        )
        assert exp.provenance_node.parents == []
        assert exp.provenance_node.relations == []

    def test_hash_chain_fields(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes,
            _simple_predict_fn,
            features,
            inference_event_id="evt-12",
            previous_hash="abc123",
            sequence_number=5,
        )
        fe = exp.forensic_event
        assert fe.previous_hash == "abc123"
        assert fe.sequence_number == 5

    def test_predicted_label_and_confidence(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-13"
        )
        assert exp.predicted_label == "person_A"
        assert isinstance(exp.original_confidence, float)
        assert 0.0 <= exp.original_confidence <= 1.0

    def test_constant_predict_fn_zero_gradients(self, image_bytes, features):
        """Constant predict_fn produces zero gradients -- should not crash."""
        exp = generate_sdd_explanation(
            image_bytes,
            _constant_predict_fn,
            features,
            inference_event_id="evt-14",
        )
        assert exp.predicted_label == "person_B"
        assert exp.original_confidence == 0.85
        # All regions should have zero importance
        for r in exp.regions:
            assert r.importance == 0.0
        assert len(exp.saliency_map) == 112

    def test_saliency_values_normalized(self, image_bytes, features):
        exp = generate_sdd_explanation(
            image_bytes, _simple_predict_fn, features, inference_event_id="evt-15"
        )
        flat = [v for row in exp.saliency_map for v in row]
        assert min(flat) >= 0.0
        assert max(flat) <= 1.0
