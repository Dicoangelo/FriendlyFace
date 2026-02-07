"""Tests for multi-modal score-level fusion."""

import pytest
from uuid import uuid4

from friendlyface.core.models import EventType
from friendlyface.recognition.fusion import (
    FusedMatch,
    FusionResult,
    fuse_scores,
)


def _face_matches() -> list[dict]:
    return [
        {"label": "alice", "confidence": 0.9},
        {"label": "bob", "confidence": 0.6},
        {"label": "charlie", "confidence": 0.3},
    ]


def _voice_matches() -> list[dict]:
    return [
        {"label": "alice", "confidence": 0.8},
        {"label": "bob", "confidence": 0.5},
        {"label": "dave", "confidence": 0.7},
    ]


class TestWeightedFusion:
    def test_correct_fused_score(self):
        result = fuse_scores(_face_matches(), _voice_matches(), face_weight=0.6, voice_weight=0.4)
        alice = next(m for m in result.fused_matches if m.label == "alice")
        expected = 0.6 * 0.9 + 0.4 * 0.8  # 0.54 + 0.32 = 0.86
        assert abs(alice.fused_confidence - expected) < 1e-9

    def test_bob_fused_score(self):
        result = fuse_scores(_face_matches(), _voice_matches(), face_weight=0.6, voice_weight=0.4)
        bob = next(m for m in result.fused_matches if m.label == "bob")
        expected = 0.6 * 0.6 + 0.4 * 0.5  # 0.36 + 0.20 = 0.56
        assert abs(bob.fused_confidence - expected) < 1e-9

    def test_sorted_by_fused_confidence(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        confidences = [m.fused_confidence for m in result.fused_matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_all_labels_present(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        labels = {m.label for m in result.fused_matches}
        assert labels == {"alice", "bob", "charlie", "dave"}


class TestMissingModality:
    def test_face_only_label(self):
        """Charlie only appears in face matches."""
        result = fuse_scores(_face_matches(), _voice_matches(), face_weight=0.6, voice_weight=0.4)
        charlie = next(m for m in result.fused_matches if m.label == "charlie")
        expected = 0.6 * 0.3  # Only face contribution
        assert abs(charlie.fused_confidence - expected) < 1e-9
        assert charlie.face_confidence == 0.3
        assert charlie.voice_confidence is None

    def test_voice_only_label(self):
        """Dave only appears in voice matches."""
        result = fuse_scores(_face_matches(), _voice_matches(), face_weight=0.6, voice_weight=0.4)
        dave = next(m for m in result.fused_matches if m.label == "dave")
        expected = 0.4 * 0.7  # Only voice contribution
        assert abs(dave.fused_confidence - expected) < 1e-9
        assert dave.face_confidence is None
        assert dave.voice_confidence == 0.7

    def test_empty_face_matches(self):
        result = fuse_scores([], _voice_matches())
        assert len(result.fused_matches) == len(_voice_matches())
        for m in result.fused_matches:
            assert m.face_confidence is None

    def test_empty_voice_matches(self):
        result = fuse_scores(_face_matches(), [])
        assert len(result.fused_matches) == len(_face_matches())
        for m in result.fused_matches:
            assert m.voice_confidence is None

    def test_both_empty(self):
        result = fuse_scores([], [])
        assert result.fused_matches == []


class TestWeightValidation:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            fuse_scores(_face_matches(), _voice_matches(), face_weight=0.5, voice_weight=0.3)

    def test_valid_weights_accepted(self):
        result = fuse_scores(_face_matches(), _voice_matches(), face_weight=0.7, voice_weight=0.3)
        assert result.face_weight == 0.7
        assert result.voice_weight == 0.3

    def test_equal_weights(self):
        result = fuse_scores(_face_matches(), _voice_matches(), face_weight=0.5, voice_weight=0.5)
        alice = next(m for m in result.fused_matches if m.label == "alice")
        expected = 0.5 * 0.9 + 0.5 * 0.8
        assert abs(alice.fused_confidence - expected) < 1e-9


class TestFusionResult:
    def test_fusion_method(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        assert result.fusion_method == "weighted_sum"

    def test_fused_match_type(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        for m in result.fused_matches:
            assert isinstance(m, FusedMatch)

    def test_returns_fusion_result(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        assert isinstance(result, FusionResult)


class TestForensicEvent:
    def test_event_created_and_sealed(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        event = result.forensic_event
        assert event.event_type == EventType.INFERENCE_RESULT
        assert event.event_hash != ""
        assert event.verify()

    def test_event_tracks_both_modalities(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        payload = result.forensic_event.payload
        assert payload["fusion_method"] == "weighted_sum"
        assert payload["n_face_matches"] == 3
        assert payload["n_voice_matches"] == 3
        assert payload["n_fused_matches"] == 4
        assert payload["face_weight"] == 0.6
        assert payload["voice_weight"] == 0.4

    def test_event_contains_fused_matches(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        fused_in_payload = result.forensic_event.payload["fused_matches"]
        assert len(fused_in_payload) == 4
        labels = {m["label"] for m in fused_in_payload}
        assert "alice" in labels

    def test_custom_actor(self):
        result = fuse_scores(_face_matches(), _voice_matches(), actor="test_fusion")
        assert result.forensic_event.actor == "test_fusion"


class TestProvenanceNode:
    def test_provenance_created_and_sealed(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        node = result.provenance_node
        assert node.entity_type == "fusion"
        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()

    def test_provenance_links_face_parent(self):
        face_id = uuid4()
        result = fuse_scores(_face_matches(), _voice_matches(), face_provenance_id=face_id)
        assert face_id in result.provenance_node.parents

    def test_provenance_links_voice_parent(self):
        voice_id = uuid4()
        result = fuse_scores(_face_matches(), _voice_matches(), voice_provenance_id=voice_id)
        assert voice_id in result.provenance_node.parents

    def test_provenance_links_both_parents(self):
        face_id = uuid4()
        voice_id = uuid4()
        result = fuse_scores(
            _face_matches(),
            _voice_matches(),
            face_provenance_id=face_id,
            voice_provenance_id=voice_id,
        )
        node = result.provenance_node
        assert face_id in node.parents
        assert voice_id in node.parents
        assert len(node.parents) == 2
        assert len(node.relations) == 2

    def test_provenance_no_parents_by_default(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        assert result.provenance_node.parents == []

    def test_provenance_metadata(self):
        result = fuse_scores(_face_matches(), _voice_matches())
        meta = result.provenance_node.metadata
        assert meta["fusion_method"] == "weighted_sum"
        assert meta["face_weight"] == 0.6
        assert meta["voice_weight"] == 0.4
        assert meta["n_fused_matches"] == 4
