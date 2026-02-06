"""Tests for SHAP explainability module.

Tests use synthetic feature vectors and a mock prediction function
to verify SHAP output structure, forensic event logging, and
provenance chain creation.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from friendlyface.core.models import EventType, ProvenanceRelation
from friendlyface.explainability.shap_explain import (
    ShapExplanation,
    _kernel_shap,
    generate_shap_explanation,
)


# Number of PCA features in test scenarios
N_FEATURES = 20


def _mock_predict_fn(features: np.ndarray) -> np.ndarray:
    """Mock prediction function that returns a linear score based on features.

    Higher values in early feature dimensions increase the score, so SHAP
    values should attribute more importance to those dimensions.
    """
    # Weighted sum: feature 0 has weight 0.5, feature 1 has 0.3, rest ~0
    weights = np.zeros(features.shape[1])
    if features.shape[1] > 0:
        weights[0] = 0.5
    if features.shape[1] > 1:
        weights[1] = 0.3
    if features.shape[1] > 2:
        weights[2] = 0.1
    return features @ weights


def _make_feature_vector(seed: int = 42) -> np.ndarray:
    """Create a synthetic PCA feature vector."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N_FEATURES)


def _make_background(n_samples: int = 10, seed: int = 0) -> np.ndarray:
    """Create a synthetic background dataset of PCA feature vectors."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, N_FEATURES))


# ---------------------------------------------------------------------------
# Unit tests: _kernel_shap
# ---------------------------------------------------------------------------


class TestKernelShap:
    def test_returns_correct_shapes(self):
        instance = _make_feature_vector()
        background = _make_background()
        shap_values, base_value = _kernel_shap(
            _mock_predict_fn, instance, background, num_samples=50
        )
        assert shap_values.shape == (N_FEATURES,)
        assert isinstance(base_value, float)

    def test_base_value_is_mean_prediction(self):
        instance = _make_feature_vector()
        background = _make_background()
        _, base_value = _kernel_shap(_mock_predict_fn, instance, background, num_samples=50)
        expected_base = float(np.mean(_mock_predict_fn(background)))
        assert abs(base_value - expected_base) < 1e-10

    def test_deterministic_with_same_seed(self):
        instance = _make_feature_vector()
        background = _make_background()
        sv1, bv1 = _kernel_shap(
            _mock_predict_fn,
            instance,
            background,
            num_samples=50,
            random_state=42,
        )
        sv2, bv2 = _kernel_shap(
            _mock_predict_fn,
            instance,
            background,
            num_samples=50,
            random_state=42,
        )
        np.testing.assert_array_equal(sv1, sv2)
        assert bv1 == bv2

    def test_different_seeds_differ(self):
        instance = _make_feature_vector()
        background = _make_background()
        sv1, _ = _kernel_shap(
            _mock_predict_fn,
            instance,
            background,
            num_samples=50,
            random_state=1,
        )
        sv2, _ = _kernel_shap(
            _mock_predict_fn,
            instance,
            background,
            num_samples=50,
            random_state=99,
        )
        assert not np.array_equal(sv1, sv2)


# ---------------------------------------------------------------------------
# Unit tests: ShapExplanation artifact
# ---------------------------------------------------------------------------


class TestShapExplanationArtifact:
    def test_artifact_hash_deterministic(self):
        inference_id = uuid4()
        explanation = ShapExplanation(
            inference_event_id=inference_id,
            predicted_label=1,
            original_confidence=0.85,
            shap_values=np.ones(N_FEATURES, dtype=np.float64),
            feature_importance_ranking=list(range(N_FEATURES)),
            base_value=0.3,
            num_samples=128,
        )
        h1 = explanation.compute_artifact_hash()
        h2 = explanation.compute_artifact_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_artifact_hash_changes_with_data(self):
        inference_id = uuid4()
        exp1 = ShapExplanation(
            inference_event_id=inference_id,
            predicted_label=1,
            original_confidence=0.85,
            shap_values=np.ones(N_FEATURES, dtype=np.float64),
            feature_importance_ranking=list(range(N_FEATURES)),
            base_value=0.3,
            num_samples=128,
        )
        exp2 = ShapExplanation(
            inference_event_id=inference_id,
            predicted_label=2,  # different label
            original_confidence=0.85,
            shap_values=np.ones(N_FEATURES, dtype=np.float64),
            feature_importance_ranking=list(range(N_FEATURES)),
            base_value=0.3,
            num_samples=128,
        )
        assert exp1.compute_artifact_hash() != exp2.compute_artifact_hash()

    def test_artifact_hash_changes_with_shap_values(self):
        inference_id = uuid4()
        sv1 = np.ones(N_FEATURES, dtype=np.float64)
        sv2 = np.zeros(N_FEATURES, dtype=np.float64)
        exp1 = ShapExplanation(
            inference_event_id=inference_id,
            predicted_label=1,
            original_confidence=0.85,
            shap_values=sv1,
            feature_importance_ranking=list(range(N_FEATURES)),
            base_value=0.3,
            num_samples=128,
        )
        exp2 = ShapExplanation(
            inference_event_id=inference_id,
            predicted_label=1,
            original_confidence=0.85,
            shap_values=sv2,
            feature_importance_ranking=list(range(N_FEATURES)),
            base_value=0.3,
            num_samples=128,
        )
        assert exp1.compute_artifact_hash() != exp2.compute_artifact_hash()


# ---------------------------------------------------------------------------
# Unit tests: generate_shap_explanation
# ---------------------------------------------------------------------------


class TestGenerateShapExplanation:
    """Tests for the full SHAP explanation generation pipeline."""

    def _run(
        self,
        *,
        num_samples: int = 50,
        inference_provenance_id: uuid4 | None = None,
        previous_hash: str = "GENESIS",
        sequence_number: int = 0,
    ) -> ShapExplanation:
        """Helper to run generate_shap_explanation with test data."""
        feature_vector = _make_feature_vector()
        background = _make_background()
        inference_event_id = uuid4()

        return generate_shap_explanation(
            feature_vector,
            _mock_predict_fn,
            background,
            inference_event_id=inference_event_id,
            predicted_label=1,
            original_confidence=0.7,
            num_samples=num_samples,
            inference_provenance_id=inference_provenance_id,
            previous_hash=previous_hash,
            sequence_number=sequence_number,
        )

    def test_returns_shap_explanation(self):
        explanation = self._run()
        assert isinstance(explanation, ShapExplanation)

    def test_shap_values_shape(self):
        explanation = self._run()
        assert explanation.shap_values.shape == (N_FEATURES,)

    def test_shap_values_per_feature(self):
        """SHAP values are computed per feature dimension."""
        explanation = self._run()
        assert len(explanation.shap_values) == N_FEATURES
        # Each feature has a SHAP value (may be zero for irrelevant features)
        assert explanation.shap_values.dtype == np.float64

    def test_feature_importance_ranking_length(self):
        explanation = self._run()
        assert len(explanation.feature_importance_ranking) == N_FEATURES

    def test_feature_importance_ranking_sorted_by_abs_shap(self):
        explanation = self._run()
        abs_shap = np.abs(explanation.shap_values)
        ranked_abs = [abs_shap[i] for i in explanation.feature_importance_ranking]
        assert ranked_abs == sorted(ranked_abs, reverse=True)

    def test_feature_importance_ranking_contains_all_indices(self):
        explanation = self._run()
        assert sorted(explanation.feature_importance_ranking) == list(range(N_FEATURES))

    def test_base_value_is_float(self):
        explanation = self._run()
        assert isinstance(explanation.base_value, float)

    def test_artifact_hash_set(self):
        explanation = self._run()
        assert explanation.artifact_hash != ""
        assert len(explanation.artifact_hash) == 64

    def test_artifact_hash_matches_recompute(self):
        explanation = self._run()
        assert explanation.artifact_hash == explanation.compute_artifact_hash()

    # -------------------------------------------------------------------
    # Forensic event tests
    # -------------------------------------------------------------------

    def test_forensic_event_created(self):
        explanation = self._run()
        event = explanation.forensic_event
        assert event is not None
        assert event.event_type == EventType.EXPLANATION_GENERATED
        assert event.event_hash != ""
        assert event.verify()

    def test_forensic_event_payload_type(self):
        explanation = self._run()
        payload = explanation.forensic_event.payload
        assert payload["explanation_type"] == "SHAP"

    def test_forensic_event_payload_method(self):
        explanation = self._run()
        payload = explanation.forensic_event.payload
        assert payload["method"] == "shap"

    def test_forensic_event_payload_artifact_hash(self):
        explanation = self._run()
        payload = explanation.forensic_event.payload
        assert payload["artifact_hash"] == explanation.artifact_hash

    def test_forensic_event_payload_predicted_label(self):
        explanation = self._run()
        payload = explanation.forensic_event.payload
        assert payload["predicted_label"] == 1

    def test_forensic_event_payload_base_value(self):
        explanation = self._run()
        payload = explanation.forensic_event.payload
        assert "base_value" in payload
        assert isinstance(payload["base_value"], float)

    def test_forensic_event_payload_top_features(self):
        explanation = self._run()
        payload = explanation.forensic_event.payload
        assert "top_features" in payload
        top_features = payload["top_features"]
        assert len(top_features) <= 10
        for feat in top_features:
            assert "feature_index" in feat
            assert "shap_value" in feat

    def test_forensic_event_payload_num_features(self):
        explanation = self._run()
        payload = explanation.forensic_event.payload
        assert payload["num_features"] == N_FEATURES

    def test_forensic_event_chain_params(self):
        explanation = self._run(
            previous_hash="prev_abc",
            sequence_number=7,
        )
        event = explanation.forensic_event
        assert event.previous_hash == "prev_abc"
        assert event.sequence_number == 7

    # -------------------------------------------------------------------
    # Provenance node tests
    # -------------------------------------------------------------------

    def test_provenance_node_created(self):
        explanation = self._run()
        node = explanation.provenance_node
        assert node is not None
        assert node.entity_type == "explanation"
        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()

    def test_provenance_node_metadata(self):
        explanation = self._run()
        meta = explanation.provenance_node.metadata
        assert meta["explanation_type"] == "SHAP"
        assert meta["method"] == "shap"
        assert meta["artifact_hash"] == explanation.artifact_hash
        assert meta["predicted_label"] == 1
        assert meta["num_features"] == N_FEATURES

    def test_provenance_node_with_parent(self):
        parent_id = uuid4()
        explanation = self._run(inference_provenance_id=parent_id)
        node = explanation.provenance_node
        assert parent_id in node.parents
        assert ProvenanceRelation.DERIVED_FROM in node.relations

    def test_provenance_node_no_parent(self):
        explanation = self._run(inference_provenance_id=None)
        node = explanation.provenance_node
        assert node.parents == []
        assert node.relations == []

    # -------------------------------------------------------------------
    # Explanation content tests
    # -------------------------------------------------------------------

    def test_predicted_label_preserved(self):
        explanation = self._run()
        assert explanation.predicted_label == 1

    def test_original_confidence_preserved(self):
        explanation = self._run()
        assert explanation.original_confidence == 0.7

    def test_num_samples_preserved(self):
        explanation = self._run(num_samples=64)
        assert explanation.num_samples == 64

    def test_explanation_includes_shap_values(self):
        """Verify the explanation artifact includes SHAP values."""
        explanation = self._run()
        assert explanation.shap_values is not None
        assert isinstance(explanation.shap_values, np.ndarray)

    def test_explanation_includes_feature_importance_ranking(self):
        """Verify the explanation artifact includes feature importance ranking."""
        explanation = self._run()
        assert explanation.feature_importance_ranking is not None
        assert isinstance(explanation.feature_importance_ranking, list)

    def test_explanation_includes_base_value(self):
        """Verify the explanation artifact includes base value."""
        explanation = self._run()
        assert explanation.base_value is not None
