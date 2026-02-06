"""Tests for SVM classifier training pipeline."""

from __future__ import annotations

import pickle
from uuid import uuid4

import numpy as np
import pytest

from friendlyface.core.models import EventType, ProvenanceRelation
from friendlyface.recognition.svm import (
    SVMTrainingResult,
    _hash_model,
    train_svm,
)


def _make_features_labels(
    n_samples: int = 20,
    n_features: int = 5,
    n_classes: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic PCA-reduced features with integer labels."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_samples, n_features))
    labels = np.array([i % n_classes for i in range(n_samples)])
    return features, labels


class TestInputValidation:
    def test_features_must_be_2d(self, tmp_path):
        labels = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="2-D"):
            train_svm(np.array([1, 2, 3]), labels, tmp_path / "m.pkl")

    def test_labels_must_be_1d(self, tmp_path):
        features = np.ones((3, 5))
        with pytest.raises(ValueError, match="1-D"):
            train_svm(features, np.ones((3, 1)), tmp_path / "m.pkl")

    def test_length_mismatch_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mismatch"):
            train_svm(np.ones((3, 5)), np.array([0, 1]), tmp_path / "m.pkl")

    def test_empty_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            train_svm(np.ones((0, 5)), np.array([]), tmp_path / "m.pkl")

    def test_single_class_raises(self, tmp_path):
        with pytest.raises(ValueError, match="2 classes"):
            train_svm(np.ones((5, 3)), np.zeros(5), tmp_path / "m.pkl")


class TestHashModel:
    def test_deterministic(self):
        from sklearn.svm import SVC

        svm = SVC(C=1.0, kernel="linear")
        features, labels = _make_features_labels(n_samples=10, n_features=3)
        svm.fit(features, labels)
        h1 = _hash_model(svm)
        h2 = _hash_model(svm)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex


class TestTrainSVM:
    def test_basic_training(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output)

        assert isinstance(result, SVMTrainingResult)
        assert result.n_samples == 20
        assert result.n_features == 5
        assert result.n_classes == 2
        assert output.exists()

    def test_model_predicts_correct_shape(self, tmp_path):
        features, labels = _make_features_labels(n_samples=30, n_features=8, n_classes=3)
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output)

        predictions = result.model.predict(features[:5])
        assert predictions.shape == (5,)

    def test_multiclass(self, tmp_path):
        features, labels = _make_features_labels(n_samples=30, n_features=5, n_classes=4)
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output)

        assert result.n_classes == 4

    def test_custom_hyperparameters(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output, C=10.0, kernel="rbf")

        assert result.hyperparameters == {"C": 10.0, "kernel": "rbf"}

    def test_cv_accuracy_stored(self, tmp_path):
        features, labels = _make_features_labels(n_samples=20, n_features=5)
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output, cv_folds=3)

        assert 0.0 <= result.cv_accuracy <= 1.0
        assert len(result.cv_scores) == 3

    def test_cv_folds_clamped_to_min_class_count(self, tmp_path):
        """When cv_folds > min class count, clamp to min class count."""
        features, labels = _make_features_labels(n_samples=4, n_features=3, n_classes=2)
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output, cv_folds=100)

        # 4 samples, 2 classes → 2 per class → clamped to 2 folds
        assert len(result.cv_scores) == 2

    def test_serialized_model_loadable(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        train_svm(features, labels, output, C=2.0, kernel="linear")

        with open(output, "rb") as f:
            blob = pickle.load(f)

        assert "svm" in blob
        assert blob["hyperparameters"] == {"C": 2.0, "kernel": "linear"}
        assert blob["n_samples"] == 20
        assert blob["n_features"] == 5
        assert blob["n_classes"] == 2
        assert 0.0 <= blob["cv_accuracy"] <= 1.0
        assert blob["model_hash"]

    def test_model_hash_in_result(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output)

        assert result.model_hash
        assert len(result.model_hash) == 64
        assert result.model_hash == _hash_model(result.model)

    def test_forensic_event_created(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output)

        event = result.training_event
        assert event.event_type == EventType.TRAINING_COMPLETE
        assert event.event_hash != ""
        assert event.verify()
        assert event.payload["model_type"] == "SVM"
        assert event.payload["model_hash"] == result.model_hash
        assert event.payload["n_samples"] == 20
        assert event.payload["n_features"] == 5
        assert event.payload["n_classes"] == 2
        assert event.payload["cv_accuracy"] == result.cv_accuracy
        assert event.payload["hyperparameters"] == result.hyperparameters

    def test_forensic_event_chain(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        result = train_svm(
            features,
            labels,
            output,
            previous_hash="abc123",
            sequence_number=7,
        )

        assert result.training_event.previous_hash == "abc123"
        assert result.training_event.sequence_number == 7

    def test_provenance_node_created(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output)

        node = result.provenance_node
        assert node.entity_type == "training"
        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()
        assert node.metadata["model_type"] == "SVM"
        assert node.metadata["model_hash"] == result.model_hash
        assert node.metadata["cv_accuracy"] == result.cv_accuracy

    def test_provenance_node_links_to_pca(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"
        pca_node_id = uuid4()

        result = train_svm(
            features,
            labels,
            output,
            pca_provenance_id=pca_node_id,
        )

        node = result.provenance_node
        assert pca_node_id in node.parents
        assert len(node.relations) == 1
        assert node.relations[0] == ProvenanceRelation.DERIVED_FROM

    def test_provenance_no_parent_by_default(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "svm.pkl"

        result = train_svm(features, labels, output)

        assert result.provenance_node.parents == []
        assert result.provenance_node.relations == []

    def test_output_parent_dirs_created(self, tmp_path):
        features, labels = _make_features_labels()
        output = tmp_path / "nested" / "deep" / "svm.pkl"

        result = train_svm(features, labels, output)
        assert result.model_path.exists()
