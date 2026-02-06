"""Tests for PCA training pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image

from friendlyface.core.models import EventType
from friendlyface.recognition.pca import (
    EXPECTED_PIXELS,
    IMAGE_SIZE,
    PCATrainingResult,
    _hash_directory,
    _load_images,
    train_pca,
)


def _create_test_images(directory: Path, count: int = 5) -> None:
    """Create synthetic 112x112 grayscale images for testing."""
    rng = np.random.default_rng(42)
    for i in range(count):
        pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
        img = Image.fromarray(pixels, mode="L")
        img.save(directory / f"face_{i:03d}.png")


class TestLoadImages:
    def test_loads_correct_shape(self, tmp_path):
        _create_test_images(tmp_path, count=3)
        data = _load_images(tmp_path)
        assert data.shape == (3, EXPECTED_PIXELS)
        assert data.dtype == np.float64

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No images found"):
            _load_images(tmp_path)

    def test_wrong_size_raises(self, tmp_path):
        img = Image.fromarray(np.zeros((64, 64), dtype=np.uint8), mode="L")
        img.save(tmp_path / "wrong.png")
        with pytest.raises(ValueError, match="expected \\(112, 112\\)"):
            _load_images(tmp_path)


class TestHashDirectory:
    def test_deterministic(self, tmp_path):
        _create_test_images(tmp_path, count=2)
        h1 = _hash_directory(tmp_path)
        h2 = _hash_directory(tmp_path)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        _create_test_images(d1, count=2)
        _create_test_images(d2, count=3)
        assert _hash_directory(d1) != _hash_directory(d2)


class TestTrainPCA:
    def test_basic_training(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=10)
        output = tmp_path / "model.pkl"

        result = train_pca(img_dir, output, n_components=5)

        assert isinstance(result, PCATrainingResult)
        assert result.n_components == 5
        assert result.n_samples == 10
        assert output.exists()

    def test_output_dimensions(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=8)
        output = tmp_path / "pca.pkl"

        result = train_pca(img_dir, output, n_components=4)

        # PCA should produce n_components eigenvalues
        assert len(result.explained_variance) == 4
        assert len(result.explained_variance_ratio) == 4
        # Transform a sample to verify output dimension
        sample = np.random.default_rng(0).random((1, EXPECTED_PIXELS))
        transformed = result.model.transform(sample)
        assert transformed.shape == (1, 4)

    def test_components_clamped_to_n_samples(self, tmp_path):
        """When n_components > n_samples, clamp to n_samples."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=3)
        output = tmp_path / "model.pkl"

        result = train_pca(img_dir, output, n_components=1000)
        assert result.n_components == 3  # clamped to n_samples

    def test_serialized_model_loadable(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=5)
        output = tmp_path / "model.pkl"

        train_pca(img_dir, output, n_components=3)

        with open(output, "rb") as f:
            blob = pickle.load(f)

        assert "pca" in blob
        assert blob["n_components"] == 3
        assert blob["n_samples"] == 5
        assert blob["dataset_hash"]
        assert blob["image_size"] == IMAGE_SIZE
        assert len(blob["explained_variance"]) == 3
        assert len(blob["explained_variance_ratio"]) == 3

    def test_forensic_event_created(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=5)
        output = tmp_path / "model.pkl"

        result = train_pca(img_dir, output, n_components=3)

        event = result.training_event
        assert event.event_type == EventType.TRAINING_COMPLETE
        assert event.event_hash != ""
        assert event.verify()
        assert event.payload["model_type"] == "PCA"
        assert event.payload["n_components"] == 3
        assert event.payload["n_samples"] == 5
        assert event.payload["dataset_hash"] == result.dataset_hash

    def test_forensic_event_chain(self, tmp_path):
        """Training event respects hash chaining parameters."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=5)
        output = tmp_path / "model.pkl"

        result = train_pca(
            img_dir,
            output,
            n_components=3,
            previous_hash="abc123",
            sequence_number=7,
        )

        assert result.training_event.previous_hash == "abc123"
        assert result.training_event.sequence_number == 7

    def test_provenance_node_created(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=5)
        output = tmp_path / "model.pkl"

        result = train_pca(img_dir, output, n_components=3)

        node = result.provenance_node
        assert node.entity_type == "training"
        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()
        assert node.metadata["model_type"] == "PCA"
        assert node.metadata["dataset_hash"] == result.dataset_hash

    def test_provenance_node_with_parent(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=5)
        output = tmp_path / "model.pkl"
        parent_id = uuid4()

        result = train_pca(
            img_dir,
            output,
            n_components=3,
            dataset_provenance_id=parent_id,
        )

        node = result.provenance_node
        assert parent_id in node.parents
        assert len(node.relations) == 1

    def test_dataset_hash_in_payload(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=5)
        output = tmp_path / "model.pkl"

        result = train_pca(img_dir, output, n_components=3)

        assert result.dataset_hash
        assert result.training_event.payload["dataset_hash"] == result.dataset_hash
        assert result.provenance_node.metadata["dataset_hash"] == result.dataset_hash

    def test_output_parent_dirs_created(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _create_test_images(img_dir, count=5)
        output = tmp_path / "nested" / "deep" / "model.pkl"

        result = train_pca(img_dir, output, n_components=3)
        assert result.model_path.exists()
