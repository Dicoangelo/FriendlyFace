"""PCA dimensionality reduction training pipeline for face images.

Trains a PCA model on aligned grayscale face images (112x112) and
serializes it with forensic logging of all training events.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from friendlyface.core.models import EventType, ForensicEvent, ProvenanceNode, ProvenanceRelation


IMAGE_SIZE = (112, 112)
EXPECTED_PIXELS = IMAGE_SIZE[0] * IMAGE_SIZE[1]


def _hash_directory(image_dir: Path) -> str:
    """Compute a SHA-256 digest over sorted file contents in a directory."""
    h = hashlib.sha256()
    for fp in sorted(image_dir.iterdir()):
        if fp.is_file():
            h.update(fp.read_bytes())
    return h.hexdigest()


def _load_images(image_dir: Path) -> np.ndarray:
    """Load aligned grayscale face images from a directory.

    Each image is converted to grayscale, verified to be 112x112,
    and flattened into a 1-D vector of 12544 pixels.

    Returns an (N, 12544) float64 array.
    """
    vectors: list[np.ndarray] = []
    for fp in sorted(image_dir.iterdir()):
        if not fp.is_file():
            continue
        img = Image.open(fp).convert("L")
        if img.size != IMAGE_SIZE:
            raise ValueError(
                f"Image {fp.name} has size {img.size}, expected {IMAGE_SIZE}"
            )
        vectors.append(np.asarray(img, dtype=np.float64).ravel())

    if not vectors:
        raise ValueError(f"No images found in {image_dir}")

    return np.stack(vectors)


class PCATrainingResult:
    """Container for PCA training outputs."""

    def __init__(
        self,
        model: PCA,
        model_path: Path,
        explained_variance: np.ndarray,
        explained_variance_ratio: np.ndarray,
        n_components: int,
        n_samples: int,
        dataset_hash: str,
        training_event: ForensicEvent,
        provenance_node: ProvenanceNode,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio
        self.n_components = n_components
        self.n_samples = n_samples
        self.dataset_hash = dataset_hash
        self.training_event = training_event
        self.provenance_node = provenance_node


def train_pca(
    image_dir: Path,
    output_path: Path,
    n_components: int = 128,
    *,
    actor: str = "pca_trainer",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    dataset_provenance_id: Any | None = None,
) -> PCATrainingResult:
    """Train a PCA model on aligned face images and serialize it.

    Parameters
    ----------
    image_dir:
        Directory of aligned grayscale 112x112 face images.
    output_path:
        Where to write the serialized PCA model (.pkl).
    n_components:
        Number of principal components to retain.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain (for chaining).
    sequence_number:
        Sequence position in the event chain.
    dataset_provenance_id:
        Optional UUID of a parent dataset provenance node.

    Returns
    -------
    PCATrainingResult with the trained model, metadata, forensic event,
    and provenance node.
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)

    # Hash the dataset for forensic traceability
    dataset_hash = _hash_directory(image_dir)

    # Load and validate images
    data = _load_images(image_dir)
    n_samples = data.shape[0]

    # Clamp n_components to feasible range
    max_components = min(n_samples, EXPECTED_PIXELS)
    actual_components = min(n_components, max_components)

    # Train PCA
    pca = PCA(n_components=actual_components)
    pca.fit(data)

    # Serialize model with explained variance metadata
    model_blob = {
        "pca": pca,
        "n_components": actual_components,
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_samples": n_samples,
        "dataset_hash": dataset_hash,
        "image_size": IMAGE_SIZE,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_blob, f)

    # Log forensic training event
    training_event = ForensicEvent(
        event_type=EventType.TRAINING_COMPLETE,
        actor=actor,
        payload={
            "model_type": "PCA",
            "n_components": actual_components,
            "n_samples": n_samples,
            "dataset_hash": dataset_hash,
            "output_path": str(output_path),
            "explained_variance_ratio_sum": float(
                np.sum(pca.explained_variance_ratio_)
            ),
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # Create provenance node for this training run
    parent_ids = []
    relations = []
    if dataset_provenance_id is not None:
        parent_ids = [dataset_provenance_id]
        relations = [ProvenanceRelation.DERIVED_FROM]

    provenance_node = ProvenanceNode(
        entity_type="training",
        entity_id=str(output_path),
        metadata={
            "model_type": "PCA",
            "n_components": actual_components,
            "dataset_hash": dataset_hash,
            "n_samples": n_samples,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    return PCATrainingResult(
        model=pca,
        model_path=output_path,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        n_components=actual_components,
        n_samples=n_samples,
        dataset_hash=dataset_hash,
        training_event=training_event,
        provenance_node=provenance_node,
    )
