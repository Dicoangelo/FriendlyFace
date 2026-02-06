"""SVM classifier training pipeline for PCA-reduced face features.

Trains a linear SVM on PCA-reduced face embeddings with cross-validation,
and serializes the model with forensic logging of all training events.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from friendlyface.core.models import EventType, ForensicEvent, ProvenanceNode, ProvenanceRelation


def _hash_model(model: SVC) -> str:
    """Compute a SHA-256 digest of the serialized model."""
    return hashlib.sha256(pickle.dumps(model)).hexdigest()


class SVMTrainingResult:
    """Container for SVM training outputs."""

    def __init__(
        self,
        model: SVC,
        model_path: Path,
        model_hash: str,
        n_samples: int,
        n_features: int,
        n_classes: int,
        hyperparameters: dict[str, Any],
        cv_accuracy: float,
        cv_scores: np.ndarray,
        training_event: ForensicEvent,
        provenance_node: ProvenanceNode,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.model_hash = model_hash
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.hyperparameters = hyperparameters
        self.cv_accuracy = cv_accuracy
        self.cv_scores = cv_scores
        self.training_event = training_event
        self.provenance_node = provenance_node


def train_svm(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    C: float = 1.0,
    kernel: str = "linear",
    cv_folds: int = 5,
    actor: str = "svm_trainer",
    previous_hash: str = "GENESIS",
    sequence_number: int = 0,
    pca_provenance_id: Any | None = None,
) -> SVMTrainingResult:
    """Train an SVM classifier on PCA-reduced features and serialize it.

    Parameters
    ----------
    features:
        PCA-reduced feature matrix of shape (n_samples, n_features).
    labels:
        Integer label array of shape (n_samples,).
    output_path:
        Where to write the serialized SVM model (.pkl).
    C:
        SVM regularization parameter.
    kernel:
        SVM kernel type (e.g., 'linear', 'rbf').
    cv_folds:
        Number of cross-validation folds.
    actor:
        Actor identity for forensic event logging.
    previous_hash:
        Previous hash in the event chain (for chaining).
    sequence_number:
        Sequence position in the event chain.
    pca_provenance_id:
        Optional UUID of a parent PCA training provenance node.

    Returns
    -------
    SVMTrainingResult with the trained model, metrics, forensic event,
    and provenance node.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got {features.ndim}-D")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got {labels.ndim}-D")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"features and labels length mismatch: {features.shape[0]} vs {labels.shape[0]}"
        )
    if features.shape[0] == 0:
        raise ValueError("Cannot train on empty dataset")

    n_samples, n_features = features.shape
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}")

    hyperparameters = {"C": C, "kernel": kernel}

    # Train SVM
    svm = SVC(C=C, kernel=kernel)
    svm.fit(features, labels)

    # Cross-validation (StratifiedKFold requires folds <= min class count)
    _, class_counts = np.unique(labels, return_counts=True)
    min_class_count = int(np.min(class_counts))
    actual_folds = min(cv_folds, min_class_count)
    cv_scores = cross_val_score(svm, features, labels, cv=actual_folds)
    cv_accuracy = float(np.mean(cv_scores))

    # Compute model hash for forensic traceability
    model_hash = _hash_model(svm)

    # Serialize model with metadata
    model_blob = {
        "svm": svm,
        "hyperparameters": hyperparameters,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "cv_accuracy": cv_accuracy,
        "cv_scores": cv_scores.tolist(),
        "model_hash": model_hash,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model_blob, f)

    # Log forensic training event
    training_event = ForensicEvent(
        event_type=EventType.TRAINING_COMPLETE,
        actor=actor,
        payload={
            "model_type": "SVM",
            "model_hash": model_hash,
            "hyperparameters": hyperparameters,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "cv_accuracy": cv_accuracy,
            "cv_scores": cv_scores.tolist(),
            "output_path": str(output_path),
        },
        previous_hash=previous_hash,
        sequence_number=sequence_number,
    ).seal()

    # Create provenance node linked to PCA training node
    parent_ids = []
    relations = []
    if pca_provenance_id is not None:
        parent_ids = [pca_provenance_id]
        relations = [ProvenanceRelation.DERIVED_FROM]

    provenance_node = ProvenanceNode(
        entity_type="training",
        entity_id=str(output_path),
        metadata={
            "model_type": "SVM",
            "model_hash": model_hash,
            "hyperparameters": hyperparameters,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
            "cv_accuracy": cv_accuracy,
        },
        parents=parent_ids,
        relations=relations,
    ).seal()

    return SVMTrainingResult(
        model=svm,
        model_path=output_path,
        model_hash=model_hash,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        hyperparameters=hyperparameters,
        cv_accuracy=cv_accuracy,
        cv_scores=cv_scores,
        training_event=training_event,
        provenance_node=provenance_node,
    )
