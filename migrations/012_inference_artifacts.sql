-- Migration 012: Inference artifacts for explainability
CREATE TABLE IF NOT EXISTS inference_artifacts (
    event_id TEXT PRIMARY KEY,
    image_bytes BLOB NOT NULL,
    feature_vector BLOB,
    pca_model_path TEXT,
    svm_model_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
