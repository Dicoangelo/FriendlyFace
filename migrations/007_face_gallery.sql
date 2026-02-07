-- Migration 007: Face gallery for deep embedding recognition
CREATE TABLE IF NOT EXISTS face_gallery (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL DEFAULT 512,
    model_version TEXT NOT NULL,
    quality_score REAL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_face_gallery_subject
    ON face_gallery (subject_id);

CREATE INDEX IF NOT EXISTS idx_face_gallery_model
    ON face_gallery (model_version);
