-- Migration 013: Recognition results for real demographic auditing
CREATE TABLE IF NOT EXISTS recognition_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    demographic_group TEXT NOT NULL,
    true_label INTEGER,
    predicted_label INTEGER NOT NULL,
    confidence REAL NOT NULL,
    correct INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_recognition_results_group
    ON recognition_results (demographic_group);

CREATE INDEX IF NOT EXISTS idx_recognition_results_created
    ON recognition_results (created_at DESC);
