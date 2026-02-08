-- Migration 008: Persistent explanation records
CREATE TABLE IF NOT EXISTS explanation_records (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    method TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_explanation_records_event
    ON explanation_records (event_id);

CREATE INDEX IF NOT EXISTS idx_explanation_records_method
    ON explanation_records (method);
