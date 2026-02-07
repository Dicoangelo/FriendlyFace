-- Migration 003: Subject encryption keys and erasure records for GDPR Art 17
CREATE TABLE IF NOT EXISTS subject_keys (
    subject_id TEXT PRIMARY KEY,
    encrypted_key BLOB NOT NULL,
    key_nonce BLOB NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS erasure_records (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    tables_affected TEXT NOT NULL DEFAULT '[]',
    event_count INTEGER NOT NULL DEFAULT 0,
    method TEXT NOT NULL DEFAULT 'key_deletion'
);

CREATE INDEX IF NOT EXISTS idx_erasure_records_subject
    ON erasure_records (subject_id);

CREATE INDEX IF NOT EXISTS idx_erasure_records_status
    ON erasure_records (status);
