-- Migration 005: Persistent DID key storage
CREATE TABLE IF NOT EXISTS did_keys (
    did TEXT PRIMARY KEY,
    public_key BLOB NOT NULL,
    encrypted_private_key BLOB,
    key_type TEXT NOT NULL DEFAULT 'Ed25519',
    created_at TEXT NOT NULL,
    label TEXT,
    is_platform_key INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_did_keys_platform
    ON did_keys (is_platform_key);
