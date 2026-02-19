-- Migration 016: User-managed API keys
CREATE TABLE IF NOT EXISTS ff_api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES ff_users(id),
    key_hash TEXT NOT NULL,
    name TEXT,
    rate_limit INTEGER DEFAULT 100,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    revoked_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_ff_api_keys_user_id ON ff_api_keys (user_id);
CREATE INDEX IF NOT EXISTS idx_ff_api_keys_key_hash ON ff_api_keys (key_hash);
