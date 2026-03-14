CREATE TABLE IF NOT EXISTS seals (
    id TEXT PRIMARY KEY,
    system_id TEXT NOT NULL,
    system_name TEXT NOT NULL,
    assessment_scope TEXT DEFAULT 'full',
    credential TEXT NOT NULL,
    compliance_score REAL NOT NULL,
    compliance_summary TEXT NOT NULL,
    merkle_root TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    revocation_reason TEXT,
    previous_seal_id TEXT,
    bundle_ids TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_seals_system_id ON seals(system_id);
CREATE INDEX IF NOT EXISTS idx_seals_status ON seals(status);
