-- Migration 004: Data retention policies
CREATE TABLE IF NOT EXISTS retention_policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    retention_days INTEGER NOT NULL,
    action TEXT NOT NULL DEFAULT 'erase',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retention_policies_entity
    ON retention_policies (entity_type);
