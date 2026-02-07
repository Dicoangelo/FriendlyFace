-- Migration 006: Merkle tree checkpoints for fast rebuild
CREATE TABLE IF NOT EXISTS merkle_checkpoints (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    leaf_count INTEGER NOT NULL,
    root_hash TEXT NOT NULL,
    leaves_json TEXT NOT NULL,
    event_index_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_merkle_checkpoints_leaf_count
    ON merkle_checkpoints (leaf_count DESC);
