CREATE TABLE IF NOT EXISTS anchors (
    id TEXT PRIMARY KEY,
    merkle_root TEXT NOT NULL,
    chain TEXT NOT NULL,
    tx_hash TEXT NOT NULL,
    block_number INTEGER NOT NULL,
    anchored_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_anchors_merkle_root ON anchors(merkle_root);
