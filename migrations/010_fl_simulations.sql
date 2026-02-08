-- Migration 010: Persistent FL simulation results
CREATE TABLE IF NOT EXISTS fl_simulations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_fl_simulations_created
    ON fl_simulations (created_at DESC);
