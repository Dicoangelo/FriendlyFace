-- Migration 011: Persistent compliance reports
CREATE TABLE IF NOT EXISTS compliance_reports (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_compliance_reports_created
    ON compliance_reports (created_at DESC);
