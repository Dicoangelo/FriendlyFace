-- Migration 002: Audit log table for operator actions
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    ip_address TEXT,
    user_agent TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp
    ON audit_log (timestamp);

CREATE INDEX IF NOT EXISTS idx_audit_log_actor
    ON audit_log (actor);

CREATE INDEX IF NOT EXISTS idx_audit_log_action
    ON audit_log (action);
