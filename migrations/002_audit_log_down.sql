-- Down migration 002: Remove audit log table
DROP INDEX IF EXISTS idx_audit_log_action;
DROP INDEX IF EXISTS idx_audit_log_actor;
DROP INDEX IF EXISTS idx_audit_log_timestamp;
DROP TABLE IF EXISTS audit_log;
