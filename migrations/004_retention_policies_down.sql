-- Down migration 004: Remove retention policies
DROP INDEX IF EXISTS idx_retention_policies_entity;
DROP TABLE IF EXISTS retention_policies;
