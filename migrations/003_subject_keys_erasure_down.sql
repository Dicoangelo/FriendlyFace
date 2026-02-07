-- Down migration 003: Remove subject keys and erasure records
DROP INDEX IF EXISTS idx_erasure_records_status;
DROP INDEX IF EXISTS idx_erasure_records_subject;
DROP TABLE IF EXISTS erasure_records;
DROP TABLE IF EXISTS subject_keys;
