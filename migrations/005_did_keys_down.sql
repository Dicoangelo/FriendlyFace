-- Down migration 005: Remove DID key storage
DROP INDEX IF EXISTS idx_did_keys_platform;
DROP TABLE IF EXISTS did_keys;
