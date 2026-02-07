-- Down migration 006: Remove Merkle checkpoints
DROP INDEX IF EXISTS idx_merkle_checkpoints_leaf_count;
DROP TABLE IF EXISTS merkle_checkpoints;
