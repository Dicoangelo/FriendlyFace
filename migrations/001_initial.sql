-- FriendlyFace: Initial Supabase migration
-- Creates all tables required by the Blockchain Forensic Layer.
-- Run via Supabase SQL editor or supabase db push.

-- forensic_events: Immutable, hash-chained event records
CREATE TABLE IF NOT EXISTS forensic_events (
    id UUID PRIMARY KEY,
    event_type TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    actor TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    previous_hash TEXT NOT NULL DEFAULT 'GENESIS',
    event_hash TEXT NOT NULL,
    sequence_number INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forensic_events_seq
    ON forensic_events (sequence_number);

-- provenance_nodes: DAG nodes tracking data lineage
CREATE TABLE IF NOT EXISTS provenance_nodes (
    id UUID PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    parents JSONB NOT NULL DEFAULT '[]',
    relations JSONB NOT NULL DEFAULT '[]',
    node_hash TEXT NOT NULL
);

-- forensic_bundles: Self-verifiable output artifacts
CREATE TABLE IF NOT EXISTS forensic_bundles (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    event_ids JSONB NOT NULL DEFAULT '[]',
    merkle_root TEXT NOT NULL DEFAULT '',
    merkle_proofs JSONB NOT NULL DEFAULT '[]',
    provenance_chain JSONB NOT NULL DEFAULT '[]',
    bias_audit TEXT,
    zk_proof_placeholder TEXT,
    did_credential_placeholder TEXT,
    bundle_hash TEXT NOT NULL DEFAULT ''
);

-- bias_audits: Demographic parity + equalized odds per event
CREATE TABLE IF NOT EXISTS bias_audits (
    id UUID PRIMARY KEY,
    event_id UUID REFERENCES forensic_events(id),
    timestamp TIMESTAMPTZ NOT NULL,
    demographic_parity_gap DOUBLE PRECISION NOT NULL,
    equalized_odds_gap DOUBLE PRECISION NOT NULL,
    groups_evaluated JSONB NOT NULL DEFAULT '[]',
    compliant BOOLEAN NOT NULL DEFAULT TRUE,
    details JSONB NOT NULL DEFAULT '{}'
);

-- Enable Row Level Security (RLS) on all tables.
-- Policies should be configured separately per deployment.
ALTER TABLE forensic_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE provenance_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE forensic_bundles ENABLE ROW LEVEL SECURITY;
ALTER TABLE bias_audits ENABLE ROW LEVEL SECURITY;

-- Allow service_role full access (needed for the backend).
CREATE POLICY "service_role_all_forensic_events" ON forensic_events
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "service_role_all_provenance_nodes" ON provenance_nodes
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "service_role_all_forensic_bundles" ON forensic_bundles
    FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "service_role_all_bias_audits" ON bias_audits
    FOR ALL USING (auth.role() = 'service_role');
