/**
 * Demo data for FriendlyFace dashboard — used when no backend is available.
 * Provides realistic mock responses for all key endpoints.
 */

const minutesAgo = (m: number) =>
    new Date(Date.now() - m * 60_000).toISOString();

// ── Health ──────────────────────────────────────────────
export const demoHealth = {
    status: "healthy",
    version: "0.1.0",
    storage: "demo",
    uptime_seconds: 86400,
};

// ── Dashboard ───────────────────────────────────────────
export const demoDashboard = {
    uptime_seconds: 86400,
    storage_backend: "demo",
    total_events: 1247,
    total_bundles: 38,
    total_provenance_nodes: 156,
    events_by_type: {
        recognition_inference: 412,
        model_training: 89,
        bias_audit: 67,
        consent_grant: 203,
        bundle_creation: 38,
        fl_round_complete: 145,
        explanation_generated: 178,
        consent_revoke: 12,
        data_erasure: 8,
        provenance_link: 95,
    },
    recent_events: [
        { id: "evt-001", event_type: "recognition_inference", actor: "system", timestamp: minutesAgo(2) },
        { id: "evt-002", event_type: "bias_audit", actor: "auditor@forensic.ai", timestamp: minutesAgo(5) },
        { id: "evt-003", event_type: "consent_grant", actor: "subject-42", timestamp: minutesAgo(8) },
        { id: "evt-004", event_type: "fl_round_complete", actor: "fl-orchestrator", timestamp: minutesAgo(12) },
        { id: "evt-005", event_type: "bundle_creation", actor: "system", timestamp: minutesAgo(15) },
        { id: "evt-006", event_type: "explanation_generated", actor: "xai-engine", timestamp: minutesAgo(18) },
        { id: "evt-007", event_type: "model_training", actor: "trainer@lab.edu", timestamp: minutesAgo(25) },
        { id: "evt-008", event_type: "recognition_inference", actor: "system", timestamp: minutesAgo(30) },
    ],
    chain_integrity: { valid: true, count: 1247 },
    crypto_status: {
        did_enabled: true,
        zk_scheme: "schnorr-fiat-shamir",
        total_dids: 12,
        total_vcs: 34,
    },
};

// ── Events ──────────────────────────────────────────────
export const demoEvents = {
    items: demoDashboard.recent_events.map((e, i) => ({
        ...e,
        event_hash: `sha256:${Array(64).fill(0).map(() => "0123456789abcdef"[Math.floor(Math.random() * 16)]).join("")}`,
        previous_hash: i === 0 ? "genesis" : `sha256:${Array(64).fill(0).map(() => "0123456789abcdef"[Math.floor(Math.random() * 16)]).join("")}`,
        payload: { confidence: 0.94, model_id: "pca-svm-v3" },
        sequence_number: 1247 - i,
    })),
    total: 1247,
    limit: 50,
    offset: 0,
};

// ── FL Rounds ───────────────────────────────────────────
export const demoFLRounds = [
    { simulation_id: "sim-a1b2c3d4", round_number: 5, global_accuracy: 0.923, num_clients: 8, dp_enabled: true, timestamp: minutesAgo(10) },
    { simulation_id: "sim-a1b2c3d4", round_number: 4, global_accuracy: 0.891, num_clients: 8, dp_enabled: true, timestamp: minutesAgo(20) },
    { simulation_id: "sim-a1b2c3d4", round_number: 3, global_accuracy: 0.845, num_clients: 8, dp_enabled: true, timestamp: minutesAgo(30) },
    { simulation_id: "sim-e5f6g7h8", round_number: 3, global_accuracy: 0.876, num_clients: 5, dp_enabled: false, timestamp: minutesAgo(60) },
    { simulation_id: "sim-e5f6g7h8", round_number: 2, global_accuracy: 0.812, num_clients: 5, dp_enabled: false, timestamp: minutesAgo(70) },
];

// ── Recognition Models ──────────────────────────────────
export const demoModels = [
    { model_id: "mdl-pca-svm-001", model_type: "PCA+SVM", accuracy: 0.946, trained_at: minutesAgo(120), num_classes: 15 },
    { model_id: "mdl-pca-svm-002", model_type: "PCA+SVM", accuracy: 0.912, trained_at: minutesAgo(360), num_classes: 10 },
];

// ── Gallery ─────────────────────────────────────────────
export const demoGalleryCount = { total: 42 };

// ── Compliance ──────────────────────────────────────────
export const demoCompliance = {
    compliant: true,
    overall_score: 0.87,
    checks: {
        hash_chain_integrity: { passed: true, score: 1.0 },
        merkle_verification: { passed: true, score: 1.0 },
        consent_coverage: { passed: true, score: 0.85 },
        bias_audit_current: { passed: true, score: 0.78 },
        explainability_coverage: { passed: false, score: 0.65 },
        data_retention_policy: { passed: true, score: 0.92 },
    },
};

// ── Merkle ──────────────────────────────────────────────
export const demoMerkleRoot = {
    root: "sha256:4a7d1ed414474e4033ac29ccb8653d9b0c8e8f4a2b6c3e9ef1c2d3a4b5c6d7e8",
    leaf_count: 1247,
    tree_height: 11,
};

// ── Bundles ─────────────────────────────────────────────
export const demoBundles = {
    items: [
        {
            bundle_id: "bnd-001",
            event_count: 12,
            created_at: minutesAgo(15),
            verified: true,
            has_zk_proof: true,
            has_did_credential: true,
        },
        {
            bundle_id: "bnd-002",
            event_count: 8,
            created_at: minutesAgo(45),
            verified: true,
            has_zk_proof: true,
            has_did_credential: true,
        },
        {
            bundle_id: "bnd-003",
            event_count: 15,
            created_at: minutesAgo(120),
            verified: true,
            has_zk_proof: true,
            has_did_credential: true,
        },
    ],
    total: 38,
    limit: 50,
    offset: 0,
};

// ── Consent ─────────────────────────────────────────────
export const demoConsentStatus = {
    subject_id: "subject-42",
    consent_granted: true,
    purposes: ["facial_recognition", "training", "research"],
    granted_at: minutesAgo(1440),
    expires_at: null,
};

// ── Fairness ────────────────────────────────────────────
export const demoFairnessStatus = {
    last_audit: minutesAgo(5),
    demographic_parity_gap: 0.04,
    equalized_odds_gap: 0.06,
    compliant: true,
    groups_audited: 4,
};

export const demoFairnessAudits = {
    items: [
        {
            audit_id: "aud-001",
            timestamp: minutesAgo(5),
            demographic_parity: 0.96,
            equalized_odds: 0.94,
            overall_fair: true,
            num_groups: 4,
        },
        {
            audit_id: "aud-002",
            timestamp: minutesAgo(60),
            demographic_parity: 0.93,
            equalized_odds: 0.91,
            overall_fair: true,
            num_groups: 4,
        },
    ],
    total: 67,
    limit: 50,
    offset: 0,
};
