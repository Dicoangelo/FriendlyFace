#!/usr/bin/env python3
"""Seed FriendlyFace with realistic forensic demo data.

Valid event_types: training_start, training_complete, model_registered,
inference_request, inference_result, explanation_generated, bias_audit,
consent_recorded, consent_update, bundle_created, fl_round,
security_alert, compliance_report

Fairness groups need: group_name, true_positives, false_positives,
true_negatives, false_negatives
"""

import requests
import time
import json

BASE = "https://friendlyface.metaventionsai.com/api/v1"

def post(path, data):
    r = requests.post(f"{BASE}{path}", json=data, timeout=30)
    if r.status_code in (200, 201):
        return r.json()
    print(f"  WARN {path} → {r.status_code}: {r.text[:200]}")
    return None

def get(path):
    r = requests.get(f"{BASE}{path}", timeout=30)
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            return None
    return None

# ──────────────────────────────────────────────────────────
# 1. Forensic events — realistic face recognition pipeline
# ──────────────────────────────────────────────────────────
print("═══ Phase 1: Seeding forensic events ═══")

EVENT_SCENARIOS = [
    # Training pipeline
    {"event_type": "training_start", "actor": "ml-pipeline-v2", "payload": {
        "dataset": "LFW-aligned-112x112", "n_subjects": 5749, "n_images": 13233,
        "model": "ArcFace-ResNet50", "optimizer": "SGD", "lr": 0.01
    }},
    {"event_type": "training_complete", "actor": "ml-pipeline-v2", "payload": {
        "accuracy": 0.9947, "loss": 0.0312, "epochs": 45, "duration_seconds": 7823,
        "model_hash": "sha256:a1b2c3d4e5f6789012345678abcdef0123456789abcdef01",
        "embedding_dim": 512, "n_params": "25.6M"
    }},
    {"event_type": "model_registered", "actor": "model-registry", "payload": {
        "model_id": "arcface-resnet50-v2.1", "version": "2.1.0",
        "accuracy": 0.9952, "eer": 0.0078, "auc": 0.9998,
        "framework": "PyTorch", "params": "25.6M"
    }},
    # Inference events — various subjects and cameras
    {"event_type": "inference_request", "actor": "edge-node-cam-01", "payload": {
        "camera_id": "entrance-north", "frame_id": 48291,
        "resolution": "1920x1080", "timestamp_ms": 1707321600000
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-01", "payload": {
        "subject_id": "subj-0042", "confidence": 0.9823, "latency_ms": 47,
        "camera_id": "entrance-north", "liveness_score": 0.991,
        "embedding_distance": 0.1247, "threshold": 0.35
    }},
    {"event_type": "inference_request", "actor": "edge-node-cam-02", "payload": {
        "camera_id": "lobby-east", "frame_id": 51023,
        "resolution": "1920x1080", "timestamp_ms": 1707321660000
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-02", "payload": {
        "subject_id": "subj-0108", "confidence": 0.9456, "latency_ms": 52,
        "camera_id": "lobby-east", "liveness_score": 0.987,
        "embedding_distance": 0.1891, "threshold": 0.35
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-03", "payload": {
        "subject_id": "subj-0015", "confidence": 0.9712, "latency_ms": 39,
        "camera_id": "parking-A", "liveness_score": 0.995,
        "embedding_distance": 0.1056, "threshold": 0.35
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-01", "payload": {
        "subject_id": "unknown", "confidence": 0.2134, "latency_ms": 44,
        "camera_id": "entrance-north", "liveness_score": 0.982,
        "embedding_distance": 0.8721, "threshold": 0.35, "alert": "unknown_person"
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-04", "payload": {
        "subject_id": "subj-0042", "confidence": 0.9791, "latency_ms": 41,
        "camera_id": "corridor-B2", "liveness_score": 0.993,
        "embedding_distance": 0.1312, "threshold": 0.35
    }},
    # FL rounds
    {"event_type": "fl_round", "actor": "fl-coordinator", "payload": {
        "round": 1, "n_clients": 5, "aggregation": "FedAvg-DP",
        "epsilon": 1.0, "delta": 1e-5, "accuracy": 0.9891
    }},
    {"event_type": "fl_round", "actor": "fl-coordinator", "payload": {
        "round": 2, "n_clients": 5, "aggregation": "FedAvg-DP",
        "epsilon": 1.0, "delta": 1e-5, "accuracy": 0.9923
    }},
    {"event_type": "fl_round", "actor": "fl-coordinator", "payload": {
        "round": 3, "n_clients": 5, "aggregation": "FedAvg-DP",
        "epsilon": 1.0, "delta": 1e-5, "accuracy": 0.9947,
        "convergence_metric": 0.9970
    }},
    # Bias audit
    {"event_type": "bias_audit", "actor": "fairness-monitor", "payload": {
        "groups_audited": 6, "demographic_parity_pass": True,
        "equalized_odds_pass": True, "disparate_impact_ratio": 0.92
    }},
    # Consent events
    {"event_type": "consent_recorded", "actor": "consent-api", "payload": {
        "subject_id": "subj-0042", "purpose": "recognition",
        "method": "explicit_opt_in", "jurisdiction": "EU-GDPR"
    }},
    {"event_type": "consent_recorded", "actor": "consent-api", "payload": {
        "subject_id": "subj-0108", "purpose": "recognition",
        "method": "explicit_opt_in", "jurisdiction": "EU-GDPR"
    }},
    {"event_type": "consent_recorded", "actor": "consent-api", "payload": {
        "subject_id": "subj-0015", "purpose": "training",
        "method": "informed_consent", "jurisdiction": "US-BIPA"
    }},
    {"event_type": "consent_update", "actor": "consent-api", "payload": {
        "subject_id": "subj-0042", "purpose": "training",
        "action": "granted", "method": "explicit_opt_in"
    }},
    # Security alert
    {"event_type": "security_alert", "actor": "edge-node-cam-04", "payload": {
        "alert_type": "liveness_failure", "liveness_score": 0.231,
        "camera_id": "corridor-B2", "detection": "print_attack",
        "alert_level": "high", "subject_id": "unknown"
    }},
    {"event_type": "security_alert", "actor": "fl-coordinator", "payload": {
        "alert_type": "poisoning_attempt", "client_id": "client-3",
        "round": 2, "anomaly_score": 4.7, "threshold": 3.0,
        "action": "client_excluded"
    }},
    # Explanation events
    {"event_type": "explanation_generated", "actor": "xai-service", "payload": {
        "method": "LIME", "num_superpixels": 50,
        "top_features": ["left_eye", "nose_bridge", "jawline"],
        "prediction_confidence": 0.9823
    }},
    {"event_type": "explanation_generated", "actor": "xai-service", "payload": {
        "method": "SHAP", "num_samples": 128,
        "top_features": ["periocular", "nose", "chin"],
        "prediction_confidence": 0.9456
    }},
    # Compliance report
    {"event_type": "compliance_report", "actor": "governance-engine", "payload": {
        "framework": "GDPR", "articles_checked": 12,
        "compliant": True, "risk_level": "low",
        "recommendations": ["Review consent expiry dates"]
    }},
    {"event_type": "compliance_report", "actor": "governance-engine", "payload": {
        "framework": "NIST-AI-RMF", "categories_checked": 8,
        "compliant": True, "risk_level": "low"
    }},
    # More inference for volume
    {"event_type": "inference_result", "actor": "edge-node-cam-02", "payload": {
        "subject_id": "subj-0227", "confidence": 0.9634, "latency_ms": 45,
        "camera_id": "lobby-east", "liveness_score": 0.989,
        "embedding_distance": 0.1543, "threshold": 0.35
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-05", "payload": {
        "subject_id": "subj-0042", "confidence": 0.9878, "latency_ms": 38,
        "camera_id": "server-room", "liveness_score": 0.997,
        "embedding_distance": 0.0987, "threshold": 0.35
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-01", "payload": {
        "subject_id": "subj-0331", "confidence": 0.9567, "latency_ms": 51,
        "camera_id": "entrance-north", "liveness_score": 0.984,
        "embedding_distance": 0.1678, "threshold": 0.35
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-03", "payload": {
        "subject_id": "subj-0108", "confidence": 0.9389, "latency_ms": 46,
        "camera_id": "parking-A", "liveness_score": 0.991,
        "embedding_distance": 0.2012, "threshold": 0.35
    }},
    {"event_type": "inference_request", "actor": "edge-node-cam-05", "payload": {
        "camera_id": "server-room", "frame_id": 72019,
        "resolution": "1920x1080", "timestamp_ms": 1707322200000
    }},
    {"event_type": "inference_result", "actor": "edge-node-cam-05", "payload": {
        "subject_id": "subj-0015", "confidence": 0.9801, "latency_ms": 36,
        "camera_id": "server-room", "liveness_score": 0.998,
        "embedding_distance": 0.1012, "threshold": 0.35
    }},
    # Bundle creation event
    {"event_type": "bundle_created", "actor": "forensic-engine", "payload": {
        "bundle_type": "training_pipeline", "events_included": 3,
        "merkle_root": "abc123...", "integrity_verified": True
    }},
    {"event_type": "bundle_created", "actor": "forensic-engine", "payload": {
        "bundle_type": "inference_session", "events_included": 12,
        "merkle_root": "def456...", "integrity_verified": True
    }},
    # Model evaluation
    {"event_type": "training_complete", "actor": "ml-pipeline-v2", "payload": {
        "model": "ArcFace-ResNet50-v2.1-ft", "dataset": "CelebA-fine-tune",
        "accuracy": 0.9961, "loss": 0.0198, "epochs": 10,
        "duration_seconds": 2341, "fine_tuned_from": "arcface-resnet50-v2.1"
    }},
]

event_ids = []
for i, evt in enumerate(EVENT_SCENARIOS):
    result = post("/events", evt)
    if result:
        event_ids.append(result["id"])
        print(f"  [{i+1}/{len(EVENT_SCENARIOS)}] {evt['event_type']} → {result['id'][:8]}...")
    else:
        event_ids.append(None)
    time.sleep(0.1)

valid_ids = [eid for eid in event_ids if eid is not None]
print(f"\n  Created {len(valid_ids)} events\n")

# ──────────────────────────────────────────────────────────
# 2. Forensic bundles
# ──────────────────────────────────────────────────────────
print("═══ Phase 2: Creating forensic bundles ═══")

bundles = []

# Training pipeline bundle (events 0-2)
train_ids = [eid for eid in event_ids[0:3] if eid]
if len(train_ids) >= 2:
    b = post("/bundles", {"event_ids": train_ids})
    if b:
        bundles.append(b)
        print(f"  Bundle 1 (Training Pipeline): {b['id'][:8]}...")

# Inference session bundle (events 3-9)
infer_ids = [eid for eid in event_ids[3:10] if eid]
if len(infer_ids) >= 2:
    b = post("/bundles", {"event_ids": infer_ids})
    if b:
        bundles.append(b)
        print(f"  Bundle 2 (Inference Session): {b['id'][:8]}...")

# FL rounds bundle (events 10-12)
fl_ids = [eid for eid in event_ids[10:13] if eid]
if len(fl_ids) >= 2:
    b = post("/bundles", {"event_ids": fl_ids})
    if b:
        bundles.append(b)
        print(f"  Bundle 3 (FL Rounds): {b['id'][:8]}...")

# Compliance bundle (bias + consent + compliance)
comp_ids = [eid for eid in [event_ids[13]] + event_ids[14:18] + event_ids[22:24] if eid]
if len(comp_ids) >= 2:
    b = post("/bundles", {"event_ids": comp_ids})
    if b:
        bundles.append(b)
        print(f"  Bundle 4 (Compliance): {b['id'][:8]}...")

# Security + XAI bundle
sec_ids = [eid for eid in event_ids[18:22] if eid]
if len(sec_ids) >= 2:
    b = post("/bundles", {"event_ids": sec_ids})
    if b:
        bundles.append(b)
        print(f"  Bundle 5 (Security + XAI): {b['id'][:8]}...")

# Latest inference bundle
latest_ids = [eid for eid in event_ids[24:30] if eid]
if len(latest_ids) >= 2:
    b = post("/bundles", {"event_ids": latest_ids})
    if b:
        bundles.append(b)
        print(f"  Bundle 6 (Latest Inference): {b['id'][:8]}...")

print(f"\n  Created {len(bundles)} bundles\n")

# ──────────────────────────────────────────────────────────
# 3. Provenance DAG
# ──────────────────────────────────────────────────────────
print("═══ Phase 3: Building provenance DAG ═══")

prov_nodes = []

p1 = post("/provenance", {
    "entity_type": "dataset", "entity_id": "LFW-aligned-112x112",
    "metadata": {"n_subjects": 5749, "n_images": 13233, "source": "LFW"}
})
if p1:
    prov_nodes.append(p1["id"])
    print(f"  Dataset: {p1['id'][:8]}...")

p2 = post("/provenance", {
    "entity_type": "model", "entity_id": "ArcFace-ResNet50-v2.1",
    "parents": [prov_nodes[0]] if prov_nodes else [],
    "relations": ["derived_from"],
    "metadata": {"framework": "PyTorch", "params": "25.6M", "accuracy": 0.9952}
})
if p2:
    prov_nodes.append(p2["id"])
    print(f"  Model: {p2['id'][:8]}...")

p3 = post("/provenance", {
    "entity_type": "fl_aggregation", "entity_id": "fedavg-round-3",
    "parents": [prov_nodes[-1]] if prov_nodes else [],
    "relations": ["generated_by"],
    "metadata": {"n_clients": 5, "rounds": 3, "dp_epsilon": 1.0}
})
if p3:
    prov_nodes.append(p3["id"])
    print(f"  FL Aggregation: {p3['id'][:8]}...")

p4 = post("/provenance", {
    "entity_type": "deployment", "entity_id": "edge-fleet-v2.1",
    "parents": [prov_nodes[-1]] if prov_nodes else [],
    "relations": ["used"],
    "metadata": {"n_nodes": 5, "regions": ["us-east", "eu-west"]}
})
if p4:
    prov_nodes.append(p4["id"])
    print(f"  Deployment: {p4['id'][:8]}...")

p5 = post("/provenance", {
    "entity_type": "audit", "entity_id": "bias-audit-2026-02-08",
    "parents": [prov_nodes[1]] if len(prov_nodes) > 1 else [],
    "relations": ["attributed_to"],
    "metadata": {"framework": "GDPR+NIST", "result": "compliant"}
})
if p5:
    prov_nodes.append(p5["id"])
    print(f"  Audit: {p5['id'][:8]}...")

print(f"\n  Created {len(prov_nodes)} provenance nodes\n")

# ──────────────────────────────────────────────────────────
# 4. Consent grants (API endpoint)
# ──────────────────────────────────────────────────────────
print("═══ Phase 4: Granting consent ═══")

from datetime import datetime, timedelta, timezone

SUBJECTS = [
    ("subj-0042", "recognition"), ("subj-0042", "training"),
    ("subj-0108", "recognition"), ("subj-0015", "recognition"),
    ("subj-0015", "training"), ("subj-0227", "recognition"),
    ("subj-0331", "recognition"), ("subj-0331", "training"),
]
consent_count = 0
for subj, purpose in SUBJECTS:
    exp = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
    result = post("/consent/grant", {
        "subject_id": subj, "purpose": purpose,
        "expiry": exp, "actor": "enrollment-portal"
    })
    if result:
        consent_count += 1
        print(f"  {subj} → {purpose}")
    time.sleep(0.05)

print(f"\n  Granted {consent_count} consent records\n")

# ──────────────────────────────────────────────────────────
# 5. Fairness audits (correct schema)
# ──────────────────────────────────────────────────────────
print("═══ Phase 5: Running fairness audits ═══")

# Age-based audit
audit1 = post("/fairness/audit", {
    "groups": [
        {"group_name": "male_18-30", "true_positives": 487, "false_positives": 8, "true_negatives": 492, "false_negatives": 13},
        {"group_name": "female_18-30", "true_positives": 491, "false_positives": 6, "true_negatives": 494, "false_negatives": 9},
        {"group_name": "male_30-60", "true_positives": 478, "false_positives": 12, "true_negatives": 488, "false_negatives": 22},
        {"group_name": "female_30-60", "true_positives": 483, "false_positives": 9, "true_negatives": 491, "false_negatives": 17},
        {"group_name": "male_60+", "true_positives": 445, "false_positives": 15, "true_negatives": 485, "false_negatives": 55},
        {"group_name": "female_60+", "true_positives": 452, "false_positives": 12, "true_negatives": 488, "false_negatives": 48},
    ],
    "demographic_parity_threshold": 0.1,
    "equalized_odds_threshold": 0.1,
    "metadata": {"dataset": "LFW-eval-balanced", "model": "ArcFace-ResNet50-v2.1", "audit_type": "age_gender"}
})
if audit1:
    print(f"  Age/gender audit: {json.dumps(audit1)[:100]}...")

# Ethnicity audit
audit2 = post("/fairness/audit", {
    "groups": [
        {"group_name": "caucasian", "true_positives": 1823, "false_positives": 42, "true_negatives": 1858, "false_negatives": 77},
        {"group_name": "african", "true_positives": 1798, "false_positives": 55, "true_negatives": 1845, "false_negatives": 102},
        {"group_name": "east_asian", "true_positives": 1812, "false_positives": 48, "true_negatives": 1852, "false_negatives": 88},
        {"group_name": "south_asian", "true_positives": 1805, "false_positives": 51, "true_negatives": 1849, "false_negatives": 95},
    ],
    "demographic_parity_threshold": 0.1,
    "equalized_odds_threshold": 0.1,
    "metadata": {"dataset": "CelebA-eval-ethnicity", "model": "ArcFace-ResNet50-v2.1", "audit_type": "ethnicity"}
})
if audit2:
    print(f"  Ethnicity audit: {json.dumps(audit2)[:100]}...")

print()

# ──────────────────────────────────────────────────────────
# 6. ZK Proofs
# ──────────────────────────────────────────────────────────
print("═══ Phase 6: Generating ZK proofs ═══")

zk_count = 0
for b in bundles:
    zk = post("/zk/prove", {"bundle_id": b["id"]})
    if zk:
        zk_count += 1
        print(f"  ZK proof for bundle {b['id'][:8]}...")
    time.sleep(0.15)

print(f"\n  Generated {zk_count} ZK proofs\n")

# ──────────────────────────────────────────────────────────
# 7. Bundle verification
# ──────────────────────────────────────────────────────────
print("═══ Phase 7: Verifying bundles ═══")

for b in bundles:
    v = post(f"/verify/{b['id']}", {})
    if v:
        status = v.get("status", v.get("verified", "?"))
        print(f"  Bundle {b['id'][:8]}... → {status}")
    time.sleep(0.05)

# ──────────────────────────────────────────────────────────
# 8. FL simulations
# ──────────────────────────────────────────────────────────
print("\n═══ Phase 8: Running FL simulations ═══")

fl = post("/fl/start", {
    "n_clients": 5, "n_rounds": 3,
    "enable_poisoning_detection": True,
    "poisoning_threshold": 3.0, "seed": 42
})
if fl:
    print(f"  FedAvg simulation: {fl.get('simulation_id', '?')[:12]}...")

fl_dp = post("/fl/dp-start", {
    "n_clients": 4, "n_rounds": 2,
    "epsilon": 1.0, "delta": 1e-5,
    "max_grad_norm": 1.0, "seed": 123
})
if fl_dp:
    print(f"  DP-FedAvg simulation: {fl_dp.get('simulation_id', '?')[:12]}...")

# ──────────────────────────────────────────────────────────
# Final check
# ──────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("  SEED COMPLETE")
print("═" * 55)

health = get("/../health")
if health:
    print(f"  Total events:    {health.get('event_count', '?')}")
    root = health.get('merkle_root', 'none')
    print(f"  Merkle root:     {str(root)[:24]}...")

print(f"  Bundles:         {len(bundles)}")
print(f"  Provenance:      {len(prov_nodes)} nodes")
print(f"  Consent:         {consent_count} grants")
print(f"  ZK proofs:       {zk_count}")
print(f"  FL simulations:  2")
print(f"  Fairness audits: 2")
print("═" * 55)
print("\n  → https://friendlyface.metaventionsai.com")
