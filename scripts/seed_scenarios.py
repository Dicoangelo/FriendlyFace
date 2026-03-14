#!/usr/bin/env python3
"""Seed FriendlyFace with realistic forensic scenarios.

Three interconnected scenarios that tell compelling stories:
  airport     — Airport security checkpoint with traveler verification
  coldcase    — Missing persons cold case investigation
  adversarial — Adversarial attack detection and response
  all         — Run all three scenarios sequentially

Valid event_types: training_start, training_complete, model_registered,
inference_request, inference_result, explanation_generated, bias_audit,
consent_recorded, consent_update, bundle_created, fl_round,
security_alert, compliance_report
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ─── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Seed FriendlyFace with realistic forensic scenarios",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python seed_scenarios.py --scenario airport --local
  python seed_scenarios.py --scenario all --remote
  python seed_scenarios.py --scenario coldcase --base-url http://staging:8000/api/v1
""",
)
parser.add_argument(
    "--scenario",
    required=True,
    choices=["airport", "coldcase", "adversarial", "all"],
    help="Which scenario to seed",
)
parser.add_argument("--local", action="store_true", help="Target localhost:3849")
parser.add_argument(
    "--remote", action="store_true", help="Target friendlyface.metaventionsai.com"
)
parser.add_argument("--base-url", type=str, help="Custom base URL")
parser.add_argument("--api-key", type=str, help="API key for authentication")
args = parser.parse_args()

if args.base_url:
    BASE = args.base_url.rstrip("/")
elif args.local:
    BASE = "http://localhost:3849/api/v1"
elif args.remote:
    BASE = "https://friendlyface.metaventionsai.com/api/v1"
else:
    BASE = "http://localhost:3849/api/v1"

HEADERS = {}
if args.api_key:
    HEADERS["Authorization"] = f"Bearer {args.api_key}"

# ─── Helpers ──────────────────────────────────────────────────────────────────

COUNTS = {
    "events": 0,
    "bundles": 0,
    "provenance": 0,
    "consent_grants": 0,
    "consent_revocations": 0,
    "fairness_audits": 0,
    "zk_proofs": 0,
    "fl_simulations": 0,
    "verifications": 0,
    "compliance_reports": 0,
}


def post(path, data):
    """POST to API, return JSON or None."""
    try:
        r = requests.post(
            f"{BASE}{path}", json=data, headers=HEADERS, timeout=30
        )
        if r.status_code in (200, 201):
            return r.json()
        print(f"  WARN {path} -> {r.status_code}: {r.text[:200]}")
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to {BASE}")
        sys.exit(1)
    except Exception as e:
        print(f"  ERROR {path}: {e}")
    return None


def get(path):
    """GET from API, return JSON or None."""
    try:
        r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def emit_event(event_type, actor, payload):
    """Record a forensic event and return its ID."""
    result = post("/events", {
        "event_type": event_type,
        "actor": actor,
        "payload": payload,
    })
    if result:
        COUNTS["events"] += 1
        return result["id"]
    return None


def emit_events(events):
    """Emit a list of (event_type, actor, payload) tuples, return list of IDs."""
    ids = []
    for event_type, actor, payload in events:
        eid = emit_event(event_type, actor, payload)
        ids.append(eid)
        time.sleep(0.05)
    return ids


def create_bundle(event_ids, provenance_ids=None, label=""):
    """Create a forensic bundle from event IDs."""
    valid = [eid for eid in event_ids if eid]
    if len(valid) < 2:
        print(f"  SKIP bundle ({label}): not enough events")
        return None
    body = {"event_ids": valid}
    if provenance_ids:
        valid_prov = [pid for pid in provenance_ids if pid]
        if valid_prov:
            body["provenance_node_ids"] = valid_prov
    result = post("/bundles", body)
    if result:
        COUNTS["bundles"] += 1
        print(f"  Bundle ({label}): {result['id'][:12]}...")
        return result
    return None


def verify_bundle(bundle):
    """Verify a bundle and return result."""
    if not bundle:
        return None
    result = post(f"/verify/{bundle['id']}", {})
    if result:
        COUNTS["verifications"] += 1
        status = result.get("status", result.get("verified", "?"))
        print(f"  Verified {bundle['id'][:12]}... -> {status}")
    return result


def zk_prove(bundle):
    """Generate ZK proof for a bundle."""
    if not bundle:
        return None
    result = post("/zk/prove", {"bundle_id": bundle["id"]})
    if result:
        COUNTS["zk_proofs"] += 1
        print(f"  ZK proof for {bundle['id'][:12]}...")
    return result


def add_provenance(entity_type, entity_id, parents=None, relations=None, metadata=None):
    """Add a provenance node."""
    body = {"entity_type": entity_type, "entity_id": entity_id}
    if parents:
        body["parents"] = [p for p in parents if p]
    if relations:
        body["relations"] = relations
    if metadata:
        body["metadata"] = metadata
    result = post("/provenance", body)
    if result:
        COUNTS["provenance"] += 1
        return result["id"]
    return None


def grant_consent(subject_id, purpose, actor, expiry_days=365):
    """Grant consent for a subject."""
    exp = (datetime.now(timezone.utc) + timedelta(days=expiry_days)).isoformat()
    result = post("/consent/grant", {
        "subject_id": subject_id,
        "purpose": purpose,
        "expiry": exp,
        "actor": actor,
    })
    if result:
        COUNTS["consent_grants"] += 1
    return result


def revoke_consent(subject_id, purpose, reason, actor):
    """Revoke consent for a subject."""
    result = post("/consent/revoke", {
        "subject_id": subject_id,
        "purpose": purpose,
        "reason": reason,
        "actor": actor,
    })
    if result:
        COUNTS["consent_revocations"] += 1
    return result


def run_fairness_audit(groups, metadata=None, dp_threshold=0.1, eo_threshold=0.1):
    """Run a fairness audit."""
    body = {
        "groups": groups,
        "demographic_parity_threshold": dp_threshold,
        "equalized_odds_threshold": eo_threshold,
    }
    if metadata:
        body["metadata"] = metadata
    result = post("/fairness/audit", body)
    if result:
        COUNTS["fairness_audits"] += 1
        print(f"  Audit result: {json.dumps(result)[:120]}...")
    return result


def generate_compliance_report():
    """Generate a governance compliance report."""
    result = post("/governance/compliance/generate", {})
    if result:
        COUNTS["compliance_reports"] += 1
        print(f"  Compliance report generated")
    return result


def header(text):
    """Print a section header."""
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}\n")


def ts_offset(days_ago, hour=10, minute=0):
    """Generate a timestamp N days ago at a specific time."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    dt = dt.replace(hour=hour % 24, minute=minute % 60, second=0, microsecond=0)
    return int(dt.timestamp() * 1000)


def ts_iso(days_ago, hour=10, minute=0):
    """Generate an ISO timestamp N days ago."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    dt = dt.replace(hour=hour % 24, minute=minute % 60, second=0, microsecond=0)
    return dt.isoformat()


# ─── Scenario 1: Airport Security Checkpoint ─────────────────────────────────

def scenario_airport():
    """Airport deploys facial recognition for traveler verification."""
    header("SCENARIO: Airport Security Checkpoint")
    print("  Story: Heathrow International deploys a facial recognition system")
    print("  for traveler identity verification across 5 camera positions.\n")

    # ── Phase 1: Training Pipeline ────────────────────────────────────────────
    header("Phase 1: Training Pipeline")

    training_events = emit_events([
        ("training_start", "airport-ml-pipeline", {
            "dataset": "airport-enrolled-faces-v3",
            "n_subjects": 12480,
            "n_images": 87360,
            "model": "ArcFace-MobileNetV3",
            "optimizer": "AdamW",
            "lr": 0.001,
            "augmentation": ["horizontal_flip", "color_jitter", "random_crop"],
            "started_at": ts_iso(28, 6, 0),
        }),
        ("training_complete", "airport-ml-pipeline", {
            "model": "ArcFace-MobileNetV3",
            "accuracy": 0.9934,
            "loss": 0.0287,
            "epochs": 60,
            "duration_seconds": 14520,
            "model_hash": "sha256:7f3a9b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a",
            "embedding_dim": 512,
            "n_params": "4.2M",
            "completed_at": ts_iso(28, 10, 2),
        }),
        ("model_registered", "airport-model-registry", {
            "model_id": "arcface-mobilenet-airport-v3.0",
            "version": "3.0.0",
            "accuracy": 0.9934,
            "eer": 0.0062,
            "auc": 0.9997,
            "framework": "PyTorch",
            "params": "4.2M",
            "deployment_target": "edge-gpu",
            "registered_at": ts_iso(27, 8, 0),
        }),
    ])
    for i, eid in enumerate(training_events):
        if eid:
            labels = ["training_start", "training_complete", "model_registered"]
            print(f"  [{i+1}/3] {labels[i]} -> {eid[:12]}...")

    # ── Phase 2: Model Deployment ─────────────────────────────────────────────
    header("Phase 2: Model Deployment to 5 Cameras")

    CAMERAS = [
        ("cam-entrance", "Terminal 5 Main Entrance", "entrance"),
        ("cam-gate-a", "Gate A12 Boarding", "gate"),
        ("cam-gate-b", "Gate B7 Boarding", "gate"),
        ("cam-gate-c", "Gate C3 Boarding", "gate"),
        ("cam-lounge", "Executive Lounge Entry", "lounge"),
    ]

    deployment_events = []
    for cam_id, cam_name, cam_zone in CAMERAS:
        eid = emit_event("model_registered", "deployment-orchestrator", {
            "action": "deploy",
            "model_id": "arcface-mobilenet-airport-v3.0",
            "camera_id": cam_id,
            "camera_name": cam_name,
            "zone": cam_zone,
            "resolution": "1920x1080",
            "fps": 15,
            "deployed_at": ts_iso(26, 5, 30),
        })
        deployment_events.append(eid)
        if eid:
            print(f"  Deployed to {cam_name} ({cam_id})")
        time.sleep(0.05)

    # ── Phase 3: Traveler Enrollment & Consent ────────────────────────────────
    header("Phase 3: Traveler Enrollment & Consent")

    TRAVELERS = [
        ("traveler-001", "James Morrison", "BA-2741", "LHR->JFK"),
        ("traveler-002", "Aisha Patel", "BA-2741", "LHR->JFK"),
        ("traveler-003", "Chen Wei", "CX-8820", "LHR->HKG"),
        ("traveler-004", "Maria Garcia", "IB-3217", "LHR->MAD"),
        ("traveler-005", "Oluwaseun Adebayo", "VS-4501", "LHR->LOS"),
        ("traveler-006", "Yuki Tanaka", "JL-0042", "LHR->NRT"),
        ("traveler-007", "Hans Mueller", "LH-4578", "LHR->FRA"),
        ("traveler-008", "Sophie Dubois", "AF-1680", "LHR->CDG"),
        ("traveler-009", "Ahmed Hassan", "EK-0030", "LHR->DXB"),
        ("traveler-010", "Priya Sharma", "AI-0112", "LHR->DEL"),
        ("traveler-011", "Roberto Silva", "TP-3350", "LHR->LIS"),
        ("traveler-012", "Anna Kowalski", "LO-0282", "LHR->WAW"),
        ("traveler-013", "Mikhail Volkov", "SU-2573", "LHR->SVO"),
        ("traveler-014", "Fatima Al-Rashid", "QR-0008", "LHR->DOH"),
        ("traveler-015", "David Kim", "KE-0908", "LHR->ICN"),
        ("traveler-016", "Isabella Rossi", "AZ-0249", "LHR->FCO"),
        ("traveler-017", "Thomas Andersen", "SK-0812", "LHR->CPH"),
        ("traveler-018", "Nadia Petrova", "TK-1982", "LHR->IST"),
        ("traveler-019", "Carlos Mendez", "AM-0001", "LHR->MEX"),
        ("traveler-020", "Emma Wilson", "AC-0855", "LHR->YYZ"),
    ]

    for tid, name, flight, route in TRAVELERS:
        grant_consent(tid, "facial_verification", "enrollment-kiosk-T5")
        print(f"  Consent: {name} ({tid}) - {flight} {route}")
        time.sleep(0.05)

    # ── Phase 4: Inference Events (50 across 5 cameras) ──────────────────────
    header("Phase 4: Inference Operations (50 events)")

    random.seed(42)  # Reproducible demo data
    inference_ids = []
    frame_counter = 10000

    for i in range(50):
        cam_id, cam_name, cam_zone = random.choice(CAMERAS)
        frame_counter += random.randint(50, 200)
        days_ago = random.randint(1, 25)
        hour = random.randint(5, 22)
        minute = random.randint(0, 59)

        # 80% known travelers, 20% unknown
        if random.random() < 0.80:
            traveler = random.choice(TRAVELERS)
            subject_id = traveler[0]
            confidence = round(random.uniform(0.88, 0.99), 4)
            embed_dist = round(random.uniform(0.05, 0.25), 4)
            liveness = round(random.uniform(0.96, 0.999), 3)
        else:
            subject_id = "unknown"
            confidence = round(random.uniform(0.10, 0.35), 4)
            embed_dist = round(random.uniform(0.65, 0.95), 4)
            liveness = round(random.uniform(0.85, 0.999), 3)

        # Inference request
        req_id = emit_event("inference_request", f"edge-{cam_id}", {
            "camera_id": cam_id,
            "camera_name": cam_name,
            "zone": cam_zone,
            "frame_id": frame_counter,
            "resolution": "1920x1080",
            "timestamp_ms": ts_offset(days_ago, hour, minute),
        })
        time.sleep(0.05)

        # Inference result
        res_id = emit_event("inference_result", f"edge-{cam_id}", {
            "subject_id": subject_id,
            "confidence": confidence,
            "latency_ms": random.randint(28, 65),
            "camera_id": cam_id,
            "zone": cam_zone,
            "liveness_score": liveness,
            "embedding_distance": embed_dist,
            "threshold": 0.35,
            "timestamp_ms": ts_offset(days_ago, hour, minute),
        })
        inference_ids.extend([req_id, res_id])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/50] inferences processed...")
        time.sleep(0.05)

    print(f"  Completed 50 inference cycles")

    # ── Phase 5: Security Alerts ──────────────────────────────────────────────
    header("Phase 5: Security Alerts")

    alert_ids = []

    # Print attack #1
    eid = emit_event("security_alert", "edge-cam-entrance", {
        "alert_type": "liveness_failure",
        "detection": "print_attack",
        "liveness_score": 0.187,
        "camera_id": "cam-entrance",
        "camera_name": "Terminal 5 Main Entrance",
        "alert_level": "high",
        "subject_id": "unknown",
        "response": "frame_flagged_for_review",
        "timestamp_ms": ts_offset(15, 14, 32),
    })
    alert_ids.append(eid)
    print(f"  ALERT: Print attack at entrance (liveness=0.187)")
    time.sleep(0.05)

    # Print attack #2
    eid = emit_event("security_alert", "edge-cam-gate-a", {
        "alert_type": "liveness_failure",
        "detection": "print_attack",
        "liveness_score": 0.224,
        "camera_id": "cam-gate-a",
        "camera_name": "Gate A12 Boarding",
        "alert_level": "high",
        "subject_id": "unknown",
        "response": "security_notified",
        "timestamp_ms": ts_offset(8, 11, 17),
    })
    alert_ids.append(eid)
    print(f"  ALERT: Print attack at Gate A12 (liveness=0.224)")
    time.sleep(0.05)

    # Unknown person alert
    eid = emit_event("security_alert", "edge-cam-lounge", {
        "alert_type": "unknown_person",
        "camera_id": "cam-lounge",
        "camera_name": "Executive Lounge Entry",
        "alert_level": "medium",
        "subject_id": "unknown",
        "confidence": 0.148,
        "embedding_distance": 0.912,
        "response": "lounge_staff_alerted",
        "description": "Unregistered individual attempted lounge access",
        "timestamp_ms": ts_offset(5, 19, 45),
    })
    alert_ids.append(eid)
    print(f"  ALERT: Unknown person at Executive Lounge")

    # ── Phase 6: Bias Audit ───────────────────────────────────────────────────
    header("Phase 6: Fairness Audit — Demographics")

    audit_age_gender = run_fairness_audit(
        groups=[
            {"group_name": "male_18-30", "true_positives": 312, "false_positives": 5, "true_negatives": 318, "false_negatives": 8},
            {"group_name": "female_18-30", "true_positives": 308, "false_positives": 4, "true_negatives": 320, "false_negatives": 11},
            {"group_name": "male_30-60", "true_positives": 405, "false_positives": 8, "true_negatives": 410, "false_negatives": 14},
            {"group_name": "female_30-60", "true_positives": 398, "false_positives": 6, "true_negatives": 415, "false_negatives": 18},
            {"group_name": "male_60+", "true_positives": 145, "false_positives": 4, "true_negatives": 152, "false_negatives": 12},
            {"group_name": "female_60+", "true_positives": 138, "false_positives": 5, "true_negatives": 149, "false_negatives": 15},
        ],
        metadata={
            "dataset": "airport-enrolled-eval",
            "model": "arcface-mobilenet-airport-v3.0",
            "audit_type": "age_gender",
            "location": "Heathrow T5",
        },
    )

    audit_ethnicity = run_fairness_audit(
        groups=[
            {"group_name": "european", "true_positives": 520, "false_positives": 10, "true_negatives": 530, "false_negatives": 18},
            {"group_name": "south_asian", "true_positives": 315, "false_positives": 7, "true_negatives": 322, "false_negatives": 14},
            {"group_name": "east_asian", "true_positives": 285, "false_positives": 6, "true_negatives": 290, "false_negatives": 12},
            {"group_name": "african", "true_positives": 198, "false_positives": 5, "true_negatives": 205, "false_negatives": 10},
            {"group_name": "middle_eastern", "true_positives": 175, "false_positives": 4, "true_negatives": 180, "false_negatives": 8},
        ],
        metadata={
            "dataset": "airport-enrolled-eval",
            "model": "arcface-mobilenet-airport-v3.0",
            "audit_type": "ethnicity",
            "location": "Heathrow T5",
        },
    )

    # ── Phase 7: Federated Learning ───────────────────────────────────────────
    header("Phase 7: Federated Learning — 3 Airport Nodes")

    fl_result = post("/fl/start", {
        "n_clients": 3,
        "n_rounds": 5,
        "enable_poisoning_detection": True,
        "poisoning_threshold": 3.0,
        "seed": 42,
    })
    if fl_result:
        COUNTS["fl_simulations"] += 1
        print(f"  FL simulation: {fl_result.get('simulation_id', '?')[:16]}...")
        print(f"  Nodes: Heathrow-T5, Gatwick-South, Stansted")

    # FL round events
    fl_event_ids = emit_events([
        ("fl_round", "fl-coordinator-airport", {
            "round": r,
            "n_clients": 3,
            "aggregation": "FedAvg",
            "nodes": ["heathrow-t5", "gatwick-south", "stansted"],
            "accuracy": round(0.9890 + r * 0.0015, 4),
        })
        for r in range(1, 6)
    ])
    print(f"  5 FL rounds completed")

    # ── Phase 8: Provenance DAG ───────────────────────────────────────────────
    header("Phase 8: Provenance DAG")

    prov_dataset = add_provenance("dataset", "airport-enrolled-faces-v3", metadata={
        "n_subjects": 12480, "n_images": 87360, "source": "airport-enrollment",
    })
    print(f"  Dataset node")

    prov_model = add_provenance(
        "model", "arcface-mobilenet-airport-v3.0",
        parents=[prov_dataset], relations=["derived_from"],
        metadata={"framework": "PyTorch", "params": "4.2M", "accuracy": 0.9934},
    )
    print(f"  Model node (derived_from dataset)")

    prov_deployment = add_provenance(
        "deployment", "heathrow-t5-camera-fleet",
        parents=[prov_model], relations=["used"],
        metadata={"n_cameras": 5, "location": "Heathrow Terminal 5"},
    )
    print(f"  Deployment node (used model)")

    prov_inference = add_provenance(
        "inference_session", "airport-inference-session-2026-03",
        parents=[prov_deployment], relations=["generated_by"],
        metadata={"n_inferences": 50, "period": "2026-02-15 to 2026-03-14"},
    )
    print(f"  Inference session node")

    prov_audit = add_provenance(
        "audit", "airport-fairness-audit-2026-03",
        parents=[prov_model], relations=["attributed_to"],
        metadata={"type": "fairness", "result": "compliant"},
    )
    print(f"  Audit node (attributed_to model)")

    # ── Phase 9: Forensic Bundles ─────────────────────────────────────────────
    header("Phase 9: Forensic Bundles")

    bundle_training = create_bundle(
        training_events + deployment_events,
        provenance_ids=[prov_dataset, prov_model, prov_deployment],
        label="Training Pipeline",
    )

    # Pick a subset of inference IDs
    valid_inference = [eid for eid in inference_ids if eid][:20]
    bundle_inference = create_bundle(
        valid_inference,
        provenance_ids=[prov_deployment, prov_inference],
        label="Inference Session",
    )

    bundle_security = create_bundle(
        alert_ids,
        label="Security Incidents",
    )

    # ── Phase 10: Verification & ZK Proofs ────────────────────────────────────
    header("Phase 10: Bundle Verification & ZK Proofs")

    for bundle in [bundle_training, bundle_inference, bundle_security]:
        if bundle:
            verify_bundle(bundle)
            zk_prove(bundle)
            time.sleep(0.1)

    # ── Phase 11: Compliance Report ───────────────────────────────────────────
    header("Phase 11: Compliance Report")
    generate_compliance_report()


# ─── Scenario 2: Missing Persons Cold Case ───────────────────────────────────

def scenario_coldcase():
    """Law enforcement uses FriendlyFace to search for a missing person."""
    header("SCENARIO: Missing Persons Cold Case")
    print("  Story: Detective Sarah Chen uses FriendlyFace to locate a missing")
    print("  person (Case #MP-2026-0847) across city surveillance footage.\n")

    # ── Phase 1: Case Setup & Warrant ─────────────────────────────────────────
    header("Phase 1: Case Initialization & Legal Authorization")

    # Warrant-based consent
    grant_consent("case-mp-2026-0847", "law_enforcement_search", "warrant-system")
    print(f"  Warrant consent granted for Case #MP-2026-0847")
    grant_consent("case-mp-2026-0847", "facial_recognition", "warrant-system")
    print(f"  FR authorization granted")
    grant_consent("det-sarah-chen", "system_access", "le-admin")
    print(f"  Detective Chen access authorized")

    case_init_events = emit_events([
        ("consent_recorded", "warrant-system", {
            "case_id": "MP-2026-0847",
            "warrant_number": "WR-2026-DC-44821",
            "jurisdiction": "US-DC",
            "scope": "facial_recognition_search",
            "issued_by": "Judge M. Thompson",
            "valid_until": ts_iso(-30),  # 30 days from now
            "subject_name": "REDACTED",
            "detective": "Det. Sarah Chen, Badge #4412",
        }),
        ("training_start", "le-search-pipeline", {
            "purpose": "gallery_enrollment",
            "dataset": "reference-photos-mp-2026-0847",
            "n_images": 8,
            "source": "family_provided + DMV_record",
            "case_id": "MP-2026-0847",
        }),
        ("training_complete", "le-search-pipeline", {
            "purpose": "embedding_generation",
            "model": "ArcFace-ResNet100-LE",
            "n_embeddings": 8,
            "embedding_dim": 512,
            "quality_scores": [0.94, 0.91, 0.88, 0.96, 0.92, 0.87, 0.93, 0.90],
            "average_quality": 0.914,
            "case_id": "MP-2026-0847",
        }),
    ])
    for eid in case_init_events:
        if eid:
            print(f"  Event: {eid[:12]}...")

    # ── Phase 2: Upload Reference Photos ──────────────────────────────────────
    header("Phase 2: Reference Photo Analysis")

    REFERENCE_SOURCES = [
        ("ref-photo-01", "family_album", "front_face", 0.94),
        ("ref-photo-02", "family_album", "three_quarter_left", 0.91),
        ("ref-photo-03", "dmv_record", "front_face", 0.88),
        ("ref-photo-04", "social_media", "front_face", 0.96),
        ("ref-photo-05", "social_media", "profile_right", 0.92),
        ("ref-photo-06", "workplace_badge", "front_face", 0.87),
        ("ref-photo-07", "surveillance_last_seen", "three_quarter_right", 0.93),
        ("ref-photo-08", "surveillance_last_seen", "partial_occlusion", 0.90),
    ]

    ref_event_ids = []
    for photo_id, source, angle, quality in REFERENCE_SOURCES:
        eid = emit_event("inference_request", "le-gallery-service", {
            "purpose": "reference_enrollment",
            "photo_id": photo_id,
            "source": source,
            "face_angle": angle,
            "quality_score": quality,
            "case_id": "MP-2026-0847",
        })
        ref_event_ids.append(eid)
        print(f"  Enrolled: {photo_id} ({source}, {angle}, quality={quality})")
        time.sleep(0.05)

    # ── Phase 3: Surveillance Search ──────────────────────────────────────────
    header("Phase 3: Surveillance Footage Search (30 captures, 8 locations)")

    LOCATIONS = [
        ("loc-metro-union", "Union Station Metro", "transit"),
        ("loc-metro-gallery", "Gallery Place Metro", "transit"),
        ("loc-atm-main-st", "ATM 1420 Main St", "financial"),
        ("loc-gas-station", "QuikStop Gas I-66", "commercial"),
        ("loc-grocery-store", "Safeway Georgetown", "commercial"),
        ("loc-hospital-er", "GW Hospital ER Entrance", "medical"),
        ("loc-park-cam", "Rock Creek Park Trail Cam", "public"),
        ("loc-traffic-cam", "K St & 14th NW Traffic Cam", "traffic"),
    ]

    random.seed(847)
    search_event_ids = []

    for i in range(30):
        loc_id, loc_name, loc_type = random.choice(LOCATIONS)
        days_ago = random.randint(1, 28)
        hour = random.randint(6, 23)
        minute = random.randint(0, 59)

        eid = emit_event("inference_request", "le-search-engine", {
            "purpose": "surveillance_search",
            "location_id": loc_id,
            "location_name": loc_name,
            "location_type": loc_type,
            "capture_timestamp": ts_iso(days_ago, hour, minute),
            "case_id": "MP-2026-0847",
            "frame_quality": round(random.uniform(0.55, 0.95), 2),
        })
        search_event_ids.append(eid)
        time.sleep(0.05)

    print(f"  Searched 30 surveillance captures across 8 locations")

    # ── Phase 4: Matches ──────────────────────────────────────────────────────
    header("Phase 4: Potential Matches (5 hits)")

    MATCHES = [
        {
            "match_id": "match-001",
            "location": "loc-metro-gallery",
            "location_name": "Gallery Place Metro",
            "confidence": 0.92,
            "embed_dist": 0.087,
            "days_ago": 3,
            "hour": 14,
            "minute": 23,
            "notes": "Strong frontal match, good lighting",
        },
        {
            "match_id": "match-002",
            "location": "loc-grocery-store",
            "location_name": "Safeway Georgetown",
            "confidence": 0.78,
            "embed_dist": 0.156,
            "days_ago": 7,
            "hour": 19,
            "minute": 41,
            "notes": "Partial profile, wearing hat",
        },
        {
            "match_id": "match-003",
            "location": "loc-traffic-cam",
            "location_name": "K St & 14th NW Traffic Cam",
            "confidence": 0.63,
            "embed_dist": 0.247,
            "days_ago": 12,
            "hour": 8,
            "minute": 15,
            "notes": "Low resolution, motion blur",
        },
        {
            "match_id": "match-004",
            "location": "loc-park-cam",
            "location_name": "Rock Creek Park Trail Cam",
            "confidence": 0.45,
            "embed_dist": 0.389,
            "days_ago": 18,
            "hour": 6,
            "minute": 52,
            "notes": "IR camera, partial face, possible match",
        },
        {
            "match_id": "match-005",
            "location": "loc-atm-main-st",
            "location_name": "ATM 1420 Main St",
            "confidence": 0.87,
            "embed_dist": 0.112,
            "days_ago": 5,
            "hour": 22,
            "minute": 8,
            "notes": "Clear frontal, ATM camera, night lighting",
        },
    ]

    match_event_ids = []
    for m in MATCHES:
        eid = emit_event("inference_result", "le-search-engine", {
            "case_id": "MP-2026-0847",
            "match_id": m["match_id"],
            "subject_id": "case-mp-2026-0847",
            "location_id": m["location"],
            "location_name": m["location_name"],
            "confidence": m["confidence"],
            "embedding_distance": m["embed_dist"],
            "threshold": 0.35,
            "liveness_score": None,  # Surveillance footage, no liveness
            "capture_timestamp": ts_iso(m["days_ago"], m["hour"], m["minute"]),
            "analyst_notes": m["notes"],
        })
        match_event_ids.append(eid)
        confidence_bar = "█" * int(m["confidence"] * 20)
        print(f"  {m['match_id']}: {m['confidence']:.2f} {confidence_bar} @ {m['location_name']}")
        print(f"           {m['notes']}")
        time.sleep(0.05)

    # ── Phase 5: Explanations ─────────────────────────────────────────────────
    header("Phase 5: Explainability Analysis for Each Match")

    explanation_ids = []
    XAI_METHODS = ["LIME", "SHAP", "SDD"]

    for m in MATCHES:
        for method in XAI_METHODS:
            if method == "LIME":
                features = ["periocular_region", "nose_bridge", "jawline_contour"]
                detail = {"num_superpixels": 50, "num_samples": 1000}
            elif method == "SHAP":
                features = ["eye_spacing", "nose_width", "chin_shape"]
                detail = {"num_samples": 128, "background_dataset": "le-calibration"}
            else:
                features = ["left_eye", "right_eye", "nose_tip"]
                detail = {"saliency_threshold": 0.3, "map_resolution": "112x112"}

            eid = emit_event("explanation_generated", "xai-service-le", {
                "case_id": "MP-2026-0847",
                "match_id": m["match_id"],
                "method": method,
                "top_features": features,
                "prediction_confidence": m["confidence"],
                **detail,
            })
            explanation_ids.append(eid)
            time.sleep(0.05)

        print(f"  {m['match_id']}: LIME + SHAP + SDD explanations generated")

    # ── Phase 6: Security Alert ───────────────────────────────────────────────
    header("Phase 6: Security — Unauthorized Access Attempt")

    alert_id = emit_event("security_alert", "access-control", {
        "alert_type": "unauthorized_access",
        "case_id": "MP-2026-0847",
        "attempted_by": "unknown-session-x9f2",
        "ip_address": "198.51.100.42",
        "resource": "/api/v1/cases/MP-2026-0847/matches",
        "alert_level": "critical",
        "response": "session_terminated_ip_blocked",
        "description": "Unauthorized attempt to access case match data",
        "timestamp": ts_iso(2, 3, 17),
    })
    print(f"  CRITICAL: Unauthorized access attempt blocked")

    # ── Phase 7: Provenance DAG ───────────────────────────────────────────────
    header("Phase 7: Provenance DAG")

    prov_refs = add_provenance("dataset", "reference-photos-mp-2026-0847", metadata={
        "n_photos": 8, "sources": ["family", "dmv", "social_media", "surveillance"],
        "case_id": "MP-2026-0847",
    })
    print(f"  Reference photos node")

    prov_model = add_provenance(
        "model", "ArcFace-ResNet100-LE",
        parents=[prov_refs], relations=["used"],
        metadata={"type": "law_enforcement_search", "embedding_dim": 512},
    )
    print(f"  Search model node")

    prov_matches = add_provenance(
        "analysis", "match-analysis-mp-2026-0847",
        parents=[prov_model], relations=["generated_by"],
        metadata={"n_matches": 5, "confidence_range": "0.45-0.92"},
    )
    print(f"  Match analysis node")

    prov_xai = add_provenance(
        "explanation", "explanations-mp-2026-0847",
        parents=[prov_matches], relations=["derived_from"],
        metadata={"methods": ["LIME", "SHAP", "SDD"], "n_explanations": 15},
    )
    print(f"  Explanations node")

    # ── Phase 8: Forensic Bundles ─────────────────────────────────────────────
    header("Phase 8: Forensic Bundles")

    bundle_evidence = create_bundle(
        case_init_events + ref_event_ids,
        provenance_ids=[prov_refs, prov_model],
        label="Case Evidence",
    )

    bundle_matches = create_bundle(
        match_event_ids + explanation_ids[:6],  # First 2 matches' explanations
        provenance_ids=[prov_matches, prov_xai],
        label="Match Analysis",
    )

    # ── Phase 9: Verification & ZK ───────────────────────────────────────────
    header("Phase 9: Bundle Verification & ZK Proofs")

    for bundle in [bundle_evidence, bundle_matches]:
        if bundle:
            verify_bundle(bundle)
            zk_prove(bundle)
            time.sleep(0.1)

    # ── Phase 10: Compliance ──────────────────────────────────────────────────
    header("Phase 10: Law Enforcement Compliance Report")
    generate_compliance_report()


# ─── Scenario 3: Adversarial Attack Detection ────────────────────────────────

def scenario_adversarial():
    """Detect and respond to adversarial attacks on a FR system."""
    header("SCENARIO: Adversarial Attack Detection & Response")
    print("  Story: A corporate facial recognition system detects a coordinated")
    print("  adversarial attack — deepfakes, FL poisoning, and data exfiltration.\n")

    # ── Phase 1: Normal Operations Baseline ───────────────────────────────────
    header("Phase 1: Normal Operations Baseline (20 inferences)")

    random.seed(999)
    NORMAL_CAMERAS = [
        ("cam-lobby", "Main Lobby"),
        ("cam-elevator", "Elevator Bank"),
        ("cam-parking", "Parking Garage L1"),
        ("cam-server", "Server Room Entry"),
    ]

    EMPLOYEES = [
        f"emp-{str(i).zfill(4)}" for i in range(1, 16)
    ]

    # Consent for all employees
    for emp in EMPLOYEES:
        grant_consent(emp, "workplace_access", "hr-enrollment-system")
        time.sleep(0.03)
    print(f"  Consent granted for {len(EMPLOYEES)} employees")

    normal_ids = []
    for i in range(20):
        cam_id, cam_name = random.choice(NORMAL_CAMERAS)
        emp = random.choice(EMPLOYEES)
        days_ago = random.randint(20, 28)
        hour = random.randint(7, 18)

        req_id = emit_event("inference_request", f"edge-{cam_id}", {
            "camera_id": cam_id,
            "camera_name": cam_name,
            "frame_id": 50000 + i * 100,
            "resolution": "1920x1080",
            "timestamp_ms": ts_offset(days_ago, hour, random.randint(0, 59)),
            "phase": "normal_operations",
        })

        res_id = emit_event("inference_result", f"edge-{cam_id}", {
            "subject_id": emp,
            "confidence": round(random.uniform(0.93, 0.99), 4),
            "latency_ms": random.randint(25, 50),
            "camera_id": cam_id,
            "liveness_score": round(random.uniform(0.97, 0.999), 3),
            "embedding_distance": round(random.uniform(0.04, 0.18), 4),
            "threshold": 0.35,
            "phase": "normal_operations",
        })
        normal_ids.extend([req_id, res_id])
        time.sleep(0.05)

    print(f"  20 normal inferences completed (all high confidence)")

    # Pre-attack bias audit
    header("Phase 1b: Pre-Attack Bias Audit (Baseline)")

    audit_pre = run_fairness_audit(
        groups=[
            {"group_name": "male_under_40", "true_positives": 280, "false_positives": 4, "true_negatives": 285, "false_negatives": 6},
            {"group_name": "female_under_40", "true_positives": 275, "false_positives": 3, "true_negatives": 282, "false_negatives": 8},
            {"group_name": "male_over_40", "true_positives": 260, "false_positives": 5, "true_negatives": 268, "false_negatives": 10},
            {"group_name": "female_over_40", "true_positives": 255, "false_positives": 4, "true_negatives": 265, "false_negatives": 12},
        ],
        metadata={
            "model": "ArcFace-ResNet50-corporate-v2",
            "audit_type": "pre_attack_baseline",
            "phase": "normal_operations",
        },
    )

    # ── Phase 2: Attack — Deepfake Attempts ───────────────────────────────────
    header("Phase 2: ATTACK — Deepfake Liveness Failures (5 attempts)")

    deepfake_ids = []
    DEEPFAKE_ATTACKS = [
        {"camera": "cam-server", "liveness": 0.142, "method": "deepfake_video", "target": "emp-0001"},
        {"camera": "cam-server", "liveness": 0.198, "method": "deepfake_video", "target": "emp-0003"},
        {"camera": "cam-lobby", "liveness": 0.087, "method": "deepfake_3d_mask", "target": "emp-0007"},
        {"camera": "cam-elevator", "liveness": 0.231, "method": "deepfake_replay", "target": "emp-0012"},
        {"camera": "cam-parking", "liveness": 0.165, "method": "deepfake_face_swap", "target": "emp-0005"},
    ]

    for j, attack in enumerate(DEEPFAKE_ATTACKS):
        # The spoofed inference
        eid_req = emit_event("inference_request", f"edge-{attack['camera']}", {
            "camera_id": attack["camera"],
            "frame_id": 70000 + j * 50,
            "resolution": "1920x1080",
            "timestamp_ms": ts_offset(10, 2 + j, 15 * j),
            "phase": "attack_deepfake",
        })

        eid_res = emit_event("inference_result", f"edge-{attack['camera']}", {
            "subject_id": attack["target"],
            "confidence": round(random.uniform(0.75, 0.92), 4),
            "latency_ms": random.randint(35, 70),
            "camera_id": attack["camera"],
            "liveness_score": attack["liveness"],
            "embedding_distance": round(random.uniform(0.10, 0.25), 4),
            "threshold": 0.35,
            "liveness_failed": True,
            "phase": "attack_deepfake",
        })

        # Security alert
        eid_alert = emit_event("security_alert", f"edge-{attack['camera']}", {
            "alert_type": "liveness_failure",
            "detection": attack["method"],
            "liveness_score": attack["liveness"],
            "camera_id": attack["camera"],
            "target_identity": attack["target"],
            "alert_level": "critical",
            "response": "access_denied_security_alerted",
            "timestamp_ms": ts_offset(10, 2 + j, 15 * j + 1),
            "phase": "attack_deepfake",
        })
        deepfake_ids.extend([eid_req, eid_res, eid_alert])
        print(f"  ATTACK #{j+1}: {attack['method']} targeting {attack['target']} "
              f"(liveness={attack['liveness']}) -> BLOCKED")
        time.sleep(0.05)

    # Quarantine event
    quarantine_id = emit_event("security_alert", "security-orchestrator", {
        "alert_type": "system_quarantine",
        "trigger": "5_consecutive_liveness_failures",
        "action": "elevated_security_mode",
        "cameras_affected": ["cam-server", "cam-lobby", "cam-elevator", "cam-parking"],
        "additional_verification": "badge_tap_required",
        "timestamp_ms": ts_offset(10, 7, 30),
        "phase": "attack_response",
    })
    deepfake_ids.append(quarantine_id)
    print(f"\n  RESPONSE: System enters elevated security mode")

    # ── Phase 3: Attack — FL Poisoning ────────────────────────────────────────
    header("Phase 3: ATTACK — Federated Learning Poisoning Attempt")

    fl_poison = post("/fl/start", {
        "n_clients": 5,
        "n_rounds": 4,
        "enable_poisoning_detection": True,
        "poisoning_threshold": 2.5,
        "seed": 666,
    })
    if fl_poison:
        COUNTS["fl_simulations"] += 1
        print(f"  FL simulation with poisoning detection: {fl_poison.get('simulation_id', '?')[:16]}...")

    fl_attack_ids = []
    for r in range(1, 5):
        if r == 2:
            # Poisoned round
            eid = emit_event("fl_round", "fl-coordinator-corp", {
                "round": r,
                "n_clients": 5,
                "aggregation": "FedAvg",
                "accuracy": 0.9812,
                "anomaly_detected": True,
                "poisoned_client": "client-node-3",
                "anomaly_score": 4.8,
                "threshold": 2.5,
                "action": "client_excluded",
                "phase": "attack_fl_poisoning",
            })
            fl_attack_ids.append(eid)
            print(f"  Round {r}: POISONING DETECTED (client-node-3, anomaly=4.8)")
        elif r == 3:
            eid = emit_event("fl_round", "fl-coordinator-corp", {
                "round": r,
                "n_clients": 4,  # Reduced after exclusion
                "aggregation": "FedAvg",
                "accuracy": 0.9867,
                "note": "client-node-3 excluded",
                "phase": "attack_fl_poisoning",
            })
            fl_attack_ids.append(eid)
            print(f"  Round {r}: Continuing without compromised node (4 clients)")
        else:
            eid = emit_event("fl_round", "fl-coordinator-corp", {
                "round": r,
                "n_clients": 5 if r == 1 else 4,
                "aggregation": "FedAvg",
                "accuracy": round(0.9840 + r * 0.002, 4),
                "phase": "attack_fl_poisoning" if r > 1 else "normal_operations",
            })
            fl_attack_ids.append(eid)
            print(f"  Round {r}: Normal ({5 if r == 1 else 4} clients, "
                  f"accuracy={0.9840 + r * 0.002:.4f})")
        time.sleep(0.05)

    # ── Phase 4: Attack — Data Exfiltration ───────────────────────────────────
    header("Phase 4: ATTACK — Data Exfiltration Attempt")

    exfil_ids = []

    eid = emit_event("security_alert", "network-ids", {
        "alert_type": "data_exfiltration_attempt",
        "source_ip": "10.0.5.42",
        "destination_ip": "203.0.113.99",
        "data_targeted": "embedding_database",
        "volume_attempted_mb": 847,
        "alert_level": "critical",
        "response": "connection_terminated",
        "firewall_rule": "BLOCK-EXFIL-001",
        "timestamp_ms": ts_offset(9, 3, 12),
        "phase": "attack_exfiltration",
    })
    exfil_ids.append(eid)
    print(f"  CRITICAL: Exfiltration attempt from 10.0.5.42 -> 203.0.113.99")
    print(f"            Target: embedding database (847 MB)")
    print(f"            Response: Connection terminated")
    time.sleep(0.05)

    eid = emit_event("security_alert", "security-orchestrator", {
        "alert_type": "security_lockdown",
        "trigger": "data_exfiltration_attempt",
        "actions_taken": [
            "all_external_api_access_suspended",
            "embedding_database_locked",
            "incident_response_team_notified",
            "forensic_snapshot_captured",
        ],
        "lockdown_duration_minutes": 120,
        "timestamp_ms": ts_offset(9, 3, 13),
        "phase": "attack_response",
    })
    exfil_ids.append(eid)
    print(f"  LOCKDOWN: System in security lockdown (120 min)")

    eid = emit_event("security_alert", "incident-response", {
        "alert_type": "incident_investigation",
        "findings": {
            "compromised_node": "client-node-3",
            "attack_vector": "compromised_fl_client_lateral_movement",
            "data_exfiltrated": "none",
            "root_cause": "unpatched_dependency_CVE-2026-1234",
        },
        "timestamp_ms": ts_offset(9, 5, 0),
        "phase": "investigation",
    })
    exfil_ids.append(eid)
    print(f"  Investigation: Root cause = CVE-2026-1234 on client-node-3")

    # ── Phase 5: Model Rollback & Recovery ────────────────────────────────────
    header("Phase 5: Model Rollback & Recovery")

    recovery_ids = []

    eid = emit_event("model_registered", "model-recovery", {
        "action": "rollback",
        "from_version": "corporate-v2.3-contaminated",
        "to_version": "corporate-v2.2-verified",
        "reason": "fl_poisoning_detected",
        "model_hash_before": "sha256:deadbeef...",
        "model_hash_after": "sha256:a1b2c3d4...",
        "verified_integrity": True,
        "timestamp_ms": ts_offset(9, 6, 0),
        "phase": "recovery",
    })
    recovery_ids.append(eid)
    print(f"  Model rolled back: v2.3 -> v2.2 (verified)")
    time.sleep(0.05)

    # DP-FL retraining
    fl_dp_result = post("/fl/dp-start", {
        "n_clients": 4,
        "n_rounds": 3,
        "epsilon": 0.5,
        "delta": 1e-6,
        "max_grad_norm": 0.5,
        "seed": 42,
    })
    if fl_dp_result:
        COUNTS["fl_simulations"] += 1
        print(f"  DP-FL retraining: epsilon=0.5, delta=1e-6")
        print(f"    Simulation: {fl_dp_result.get('simulation_id', '?')[:16]}...")

    # Recovery FL rounds
    for r in range(1, 4):
        eid = emit_event("fl_round", "fl-coordinator-corp", {
            "round": r,
            "n_clients": 4,
            "aggregation": "DP-FedAvg",
            "epsilon": 0.5,
            "delta": 1e-6,
            "accuracy": round(0.9800 + r * 0.004, 4),
            "phase": "recovery_retraining",
        })
        recovery_ids.append(eid)
        print(f"  Recovery FL round {r}: accuracy={0.9800 + r * 0.004:.4f}")
        time.sleep(0.05)

    eid = emit_event("training_complete", "fl-coordinator-corp", {
        "model": "ArcFace-ResNet50-corporate-v2.4-hardened",
        "accuracy": 0.9920,
        "training_method": "DP-FedAvg",
        "epsilon": 0.5,
        "n_clients": 4,
        "hardening": ["gradient_clipping", "anomaly_detection", "client_attestation"],
        "phase": "recovery",
    })
    recovery_ids.append(eid)
    print(f"  New hardened model trained: v2.4-hardened (accuracy=0.9920)")

    eid = emit_event("model_registered", "model-recovery", {
        "model_id": "ArcFace-ResNet50-corporate-v2.4-hardened",
        "version": "2.4.0-hardened",
        "accuracy": 0.9920,
        "security_features": [
            "dp_training",
            "poisoning_detection",
            "gradient_clipping",
            "client_attestation",
        ],
        "phase": "recovery",
    })
    recovery_ids.append(eid)
    print(f"  Hardened model registered: v2.4.0-hardened")

    # ── Phase 6: Post-Attack Bias Audit ───────────────────────────────────────
    header("Phase 6: Post-Attack Bias Audit (Integrity Verification)")

    audit_post = run_fairness_audit(
        groups=[
            {"group_name": "male_under_40", "true_positives": 278, "false_positives": 5, "true_negatives": 283, "false_negatives": 7},
            {"group_name": "female_under_40", "true_positives": 273, "false_positives": 4, "true_negatives": 280, "false_negatives": 9},
            {"group_name": "male_over_40", "true_positives": 258, "false_positives": 6, "true_negatives": 266, "false_negatives": 11},
            {"group_name": "female_over_40", "true_positives": 253, "false_positives": 5, "true_negatives": 263, "false_negatives": 13},
        ],
        metadata={
            "model": "ArcFace-ResNet50-corporate-v2.4-hardened",
            "audit_type": "post_attack_integrity_check",
            "comparison_baseline": "pre_attack",
            "integrity_maintained": True,
            "phase": "recovery_verification",
        },
    )
    print(f"  Model integrity verified: fairness metrics within baseline tolerance")

    # ── Phase 7: Provenance DAG ───────────────────────────────────────────────
    header("Phase 7: Provenance DAG")

    prov_normal = add_provenance("operations", "normal-ops-baseline", metadata={
        "n_inferences": 20, "period": "normal operations",
        "avg_confidence": 0.96, "avg_liveness": 0.985,
    })
    print(f"  Normal operations node")

    prov_attack = add_provenance(
        "incident", "attack-detected-2026-03",
        parents=[prov_normal], relations=["disrupted_by"],
        metadata={
            "attack_types": ["deepfake", "fl_poisoning", "data_exfiltration"],
            "duration_hours": 5,
        },
    )
    print(f"  Attack incident node (disrupted_by normal ops)")

    prov_response = add_provenance(
        "response", "incident-response-2026-03",
        parents=[prov_attack], relations=["triggered_by"],
        metadata={
            "actions": ["quarantine", "lockdown", "investigation", "rollback"],
            "response_time_minutes": 18,
        },
    )
    print(f"  Incident response node (triggered_by attack)")

    prov_recovery = add_provenance(
        "recovery", "system-recovery-2026-03",
        parents=[prov_response], relations=["resulted_from"],
        metadata={
            "model_rollback": True,
            "dp_retraining": True,
            "integrity_verified": True,
            "new_model": "v2.4.0-hardened",
        },
    )
    print(f"  Recovery node (resulted_from response)")

    # ── Phase 8: Forensic Bundles ─────────────────────────────────────────────
    header("Phase 8: Forensic Bundles")

    # Pre-attack baseline
    valid_normal = [eid for eid in normal_ids if eid][:10]
    bundle_baseline = create_bundle(
        valid_normal,
        provenance_ids=[prov_normal],
        label="Pre-Attack Baseline",
    )

    # Attack evidence
    bundle_attack = create_bundle(
        deepfake_ids + fl_attack_ids + exfil_ids,
        provenance_ids=[prov_attack, prov_response],
        label="Attack Evidence",
    )

    # Recovery verification
    bundle_recovery = create_bundle(
        recovery_ids,
        provenance_ids=[prov_response, prov_recovery],
        label="Recovery Verification",
    )

    # ── Phase 9: Verification & ZK ───────────────────────────────────────────
    header("Phase 9: Bundle Verification & ZK Proofs")

    for bundle in [bundle_baseline, bundle_attack, bundle_recovery]:
        if bundle:
            verify_bundle(bundle)
            zk_prove(bundle)
            time.sleep(0.1)

    # ── Phase 10: Compliance (Incident Response) ──────────────────────────────
    header("Phase 10: Incident Response Compliance Report")

    emit_event("compliance_report", "governance-engine", {
        "framework": "NIST-CSF",
        "incident_id": "INC-2026-03-001",
        "phases_documented": ["identify", "protect", "detect", "respond", "recover"],
        "compliant": True,
        "findings": [
            "Attack detected within 3 minutes",
            "System quarantine within 5 minutes",
            "Full lockdown within 8 minutes",
            "Root cause identified within 2 hours",
            "Recovery completed within 6 hours",
        ],
        "recommendations": [
            "Implement client attestation for FL nodes",
            "Reduce poisoning detection threshold to 2.0",
            "Add network segmentation for embedding stores",
        ],
    })
    print(f"  NIST-CSF incident report documented")

    generate_compliance_report()


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_summary():
    """Print final summary of all seeded data."""
    print(f"\n{'═' * 60}")
    print(f"  SEED COMPLETE")
    print(f"{'═' * 60}")

    # Health check
    health = get("/../health")
    if health:
        print(f"  Total events (server): {health.get('event_count', '?')}")
        root = health.get("merkle_root", "none")
        print(f"  Merkle root:           {str(root)[:24]}...")

    print(f"  ─────────────────────────────────────")
    print(f"  Events:              {COUNTS['events']}")
    print(f"  Bundles:             {COUNTS['bundles']}")
    print(f"  Provenance nodes:    {COUNTS['provenance']}")
    print(f"  Consent grants:      {COUNTS['consent_grants']}")
    print(f"  Consent revocations: {COUNTS['consent_revocations']}")
    print(f"  Fairness audits:     {COUNTS['fairness_audits']}")
    print(f"  ZK proofs:           {COUNTS['zk_proofs']}")
    print(f"  FL simulations:      {COUNTS['fl_simulations']}")
    print(f"  Verifications:       {COUNTS['verifications']}")
    print(f"  Compliance reports:  {COUNTS['compliance_reports']}")
    print(f"{'═' * 60}")
    print(f"\n  -> {BASE.replace('/api/v1', '')}")


def main():
    print(f"\n  FriendlyFace Scenario Seeder")
    print(f"  Target: {BASE}")
    if args.api_key:
        print(f"  Auth:   Bearer ***{args.api_key[-4:]}")
    print()

    # Verify connectivity
    health = get("/../health")
    if not health:
        print(f"  WARNING: Cannot reach {BASE}/../health")
        print(f"  Proceeding anyway (server may still accept writes)...\n")

    scenario = args.scenario

    if scenario in ("airport", "all"):
        scenario_airport()

    if scenario in ("coldcase", "all"):
        scenario_coldcase()

    if scenario in ("adversarial", "all"):
        scenario_adversarial()

    print_summary()


if __name__ == "__main__":
    main()
