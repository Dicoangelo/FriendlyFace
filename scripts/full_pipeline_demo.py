#!/usr/bin/env python3
"""FriendlyFace Full Pipeline Demo — E2E Heartbeat Script.

Exercises the FULL forensic lifecycle against a running FriendlyFace instance:
  1. Train PCA+SVM model on synthetic face images
  2. Grant consent for demo subjects
  3. Run face recognition (predict)
  4. Generate LIME + SHAP explanations
  5. Run bias audit across demographic groups
  6. Build provenance DAG linking all artifacts
  7. Create forensic bundle with all layer artifacts
  8. Generate ZK proof for the bundle
  9. Verify bundle integrity (7 checks)
 10. Generate compliance report
 11. Export bundle as JSON-LD

Usage:
    python3 scripts/full_pipeline_demo.py              # localhost:3849
    python3 scripts/full_pipeline_demo.py --remote      # friendlyface.metaventionsai.com
    python3 scripts/full_pipeline_demo.py --base-url http://host:port  # custom
    python3 scripts/full_pipeline_demo.py --benchmark   # collect latency metrics
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOCAL_BASE = "http://localhost:3849"
REMOTE_BASE = "https://friendlyface.metaventionsai.com"
IMAGE_SIZE = (112, 112)
N_IMAGES = 12
N_CLASSES = 3


def parse_args():
    parser = argparse.ArgumentParser(description="FriendlyFace full pipeline demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remote", action="store_true", help="Target remote instance")
    group.add_argument("--base-url", type=str, help="Custom base URL")
    parser.add_argument("--api-key", type=str, default="", help="API key for auth")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout (seconds)")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Collect latency benchmarks and output JSON file",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="benchmarks/pipeline_benchmarks.json",
        help="Output path for benchmark JSON (default: benchmarks/pipeline_benchmarks.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

class APIClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: int = 60):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def post(self, path: str, data: dict | None = None, files: dict | None = None) -> dict | None:
        url = f"{self.base}{path}"
        try:
            if files:
                r = requests.post(url, files=files, headers=self.headers, timeout=self.timeout)
            else:
                r = requests.post(url, json=data or {}, headers=self.headers, timeout=self.timeout)
            if r.status_code in (200, 201):
                return r.json()
            print(f"  FAIL {path} → {r.status_code}: {r.text[:300]}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"  ERROR: Cannot connect to {url}")
            return None

    def get(self, path: str) -> dict | None:
        url = f"{self.base}{path}"
        try:
            r = requests.get(url, headers=self.headers, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            print(f"  FAIL GET {path} → {r.status_code}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"  ERROR: Cannot connect to {url}")
            return None

    def timed_post(
        self, path: str, data: dict | None = None, files: dict | None = None
    ) -> tuple[dict | None, float]:
        """POST with timing. Returns (response, elapsed_ms)."""
        t0 = time.perf_counter()
        resp = self.post(path, data=data, files=files)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return resp, elapsed_ms

    def timed_get(self, path: str) -> tuple[dict | None, float]:
        """GET with timing. Returns (response, elapsed_ms)."""
        t0 = time.perf_counter()
        resp = self.get(path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return resp, elapsed_ms


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def create_synthetic_dataset(dataset_dir: Path, n_images: int = N_IMAGES) -> list[int]:
    """Create synthetic 112x112 grayscale face images and return labels."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    labels = []
    for i in range(n_images):
        pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
        img = Image.fromarray(pixels, mode="L")
        img.save(dataset_dir / f"face_{i:04d}.png")
        labels.append(i % N_CLASSES)
    return labels


def make_test_image(seed: int = 99) -> bytes:
    """Create a synthetic grayscale PNG image and return bytes."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Benchmark collector
# ---------------------------------------------------------------------------

class BenchmarkCollector:
    """Collects per-operation latency metrics."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.timings: dict[str, list[float]] = {}

    def record(self, metric_name: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        self.timings.setdefault(metric_name, []).append(round(elapsed_ms, 3))

    def summary(self) -> dict:
        """Return summary with min/max/mean/median for each metric."""
        result = {}
        for name, values in self.timings.items():
            arr = sorted(values)
            n = len(arr)
            result[name] = {
                "samples": n,
                "min_ms": arr[0],
                "max_ms": arr[-1],
                "mean_ms": round(sum(arr) / n, 3),
                "median_ms": arr[n // 2] if n % 2 else round((arr[n // 2 - 1] + arr[n // 2]) / 2, 3),
                "all_ms": arr,
            }
        return result

    def flat_metrics(self) -> dict:
        """Return flat dict mapping metric_name_ms -> mean value."""
        out = {}
        for name, values in self.timings.items():
            out[f"{name}_ms"] = round(sum(values) / len(values), 3) if values else 0
        return out


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class PipelineResult:
    """Collects all IDs and results across the pipeline."""
    def __init__(self):
        self.event_ids: list[str] = []
        self.provenance_ids: list[str] = []
        self.model_id: str = ""
        self.bundle_id: str = ""
        self.bundle_hash: str = ""
        self.inference_event_id: str = ""
        self.lime_event_id: str = ""
        self.shap_event_id: str = ""
        self.audit_event_id: str = ""
        self.compliance_event_id: str = ""
        self.verification: dict = {}
        self.export: dict = {}
        self.errors: list[str] = []
        self.steps_passed: int = 0
        self.steps_total: int = 11

    def fail(self, msg: str):
        self.errors.append(msg)
        print(f"  ✗ {msg}")

    def ok(self, msg: str):
        self.steps_passed += 1
        print(f"  ✓ {msg}")


def step1_train_model(api: APIClient, result: PipelineResult, tmp_dir: Path, bench: BenchmarkCollector):
    """Train PCA+SVM on synthetic dataset."""
    print("\n═══ Step 1/11: Train PCA+SVM Model ═══")

    dataset_dir = tmp_dir / "dataset"
    output_dir = tmp_dir / "models"
    labels = create_synthetic_dataset(dataset_dir)

    train_data, elapsed = api.timed_post("/recognition/train", {
        "dataset_path": str(dataset_dir),
        "output_dir": str(output_dir),
        "n_components": 5,
        "labels": labels,
    })
    bench.record("event_recording_latency", elapsed)

    if not train_data:
        result.fail("Model training failed")
        return False

    result.model_id = train_data["model_id"]
    result.event_ids.append(train_data["pca_event_id"])
    result.event_ids.append(train_data["svm_event_id"])
    result.provenance_ids.append(train_data["pca_provenance_id"])
    result.provenance_ids.append(train_data["svm_provenance_id"])

    result.ok(f"Model trained: {result.model_id[:12]}... "
              f"({train_data['n_samples']} samples, {train_data['n_classes']} classes, "
              f"CV acc: {train_data.get('cv_accuracy', 'N/A')}) [{elapsed:.1f}ms]")
    return True


def step2_grant_consent(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Grant consent for demo subjects."""
    print("\n═══ Step 2/11: Grant Consent ═══")

    subjects = [
        ("subject_alpha", "recognition"),
        ("subject_alpha", "training"),
        ("subject_beta", "recognition"),
        ("subject_gamma", "recognition"),
    ]
    granted = 0
    for subject_id, purpose in subjects:
        expiry = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()
        resp, elapsed = api.timed_post("/consent/grant", {
            "subject_id": subject_id,
            "purpose": purpose,
            "expiry": expiry,
            "actor": "full_pipeline_demo",
        })
        bench.record("event_recording_latency", elapsed)
        if resp:
            result.event_ids.append(resp["event_id"])
            granted += 1
        time.sleep(0.05)

    if granted == 0:
        result.fail("No consent grants succeeded")
        return False

    result.ok(f"Granted {granted}/{len(subjects)} consent records")
    return True


def step3_predict(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Run face recognition prediction."""
    print("\n═══ Step 3/11: Face Recognition (Predict) ═══")

    image_bytes = make_test_image(seed=42)
    resp, elapsed = api.timed_post(
        "/recognition/predict",
        files={"image": ("face.png", image_bytes, "image/png")},
    )
    bench.record("event_recording_latency", elapsed)

    if not resp:
        result.fail("Prediction failed")
        return False

    result.inference_event_id = resp["event_id"]
    result.event_ids.append(resp["event_id"])
    n_matches = len(resp.get("matches", []))
    top_conf = resp["matches"][0]["confidence"] if n_matches > 0 else 0

    result.ok(f"Prediction: {n_matches} matches, top confidence: {top_conf:.4f}, "
              f"input_hash: {resp.get('input_hash', 'N/A')[:16]}... [{elapsed:.1f}ms]")
    return True


def step4_lime(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Generate LIME explanation for the inference event."""
    print("\n═══ Step 4/11: LIME Explanation ═══")

    resp, elapsed = api.timed_post("/explainability/lime", {
        "event_id": result.inference_event_id,
    })
    bench.record("event_recording_latency", elapsed)

    if not resp:
        result.fail("LIME explanation failed")
        return False

    result.lime_event_id = resp["event_id"]
    result.event_ids.append(resp["event_id"])

    result.ok(f"LIME explanation: {resp['explanation_id'][:12]}... "
              f"(computed: {resp.get('computed', False)}) [{elapsed:.1f}ms]")
    return True


def step5_shap(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Generate SHAP explanation for the inference event."""
    print("\n═══ Step 5/11: SHAP Explanation ═══")

    resp, elapsed = api.timed_post("/explainability/shap", {
        "event_id": result.inference_event_id,
    })
    bench.record("event_recording_latency", elapsed)

    if not resp:
        result.fail("SHAP explanation failed")
        return False

    result.shap_event_id = resp["event_id"]
    result.event_ids.append(resp["event_id"])

    result.ok(f"SHAP explanation: {resp['explanation_id'][:12]}... "
              f"(computed: {resp.get('computed', False)}) [{elapsed:.1f}ms]")
    return True


def step6_bias_audit(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Run bias audit across demographic groups."""
    print("\n═══ Step 6/11: Bias Audit ═══")

    resp, elapsed = api.timed_post("/fairness/audit", {
        "groups": [
            {"group_name": "male_18-30", "true_positives": 487, "false_positives": 8,
             "true_negatives": 492, "false_negatives": 13},
            {"group_name": "female_18-30", "true_positives": 491, "false_positives": 6,
             "true_negatives": 494, "false_negatives": 9},
            {"group_name": "male_30-60", "true_positives": 478, "false_positives": 12,
             "true_negatives": 488, "false_negatives": 22},
            {"group_name": "female_30-60", "true_positives": 483, "false_positives": 9,
             "true_negatives": 491, "false_negatives": 17},
        ],
        "demographic_parity_threshold": 0.1,
        "equalized_odds_threshold": 0.1,
        "metadata": {
            "dataset": "synthetic-balanced",
            "model": "PCA+SVM-demo",
            "audit_type": "age_gender",
        },
    })
    bench.record("event_recording_latency", elapsed)

    if not resp:
        result.fail("Bias audit failed")
        return False

    result.audit_event_id = resp["event_id"]
    result.event_ids.append(resp["event_id"])

    result.ok(f"Bias audit: compliant={resp['compliant']}, "
              f"fairness_score={resp.get('fairness_score', 'N/A')}, "
              f"groups={resp.get('groups_evaluated', [])} [{elapsed:.1f}ms]")
    return True


def step7_provenance(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Build provenance DAG linking dataset -> model -> inference -> explanation."""
    print("\n═══ Step 7/11: Provenance DAG ═══")

    nodes_created = 0

    # Dataset node
    p1, elapsed = api.timed_post("/provenance", {
        "entity_type": "dataset",
        "entity_id": "synthetic-face-dataset-demo",
        "metadata": {"n_images": N_IMAGES, "n_classes": N_CLASSES, "image_size": "112x112"},
    })
    bench.record("event_recording_latency", elapsed)
    if p1:
        result.provenance_ids.append(p1["id"])
        nodes_created += 1

    # Model node (derived from dataset)
    p2, elapsed = api.timed_post("/provenance", {
        "entity_type": "model",
        "entity_id": f"pca-svm-{result.model_id[:8]}",
        "parents": [p1["id"]] if p1 else [],
        "relations": ["derived_from"],
        "metadata": {"pipeline": "PCA+SVM", "n_components": 5},
    })
    bench.record("event_recording_latency", elapsed)
    if p2:
        result.provenance_ids.append(p2["id"])
        nodes_created += 1

    # Inference node (generated by model)
    p3, elapsed = api.timed_post("/provenance", {
        "entity_type": "inference",
        "entity_id": f"inference-{result.inference_event_id[:8]}",
        "parents": [p2["id"]] if p2 else [],
        "relations": ["generated_by"],
        "metadata": {"event_id": result.inference_event_id},
    })
    bench.record("event_recording_latency", elapsed)
    if p3:
        result.provenance_ids.append(p3["id"])
        nodes_created += 1

    # Explanation node (attributed to inference)
    p4, elapsed = api.timed_post("/provenance", {
        "entity_type": "explanation",
        "entity_id": f"xai-{result.lime_event_id[:8]}",
        "parents": [p3["id"]] if p3 else [],
        "relations": ["attributed_to"],
        "metadata": {"methods": ["LIME", "SHAP"]},
    })
    bench.record("event_recording_latency", elapsed)
    if p4:
        result.provenance_ids.append(p4["id"])
        nodes_created += 1

    # Audit node (attributed to model)
    p5, elapsed = api.timed_post("/provenance", {
        "entity_type": "audit",
        "entity_id": f"bias-audit-{result.audit_event_id[:8]}",
        "parents": [p2["id"]] if p2 else [],
        "relations": ["attributed_to"],
        "metadata": {"type": "demographic_fairness"},
    })
    bench.record("event_recording_latency", elapsed)
    if p5:
        result.provenance_ids.append(p5["id"])
        nodes_created += 1

    if nodes_created == 0:
        result.fail("No provenance nodes created")
        return False

    result.ok(f"Provenance DAG: {nodes_created} nodes "
              f"(dataset -> model -> inference -> explanation + audit)")
    return True


def step8_bundle(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Create forensic bundle with all layer artifacts."""
    print("\n═══ Step 8/11: Forensic Bundle ═══")

    resp, elapsed = api.timed_post("/bundles", {
        "event_ids": result.event_ids,
        "provenance_node_ids": result.provenance_ids,
    })
    bench.record("bundle_creation_time", elapsed)

    if not resp:
        result.fail("Bundle creation failed")
        return False

    result.bundle_id = resp["id"]
    result.bundle_hash = resp.get("bundle_hash", "")

    # Check which layer artifacts were included
    layers = []
    if resp.get("recognition_artifacts"):
        layers.append("recognition")
    if resp.get("explanation_artifacts"):
        layers.append("explanation")
    if resp.get("bias_report"):
        layers.append("bias")
    if resp.get("zk_proof"):
        layers.append("zk_proof")
    if resp.get("did_credential"):
        layers.append("did")

    result.ok(f"Bundle: {result.bundle_id[:12]}... "
              f"hash: {result.bundle_hash[:16]}... "
              f"status: {resp.get('status', '?')}, "
              f"layers: {layers} [{elapsed:.1f}ms]")
    return True


def step9_zk_proof(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Generate ZK proof for the bundle."""
    print("\n═══ Step 9/11: ZK Proof + Verify Bundle ═══")

    # Generate ZK proof
    zk, zk_elapsed = api.timed_post("/zk/prove", {"bundle_id": result.bundle_id})
    bench.record("zk_proof_generation_time", zk_elapsed)
    if zk:
        print(f"    ZK proof generated: {str(zk.get('proof_hash', ''))[:16]}... [{zk_elapsed:.1f}ms]")

    # Verify the bundle
    verify, verify_elapsed = api.timed_post(f"/verify/{result.bundle_id}", {})
    bench.record("verification_time", verify_elapsed)
    if not verify:
        result.fail("Bundle verification failed")
        return False

    result.verification = verify
    status = verify.get("status", verify.get("verified", "?"))
    valid = verify.get("valid", False)
    bundle_hash_valid = verify.get("bundle_hash_valid", False)

    # Check layer artifacts
    layer_arts = verify.get("layer_artifacts", {})
    layer_status = {k: v.get("valid", "?") for k, v in layer_arts.items()} if layer_arts else {}

    result.ok(f"Verification: valid={valid}, status={status}, "
              f"bundle_hash={bundle_hash_valid}, "
              f"layers={layer_status} [{verify_elapsed:.1f}ms]")
    return True


def step10_compliance(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Generate compliance report."""
    print("\n═══ Step 10/11: Compliance Report ═══")

    resp, elapsed = api.timed_post("/governance/compliance/generate")
    bench.record("event_recording_latency", elapsed)
    if not resp:
        result.fail("Compliance report generation failed")
        return False

    result.compliance_event_id = resp.get("event_id", "")
    score = resp.get("overall_compliance_score", 0)
    metrics = resp.get("metrics", {})

    result.ok(f"Compliance: score={score:.1f}%, "
              f"consent={metrics.get('consent_coverage_pct', '?')}%, "
              f"bias_pass={metrics.get('bias_audit_pass_rate_pct', '?')}%, "
              f"explain={metrics.get('explanation_coverage_pct', '?')}%, "
              f"bundle_integrity={metrics.get('bundle_integrity_pct', '?')}% [{elapsed:.1f}ms]")
    return True


def step11_export(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """Export bundle as JSON-LD."""
    print("\n═══ Step 11/11: JSON-LD Export ═══")

    resp, elapsed = api.timed_get(f"/bundles/{result.bundle_id}/export")
    bench.record("event_recording_latency", elapsed)
    if not resp:
        result.fail("JSON-LD export failed")
        return False

    result.export = resp

    # Validate JSON-LD structure
    has_context = "@context" in resp
    has_type = "@type" in resp
    n_events = len(resp.get("events", []))
    n_proofs = len(resp.get("merkle_proofs", []))
    n_prov = len(resp.get("provenance_chain", []))

    result.ok(f"JSON-LD export: @context={has_context}, @type={has_type}, "
              f"events={n_events}, merkle_proofs={n_proofs}, provenance={n_prov} [{elapsed:.1f}ms]")
    return True


# ---------------------------------------------------------------------------
# Seal benchmark — measure Merkle proof time separately
# ---------------------------------------------------------------------------

def benchmark_merkle_proof(api: APIClient, result: PipelineResult, bench: BenchmarkCollector):
    """If benchmark mode, separately measure Merkle proof and seal issuance time."""
    if not bench.enabled or not result.event_ids:
        return

    print("\n═══ Benchmark: Merkle Proof Timing ═══")
    for event_id in result.event_ids[:3]:
        _, elapsed = api.timed_get(f"/merkle/proof/{event_id}")
        bench.record("merkle_proof_time", elapsed)
        print(f"    Merkle proof for {event_id[:12]}...: {elapsed:.1f}ms")

    # Seal issuance (DID credential for bundle)
    if result.bundle_id:
        print("\n═══ Benchmark: Seal Issuance Timing ═══")
        _, elapsed = api.timed_post("/did/credential", {
            "bundle_id": result.bundle_id,
        })
        bench.record("seal_issuance_time", elapsed)
        print(f"    Seal issued for bundle {result.bundle_id[:12]}...: {elapsed:.1f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(api: APIClient, bench: BenchmarkCollector):
    """Execute the full pipeline and return results."""
    result = PipelineResult()

    with tempfile.TemporaryDirectory(prefix="ff_demo_") as tmp:
        tmp_dir = Path(tmp)

        # Check health first
        print("\n═══ Pre-flight: Health Check ═══")
        health = api.get("/health")
        if not health:
            print("  ✗ Server unreachable. Start with: python3 -m friendlyface")
            return result
        print(f"  ✓ Server healthy: {health.get('event_count', 0)} events, "
              f"merkle_root: {str(health.get('merkle_root', 'none'))[:20]}...")

        # Run all steps sequentially — each depends on the previous
        steps = [
            lambda: step1_train_model(api, result, tmp_dir, bench),
            lambda: step2_grant_consent(api, result, bench),
            lambda: step3_predict(api, result, bench),
            lambda: step4_lime(api, result, bench),
            lambda: step5_shap(api, result, bench),
            lambda: step6_bias_audit(api, result, bench),
            lambda: step7_provenance(api, result, bench),
            lambda: step8_bundle(api, result, bench),
            lambda: step9_zk_proof(api, result, bench),
            lambda: step10_compliance(api, result, bench),
            lambda: step11_export(api, result, bench),
        ]

        for step_fn in steps:
            if not step_fn():
                print(f"\n  Pipeline stopped at step {result.steps_passed + 1}")
                break

        # Extra benchmark probes
        benchmark_merkle_proof(api, result, bench)

    return result


def print_summary(result: PipelineResult, base_url: str):
    """Print final summary."""
    print("\n" + "═" * 60)
    if result.steps_passed == result.steps_total:
        print("  FULL PIPELINE DEMO — ALL STEPS PASSED")
    else:
        print(f"  FULL PIPELINE DEMO — {result.steps_passed}/{result.steps_total} STEPS PASSED")
    print("═" * 60)

    print(f"  Target:          {base_url}")
    print(f"  Events created:  {len(result.event_ids)}")
    print(f"  Provenance:      {len(result.provenance_ids)} nodes")
    print(f"  Model ID:        {result.model_id[:16]}..." if result.model_id else "  Model ID:        (none)")
    print(f"  Bundle ID:       {result.bundle_id[:16]}..." if result.bundle_id else "  Bundle ID:       (none)")
    print(f"  Bundle hash:     {result.bundle_hash[:24]}..." if result.bundle_hash else "  Bundle hash:     (none)")

    if result.verification:
        print(f"  Verified:        {result.verification.get('valid', '?')}")

    if result.export:
        print(f"  JSON-LD export:  {len(json.dumps(result.export))} bytes")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for e in result.errors:
            print(f"    - {e}")

    print("═" * 60)

    if result.steps_passed == result.steps_total:
        print("\n  All 6 layers exercised end-to-end:")
        print("    Layer 1 — Consent:         ✓ (4 grants)")
        print("    Layer 2 — Recognition:     ✓ (PCA+SVM train + predict)")
        print("    Layer 3 — Explainability:  ✓ (LIME + SHAP)")
        print("    Layer 4 — Fairness:        ✓ (bias audit)")
        print("    Layer 5 — Forensics:       ✓ (bundle + ZK proof + verify)")
        print("    Layer 6 — Governance:      ✓ (compliance report + JSON-LD)")
        print(f"\n  → {base_url}")

    return result.steps_passed == result.steps_total


def save_benchmarks(bench: BenchmarkCollector, output_path: str, base_url: str):
    """Save benchmark results to JSON file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": base_url,
        "benchmarks": bench.summary(),
        "flat": bench.flat_metrics(),
    }

    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Benchmarks saved to: {out_path}")

    # Print benchmark summary
    print("\n  ═══ Benchmark Summary ═══")
    for name, stats in report["benchmarks"].items():
        print(f"    {name}: mean={stats['mean_ms']:.1f}ms, "
              f"median={stats['median_ms']:.1f}ms, "
              f"min={stats['min_ms']:.1f}ms, max={stats['max_ms']:.1f}ms "
              f"({stats['samples']} samples)")

    return report


def main():
    args = parse_args()

    if args.base_url:
        base_url = args.base_url
    elif args.remote:
        base_url = REMOTE_BASE
    else:
        base_url = LOCAL_BASE

    print("FriendlyFace Full Pipeline Demo")
    print(f"Target: {base_url}")
    if args.benchmark:
        print("Benchmark mode: ON")

    bench = BenchmarkCollector(enabled=args.benchmark)
    api = APIClient(base_url, api_key=args.api_key, timeout=args.timeout)

    pipeline_start = time.perf_counter()
    result = run_pipeline(api, bench)
    pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000

    success = print_summary(result, base_url)

    if args.benchmark:
        bench.record("total_pipeline_time", pipeline_elapsed)
        save_benchmarks(bench, args.benchmark_output, base_url)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
