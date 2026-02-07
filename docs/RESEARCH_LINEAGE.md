# FriendlyFace — Research Lineage

## Academic Foundation

FriendlyFace is not a typical software project — it's a research implementation. Every architectural decision traces back to a specific academic paper, and the platform serves as a living integration of multiple SOTA research threads in forensic AI, federated learning, fairness, explainability, and regulatory compliance.

---

## Primary Framework

### Mohammed, ICDF2C 2024 — Forensic-Friendly AI Schema

**Author:** Safiia Mohammed, University of Windsor  
**Venue:** International Conference on Digital Forensics and Cyber Crime (ICDF2C), 2024

This is the foundational paper. It defines the 6-layer forensic architecture that FriendlyFace implements:

| Layer | Mohammed's Schema | FriendlyFace Implementation |
|-------|------------------|---------------------------|
| Layer 1 | Recognition Engine | PCA + SVM (scikit-learn) |
| Layer 2 | Federated Learning | FedAvg simulation + poisoning detection |
| Layer 3 | Blockchain Forensic Layer | Hash-chained events, Merkle tree, provenance DAG |
| Layer 4 | Fairness Auditor | Demographic parity + equalized odds |
| Layer 5 | Explainability | LIME + KernelSHAP + SDD saliency |
| Layer 6 | Consent & Governance | Append-only consent + EU AI Act compliance |

**Key contributions integrated:**
- `ForensicEvent` — Immutable, hash-chained event records with SHA-256 linking
- Provenance DAG — Directed acyclic graph tracking the full lineage from training data through model to inference to explanation
- Forensic bundles — Self-verifiable evidence packages that can be presented in court
- The principle that every AI operation must produce a forensic record

---

## Integrated Research Papers

### BioZero (arXiv:2409.17509) — Zero-Knowledge Biometric Verification

**Integration:** Merkle tree verification structure + ZK proofs on forensic bundles

**What FriendlyFace uses from this paper:**
- The concept of Merkle tree-based integrity verification for biometric systems
- Zero-knowledge proof generation for bundle verification (now implemented with Schnorr proofs)
- The idea that biometric verification should be possible without revealing the underlying data

**Implementation status:**
- ✅ Merkle tree verification (Phase 1)
- ✅ Schnorr ZK proofs replacing stubs (Phase 2 → Phase 3)

---

### TBFL (arXiv:2602.02629) — Trustworthy Blockchain-based Federated Learning

**Integration:** DID/VC identity for FL participants

**What FriendlyFace uses from this paper:**
- Decentralized Identifiers (DIDs) for cryptographic accountability of FL participants
- Verifiable Credentials for proving identity claims without a central authority
- The pattern of anchoring FL round participation to verifiable identities

**Implementation status:**
- ✅ Ed25519 DID:key identifiers (Phase 3)
- ✅ W3C-style Verifiable Credentials (Phase 3)
- ✅ DID credentials embedded in forensic bundles (Phase 3)
- 🔜 FL round participants identified by DIDs (future)

---

### FedFDP (arXiv:2402.16028) — Fairness-aware Federated Learning with Differential Privacy

**Integration:** Differential privacy in FL + fairness-aware noise calibration

**What FriendlyFace uses from this paper:**
- Differential privacy (DP) noise injection during FL aggregation
- Epsilon/delta privacy budget tracking per simulation
- Fairness-aware clipping: demographic groups receive proportional noise to avoid amplifying existing biases

**Implementation status:**
- ✅ DP-FL simulation with configurable epsilon/delta (Phase 2)
- ✅ Privacy budget tracking per round (Phase 2)
- ✅ Fairness-aware clipping (Phase 2)

---

### SDD (arXiv:2505.03837) — Saliency-Driven Decomposition for Explainability

**Integration:** Pixel-level saliency maps for face recognition explanations

**What FriendlyFace uses from this paper:**
- SDD saliency computation that shows which specific facial regions drove a recognition decision
- Complements LIME (feature-level) and SHAP (value-level) with spatial/visual explanations
- Particularly valuable for forensic examiners who need to understand "why this face matched"

**Implementation status:**
- ✅ SDD saliency endpoint `/explainability/sdd` (Phase 2)
- ✅ Integrated into compare endpoint alongside LIME and SHAP (Phase 2)

---

### EU AI Act Compliance Framework (arXiv:2512.13907)

**Integration:** Automated compliance reporting for high-risk AI systems

**What FriendlyFace uses from this paper:**
- The classification of facial recognition as a high-risk AI system under the EU AI Act
- Required documentation elements: training data provenance, bias testing records, consent mechanisms
- Compliance report structure covering technical documentation, risk management, and human oversight requirements

**Implementation status:**
- ✅ Compliance report generation (Phase 1)
- ✅ Consent coverage checks (Phase 1)
- ✅ Bias audit recency verification (Phase 1)
- ✅ Hash chain integrity as documentation evidence (Phase 1)

---

## Research Integration Map

```
                    Mohammed ICDF2C 2024
                    (Forensic Architecture)
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     BioZero          TBFL              FedFDP
  (ZK + Merkle)    (DID/VC)        (DP + Fairness)
          │                │                │
          │           Ed25519 DID      DP-FL Engine
     Schnorr ZK       + VC Issuer     + Privacy Budget
     + Merkle Tree                    + Fair Clipping
          │                │                │
          └────────┬───────┘                │
                   │                        │
              ForensicBundle ◄──────────────┘
              (ZK + DID + Merkle)
                   │
          ┌────────┴────────┐
          │                 │
        SDD           EU AI Act
    (Saliency)      (Compliance)
          │                 │
     XAI Compare      Compliance
     Endpoint          Reporter
```

---

## Citation Guide

If referencing FriendlyFace in academic work:

```
FriendlyFace implements Mohammed's forensic-friendly AI framework [1] with
integrity verification from BioZero [2], decentralized identity from TBFL [3],
fairness-aware differential privacy from FedFDP [4], saliency-based
explainability from SDD [5], and EU AI Act compliance per [6].

[1] Mohammed, S. (2024). [Forensic-Friendly AI Framework]. ICDF2C 2024.
[2] [BioZero]. arXiv:2409.17509.
[3] [TBFL]. arXiv:2602.02629.
[4] [FedFDP]. arXiv:2402.16028.
[5] [SDD]. arXiv:2505.03837.
[6] [EU AI Act Analysis]. arXiv:2512.13907.
```

---

## How to Read the Codebase Through a Research Lens

If you're a researcher trying to understand how FriendlyFace maps to the papers:

| If you're interested in... | Read these files |
|---------------------------|-----------------|
| Mohammed's hash chain schema | `core/models.py` (ForensicEvent), `core/service.py` (record_event) |
| Merkle tree verification (BioZero) | `core/merkle.py`, `core/service.py` (initialize, get_merkle_proof) |
| Provenance DAG | `core/provenance.py`, `core/service.py` (add_provenance_node) |
| DID/VC identity (TBFL) | `crypto/did.py`, `crypto/vc.py`, `core/service.py` (create_bundle) |
| ZK proofs (BioZero) | `crypto/schnorr.py`, `core/service.py` (create_bundle, verify_bundle) |
| Fairness auditing (FedFDP) | `fairness/auditor.py` |
| Differential privacy FL (FedFDP) | `fl/engine.py` (DP-FL section), `api/app.py` (/fl/dp-start) |
| SDD saliency (SDD paper) | `explainability/sdd_explain.py` |
| EU AI Act compliance | `governance/compliance.py` |
| Full lifecycle integration | `tests/test_e2e_pipeline.py` |
