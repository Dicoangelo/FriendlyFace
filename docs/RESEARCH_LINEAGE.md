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

## Research Refresh — March 2026

*Last scanned: 2026-03-14. Covers developments since Feb 6, 2026.*

### Mohammed's Latest Work

Mohammed & Ngom published **"Toward Forensic-Friendly AI: Integrating Blockchain with Federated Learning to Enhance AI Trustworthiness"** (Springer, 2025) — this is the ICDF2C 2024 paper now published in the Springer proceedings (Lecture Notes, Vol. 578). This is the paper FriendlyFace implements. Safiia Mohammed remains a PhD candidate at University of Windsor; no additional publications found beyond this one.

**Action:** Update paper reference [1] with the Springer DOI once available. Consider reaching out to Mohammed about FriendlyFace as a reference implementation.

### EU AI Act — Critical August 2026 Deadline

The **August 2, 2026** deadline is now confirmed as the most consequential enforcement date:
- Annex III high-risk AI requirements become enforceable (facial recognition is explicitly high-risk)
- Real-time remote biometric identification in public spaces **banned** (narrow law enforcement exceptions)
- Untargeted facial image scraping from internet/CCTV **prohibited absolutely** (no exceptions)
- Retrospective facial recognition requires judicial authorization, strict necessity, and documentation
- Penalties: up to **€35M or 7% global turnover** for prohibited practices
- Required: quality management systems, risk management frameworks, technical documentation, conformity assessments, EU database registration

**FriendlyFace relevance:** Our compliance reporter (`governance/compliance.py`) already covers consent coverage, bias audit pass rates, explanation coverage, and bundle integrity — directly mapping to the required documentation. The forensic bundle is exactly the "technical documentation" the Act requires. **This is our strongest differentiator — no competitor has automated EU AI Act compliance documentation.**

**Action:** Update compliance reporter to explicitly reference August 2026 deadline. Add Annex III high-risk classification checks. Consider adding conformity assessment generation.

### New Papers to Integrate

#### CryptoFair-FL (arXiv:2601.12447) — Verifiable Fairness in FL
- Combines homomorphic encryption + secure MPC + differential privacy for **cryptographically verifiable fairness guarantees** in federated learning
- **Relevance:** Extends FedFDP's fairness-aware DP with cryptographic verification — could enhance our FL layer with verifiable fairness proofs
- **Priority:** Medium — would strengthen the patent's Claim 14 (federated fairness with forensic provenance)

#### Privacy-Fairness-Accuracy Trade-offs (arXiv:2503.16233) — Empirical Analysis
- First unified large-scale empirical evaluation of DP, HE, and SMPC under fairness-aware optimizers
- Reveals inverse relationship: stricter privacy fundamentally limits bias detection
- **Relevance:** Provides empirical backing for our design choice of configurable epsilon/delta in DP-FL
- **Priority:** Low — cite in paper, no code changes needed

#### Deepfake Mitigation with Blockchain (Nature Scientific Reports, 2026)
- "An integrated framework for proactive deepfake mitigation via attention-driven watermarking and blockchain-based authenticity verification"
- **Relevance:** Adjacent — our forensic bundles could integrate deepfake detection scores as additional evidence
- **Priority:** Low — future work section in paper

#### GAN-Blockchain Privacy-Enhanced Recognition (ScienceDirect, 2024)
- SHA-256 blockchain ledgers for preserving GAN-generated synthetic face integrity
- **Relevance:** Validates our SHA-256 hash chain approach for biometric data integrity
- **Priority:** Low — cite as supporting evidence

### Competitor Landscape — No Forensic Bundles

Scanned top 20 facial recognition systems (2025-2026):
- **Clearview AI** — 50B+ images, law enforcement focus, **no forensic provenance, no audit trail**
- **NEC NeoFace Reveal** — forensic face matching (1:N), **no hash chain, no explainability**
- **Cognitec FaceVACS** — enterprise-grade, used for border control, **no consent tracking**
- **Amazon Rekognition** — cloud API, **no provenance DAG, no bundle export**
- **Innefu Argus** — forensic platform with metadata linking, **no blockchain verification**

**No competitor combines:** forensic events + Merkle proofs + ZK verification + DID credentials + provenance DAG + bias audit + explainability + consent + compliance reporting into a unified, self-verifiable bundle.

**FriendlyFace remains the only complete implementation of a forensic-friendly AI architecture.**

### Market Context

- Facial recognition market: **$8.1B (2025) → $36.2B (2035)**, 16.1% CAGR
- Accuracy: 98-99% across demographics for leading algorithms
- NIST recommends immutable logging and automated provenance in forensic-readiness guidelines
- Dubai Police piloting blockchain for investigative file documentation (World Police Summit)

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
