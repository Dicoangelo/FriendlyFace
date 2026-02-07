# FriendlyFace — Project Overview

## What Is FriendlyFace?

FriendlyFace is a **forensic-friendly AI facial recognition platform** that makes every recognition event provable, auditable, and legally defensible. It implements Safiia Mohammed's ICDF2C 2024 forensic-friendly schema with state-of-the-art 2025–2026 research components.

The core principle is simple but powerful: **every operation the system performs — training a model, running an inference, auditing for bias, explaining a decision — produces an immutable forensic event** that is hash-chained, Merkle-verified, and linked into a provenance DAG. The result is a platform where every AI decision can be traced, verified, and presented as courtroom-grade evidence.

---

## Why Does This Exist?

Facial recognition is one of the most consequential AI applications deployed today. It affects criminal justice, border security, access control, and civil liberties. Yet most FR systems are black boxes — you can't prove what model was used, what data it was trained on, whether bias was checked, or whether the subject consented.

FriendlyFace addresses this by building forensic accountability into the architecture itself, not as an afterthought. Specifically:

- **For law enforcement and forensic examiners:** Every recognition result comes with a complete chain of custody — from training data through model selection, inference, and explanation — that can withstand legal scrutiny.
- **For regulators and compliance officers:** EU AI Act readiness is built in. Compliance reports are auto-generated, bias audits are continuous, and consent is append-only (no silent deletions).
- **For researchers:** The platform is a living implementation of multiple SOTA papers, providing a testbed for forensic AI, federated learning, fairness, and explainability research.
- **For data subjects:** Consent is granular, revocable, and forensically tracked. You know exactly what your biometric data was used for and can prove it.

---

## How It Works — The 30-Second Version

```
You upload a face image for recognition
        ↓
Layer 1 (Recognition) runs PCA + SVM inference → ForensicEvent logged
        ↓
Layer 2 (FL) — the model was trained via federated learning with poisoning detection → ForensicEvents logged
        ↓
Layer 3 (Blockchain Forensic) — the event is hash-chained, added to Merkle tree, linked in provenance DAG
        ↓
Layer 4 (Fairness) — automatic bias audit checks demographic parity → ForensicEvent logged
        ↓
Layer 5 (Explainability) — LIME + SHAP + SDD explanations generated → ForensicEvents logged
        ↓
Layer 6 (Governance) — consent verified, EU AI Act compliance checked → ForensicEvents logged
        ↓
All artifacts bundled into a ForensicBundle with ZK proof + DID credential
        ↓
Bundle is self-verifiable: anyone can check integrity without trusting the platform
```

---

## Key Design Principles

### 1. Immutability by Default
Every operation produces a `ForensicEvent` with a SHA-256 hash that chains to the previous event. You can't modify history without breaking the chain. This is verified on every startup and on demand via the `/chain/integrity` endpoint.

### 2. Provenance as a First-Class Citizen
The provenance DAG tracks the full lineage: which dataset was used to train which model, which model produced which inference, which inference was explained by which method. No orphaned artifacts.

### 3. Fairness Is Not Optional
Bias auditing runs automatically after configurable intervals. Demographic parity and equalized odds are computed across protected groups. Results are forensically logged — you can't hide a failed audit.

### 4. Explainability Is Multi-Method
Three explanation methods (LIME, KernelSHAP, SDD saliency) provide corroborating evidence. A compare endpoint lets you see all three side by side for any inference event.

### 5. Consent Is Append-Only
Consent records are never deleted, only appended. Granting, revoking, and checking consent all produce forensic events. If consent is revoked, future inferences are blocked and the block is logged.

### 6. Cryptographic Verification
Forensic bundles include Schnorr ZK proofs (proving bundle integrity without revealing contents) and Ed25519 DID credentials (proving who created the bundle). External auditors can verify without accessing raw data.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Framework** | FastAPI (async) | 46+ REST endpoints |
| **ML Pipeline** | scikit-learn (PCA + SVM) | Face recognition |
| **Explainability** | LIME (library) + KernelSHAP (custom) + SDD | Multi-method XAI |
| **Cryptography** | PyNaCl (Ed25519) + numpy (Schnorr) | DIDs, ZK proofs |
| **Storage** | aiosqlite (dev) / Supabase (prod) | Switchable backends |
| **Audio DSP** | numpy (MFCC extraction) | Voice biometrics |
| **Testing** | pytest + pytest-asyncio | 560+ tests, ~29s |
| **Linting** | ruff | Code quality enforcement |
| **Containerization** | Docker | Reproducible deployments |
| **Language** | Python 3.11+ | Modern async Python |

---

## Who Built This?

FriendlyFace is built by **Dicoangelo** at **Black Amethyst Capital**, implementing the forensic-friendly AI framework developed by **Safiia Mohammed** at the University of Windsor (ICDF2C 2024 conference). The platform integrates research from BioZero, TBFL, FedFDP, SDD, and EU AI Act compliance frameworks.

---

## Quick Links

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Deep dive into the 6-layer system |
| [API_REFERENCE.md](./API_REFERENCE.md) | Complete endpoint documentation |
| [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) | Setup, testing, contributing |
| [SECURITY_AND_CRYPTOGRAPHY.md](./SECURITY_AND_CRYPTOGRAPHY.md) | DID, ZK proofs, hash chains |
| [DATA_MODELS.md](./DATA_MODELS.md) | Pydantic models and DB schema |
| [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) | Docker, Fly.io, Railway |
| [RESEARCH_LINEAGE.md](./RESEARCH_LINEAGE.md) | Academic foundations |
| [ROADMAP.md](./ROADMAP.md) | Phase status and future work |
