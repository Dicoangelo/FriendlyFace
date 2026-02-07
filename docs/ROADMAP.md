# FriendlyFace — Roadmap

## Phase Overview

| Phase | Name | Status | Tests | Key Deliverables |
|-------|------|--------|-------|-----------------|
| Phase 1 | Full 6-Layer Architecture | ✅ Complete | 560 | All 6 layers, 46 endpoints, forensic pipeline |
| Phase 2 | Advanced Features | ✅ Complete | ~725 | Real ZK, real DID/VC, DP-FL, SDD, voice, multi-modal |
| Phase 3 | Cryptographic Hardening & Ops | ✅ Complete | ~845 | Ed25519 DIDs, Schnorr ZK, SSE streaming, API endpoints |
| Phase 4 | Frontend Dashboard | ✅ Complete | ~900 | React 19 dashboard, 9 pages, consent UI, real-time SSE |
| Phase 5 | Config & Logging | ✅ Complete | ~950 | Pydantic settings, structured JSON logging, rate limiting |
| Phase 6 | Auth & API Hardening | ✅ Complete | ~1000 | Auth providers, API versioning, OSCAL export |
| Phase 7 | Backup & Migrations | ✅ Complete | ~1027 | SQLite backup, migration framework, erasure, auto-audit |
| Phase 8 | Production Readiness | ✅ Complete | 1070+ | Auth wiring, OIDC fix, RBAC, retention, rollback, CI/CD |

---

## Phase 1: Full 6-Layer Architecture ✅

**Epic:** `friendlyface-fae` (19 user stories, all closed)
**Focus:** Implement the complete Mohammed ICDF2C 2024 schema with all 6 layers functional.

### Completed User Stories

| ID | Story | Layer |
|----|-------|-------|
| US-001 | Ruff linting configuration | Infra |
| US-002 | PCA training pipeline | Layer 1 |
| US-003 | SVM classifier training | Layer 1 |
| US-004 | Face inference pipeline | Layer 1 |
| US-005 | Recognition API endpoints | Layer 1 |
| US-006 | FedAvg FL simulation | Layer 2 |
| US-007 | Data poisoning detection | Layer 2 |
| US-008 | FL API endpoints | Layer 2 |
| US-009 | Enhanced forensic bundles (multi-layer) | Layer 3 |
| US-010 | Bias audit engine | Layer 4 |
| US-011 | Fairness API endpoints | Layer 4 |
| US-012 | LIME explainability | Layer 5 |
| US-013 | SHAP explainability | Layer 5 |
| US-014 | Explainability API endpoints | Layer 5 |
| US-015 | Consent management system | Layer 6 |
| US-016 | Governance and compliance reporting | Layer 6 |
| US-017 | Consent and governance API endpoints | Layer 6 |
| US-018 | CI/CD pipeline (GitHub Actions) | Infra |
| US-019 | End-to-end forensic pipeline test | Testing |

### Phase 1 Achievements
- 46 FastAPI endpoints across 8 endpoint groups
- Hash-chained events, Merkle tree, provenance DAG fully operational
- 560 passing tests across 25 test files
- Full e2e test exercising consent → train → recognize → explain → audit → bundle → verify
- SQLite + Supabase dual storage backends
- CI/CD with GitHub Actions

---

## Phase 2: Advanced Features ✅

**Epic:** `friendlyface-jnb` (7 user stories, all closed)
**Focus:** Replace stubs with real implementations and add advanced capabilities.

### Completed User Stories

| ID | Story | What Changed |
|----|-------|-------------|
| US-020 | Real ZK proofs (BioZero) | Pedersen-SHA256 commitment scheme replacing stubs |
| US-021 | Real DID/VC (TBFL) | DID:key identifiers + W3C Verifiable Credentials |
| US-022 | Real Flower FL | FL engine enhancements |
| US-023 | Differential privacy FL (FedFDP) | DP-FL with epsilon/delta tracking + fairness-aware clipping |
| US-024 | SDD saliency maps | Pixel-level explainability per arXiv:2505.03837 |
| US-025 | Multi-modal biometrics | Voice biometrics (MFCC) + face+voice fusion |
| US-026 | Production hardening | Various stability improvements |

### Phase 2 Achievements
- Voice biometrics with numpy-only MFCC extraction (no external audio libraries)
- Multi-modal fusion (face + voice) with configurable weighting
- Three explanation methods (LIME, SHAP, SDD) with comparison endpoint
- Differential privacy FL with privacy budget tracking
- ~725 tests passing

---

## Phase 3: Cryptographic Hardening & Operational Maturity ✅

**Epic:** `friendlyface-5xa` (closed)
**Focus:** Replace Phase 2 crypto with production-grade implementations and add operational features.

### User Stories

| ID | Story | Status |
|----|-------|--------|
| US-027 | Ed25519 DID:key with PyNaCl | ✅ Closed |
| US-028 | Verifiable Credentials with Ed25519 signing | ✅ Closed |
| US-029 | Schnorr ZK proofs (numpy-only) | ✅ Closed |
| US-030 | Wire DID + ZK into ForensicBundle lifecycle | ✅ Closed |
| US-031 | DID/VC API endpoints | ✅ Closed |
| US-032 | ZK proof API endpoints | ✅ Closed |
| US-033+ | SSE streaming, JSON-LD export | ✅ Closed |

### Phase 3 Achievements
- Real Ed25519 keypairs via PyNaCl (replacing HMAC stubs)
- DID:key format: `did:key:z6Mk<base58-ed25519-pubkey>`
- Schnorr non-interactive ZK proofs with Fiat-Shamir heuristic
- Automatic DID credential + ZK proof generation in every bundle
- Bundle verification: ZK proof + DID credential + hash chain + Merkle + provenance
- Platform DID with deterministic seed support (`FF_DID_SEED`)
- SSE event streaming endpoint (`GET /events/stream`)

---

## Phases 4–7: Dashboard, Config, Auth, Backup ✅

Phases 4 through 7 delivered the React 19 frontend dashboard, Pydantic configuration, structured logging, pluggable auth providers, API versioning, OSCAL compliance export, SQLite backup/restore, SQL migration framework, GDPR erasure support, and auto-audit triggers. ~1027 tests across 40+ test files.

---

## Phase 8: Production Readiness & Operational Maturity ✅

**Focus:** Wire isolated modules together, fix security gaps, add operational capabilities.

### User Stories

| ID | Story | Status |
|----|-------|--------|
| US-076 | Wire auth provider factory into FastAPI app | ✅ Closed |
| US-077 | Fix OIDC signature verification (JWKS) | ✅ Closed |
| US-078 | Backup retention policy | ✅ Closed |
| US-079 | Migration rollback support | ✅ Closed |
| US-080 | RBAC on sensitive endpoints | ✅ Closed |
| US-081 | CI coverage + security scanning | ✅ Closed |
| US-082 | Operational runbook | ✅ Closed |
| US-083 | Update documentation | ✅ Closed |

### Phase 8 Achievements
- Auth provider factory wired into FastAPI global dependency (Bearer, X-API-Key, query param)
- OIDC provider now verifies JWT signatures via JWKS (RS256/ES256) with 1-hour cache
- Backup retention policy with count and age-based limits, auto-cleanup after each backup
- Migration rollback via `_down.sql` companion files for all migrations 002-007
- Role-based access control on admin endpoints (backup, restore, cleanup, rollback, compliance)
- CI pipeline: pytest-cov (fail_under=80%), bandit security scanning, coverage artifact upload
- Operational runbook covering startup, monitoring, backup, migration, auth, troubleshooting
- 1070+ tests across 40+ test files

---

## Issue Tracking

All work is tracked via `bd` (beads). Key commands:

```bash
bd ready              # Available work
bd show <id>          # Issue details
bd update <id> --status in_progress
bd close <id>
bd sync               # Sync with git
```

### Epic IDs

| Epic | Phase | Status |
|------|-------|--------|
| `friendlyface-fae` | Phase 1: Full 6-Layer Architecture | ✅ Closed |
| `friendlyface-jnb` | Phase 2: Advanced Features | ✅ Closed |
| `friendlyface-5xa` | Phase 3: Crypto Hardening & Ops | ✅ Closed |

---

## Contributing

See [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) for setup instructions and development workflow. Key points:

1. Pick an issue from `bd ready`
2. Create a branch, implement, test
3. Run quality gates: `pytest tests/ -v && ruff check friendlyface/ tests/`
4. Push and open a PR
5. CI must pass before merge
