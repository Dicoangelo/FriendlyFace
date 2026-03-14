# PRD: FriendlyFace ForensicSeal — Verifiable AI Compliance Certificates

## Overview

ForensicSeal is the centerpiece product of FriendlyFace: a cryptographically signed, publicly verifiable AI compliance certificate backed by real forensic evidence. It collapses all 6 layers of the platform (recognition, federated learning, forensics, fairness, explainability, governance) into a single W3C Verifiable Credential — the artifact a CISO shows their board, a regulator verifies, and a court trusts.

**The analogy:** SSL certificates made web security verifiable. ForensicSeal makes AI compliance verifiable.

**Market trigger:** EU AI Act high-risk enforcement begins August 2, 2026. Every company deploying facial recognition in Europe needs provable compliance or faces penalties up to €35M / 7% global turnover. No system currently issues verifiable AI compliance certificates.

**Target customers:** Enterprises deploying facial recognition (airports, banks, border control), FR vendors needing to certify their products, and compliance consultancies advising on AI Act readiness.

## Goals

- Ship ForensicSeal as a one-endpoint product: `POST /seal/issue` → returns a verifiable compliance certificate
- Make any ForensicSeal independently verifiable by anyone via `POST /seal/verify` (no account needed, no API key)
- Support continuous compliance: seals expire, must be re-issued, building a compliance history
- Create the compliance proxy that wraps ANY facial recognition API with forensic logging
- Produce EU AI Act Annex IV conformity documentation automatically
- Reach first paying customer before August 2, 2026

## Quality Gates

These commands must pass for every user story:
- `pytest tests/ -v` — All tests pass
- `ruff check friendlyface/ tests/` — Linting

For UI stories, also include:
- `cd frontend && npm run build && npm run lint` — Frontend builds clean

## User Stories

### US-001: ForensicSeal Issuance Endpoint
**Description:** As a compliance officer, I want to request a ForensicSeal for my AI system so that I have a cryptographically verifiable proof of compliance.

**Acceptance Criteria:**
- [ ] `POST /seal/issue` accepts `{system_id, system_name, assessment_scope, bundle_ids[]}`
- [ ] Endpoint runs compliance checks across all 6 layers: consent coverage, bias audit recency, explanation coverage, hash chain integrity, bundle verification, provenance completeness
- [ ] Each check produces a pass/fail with score (0-100) and evidence references
- [ ] If all checks pass threshold (configurable, default 80%), issues a ForensicSeal
- [ ] ForensicSeal is a W3C Verifiable Credential signed with the platform's Ed25519 DID:key
- [ ] Credential contains: issuer DID, subject system ID, issuance timestamp, expiry (default 90 days), compliance score, ZK proof of underlying evidence, Merkle root at time of issuance
- [ ] Seal is persisted to DB with unique `seal_id`
- [ ] Returns JSON with `seal_id`, `credential`, `verification_url`, `expires_at`, `compliance_summary`
- [ ] If checks fail, returns 422 with detailed failure report and remediation steps

### US-002: ForensicSeal Public Verification
**Description:** As a regulator or auditor, I want to verify a ForensicSeal without needing an account so that I can independently confirm an AI system's compliance status.

**Acceptance Criteria:**
- [ ] `POST /seal/verify` accepts `{credential}` (the full VC JSON)
- [ ] `GET /seal/verify/{seal_id}` verifies by seal ID (public, no auth)
- [ ] Verification checks: DID signature valid, credential not expired, credential not revoked, ZK proof valid, Merkle root matches chain state at issuance time
- [ ] Returns `{valid: bool, checks: {signature, expiry, revocation, zk_proof, merkle}, issuer, issued_at, expires_at, compliance_score}`
- [ ] No API key required — this endpoint is fully public
- [ ] Verification works offline with just the credential JSON (DID + ZK proof are self-contained)

### US-003: ForensicSeal Dashboard Page
**Description:** As a platform user, I want a dashboard showing all issued seals, their status, and expiry so that I can manage my compliance posture.

**Acceptance Criteria:**
- [ ] New `/seals` page in React frontend
- [ ] Shows table of all issued seals: seal_id, system_name, issued_at, expires_at, status (active/expired/revoked), compliance_score
- [ ] Status badges: green (active, >30 days), yellow (expiring <30 days), red (expired/revoked)
- [ ] Click seal to see full compliance breakdown (per-layer scores, evidence links)
- [ ] "Issue New Seal" button triggers seal issuance flow
- [ ] "Verify External Seal" input field for pasting credentials
- [ ] Seal history chart showing compliance score over time

### US-004: Compliance Proxy Middleware
**Description:** As an enterprise running NEC/Cognitec/Amazon Rekognition, I want to route my API calls through FriendlyFace so that every recognition request is automatically forensically logged without changing my integration.

**Acceptance Criteria:**
- [ ] New endpoint `POST /proxy/recognize` accepts `{upstream_url, upstream_headers, image, metadata}`
- [ ] Proxy forwards the image to the upstream recognition API
- [ ] Before forwarding: checks consent for subject (if subject_id provided), logs `inference_request` forensic event
- [ ] After response: logs `inference_result` forensic event with upstream response, input hash, latency
- [ ] Accumulated results feed into bias audits and compliance scoring
- [ ] Configuration via `FF_PROXY_UPSTREAM_URL` and `FF_PROXY_UPSTREAM_KEY` env vars for default upstream
- [ ] Proxy adds <5ms overhead (excluding upstream latency)
- [ ] Works with any REST-based recognition API (vendor-agnostic)

### US-005: EU AI Act Annex IV Conformity Document Generator
**Description:** As a compliance officer, I want to generate a full Annex IV technical documentation package so that I can demonstrate conformity before the August 2, 2026 deadline.

**Acceptance Criteria:**
- [ ] `POST /governance/conformity-assessment` generates a structured document
- [ ] Document covers all Annex IV requirements: system description, intended purpose, risk classification, training data provenance, bias testing records, performance metrics, human oversight measures, cybersecurity measures, quality management procedures
- [ ] Each section auto-populated from FriendlyFace data: provenance DAG, bias audits, forensic bundles, consent records, compliance reports
- [ ] Output formats: JSON (structured), HTML (human-readable), PDF-ready Markdown
- [ ] Gaps identified and flagged: "Section 4.2: No bias audit found for ethnicity — action required"
- [ ] Document references specific ForensicSeal IDs as evidence

### US-006: Seal Expiry and Continuous Compliance
**Description:** As a compliance officer, I want seals to expire and require re-issuance so that compliance is continuous, not point-in-time.

**Acceptance Criteria:**
- [ ] Seals have configurable expiry (default 90 days, min 7, max 365)
- [ ] `GET /seal/status/{seal_id}` returns current status including days until expiry
- [ ] Expired seals automatically marked as `expired` (background check or on-access)
- [ ] `POST /seal/renew/{seal_id}` re-runs compliance checks and issues a new seal if passing
- [ ] Renewal links to previous seal, creating a compliance chain
- [ ] SSE event emitted when seal is within 30 days of expiry: `seal_expiring`
- [ ] Dashboard shows renewal history and compliance trend

### US-007: Seal Revocation
**Description:** As an admin, I want to revoke a seal if compliance is breached so that the public verification endpoint reflects current reality.

**Acceptance Criteria:**
- [ ] `POST /seal/revoke/{seal_id}` with `{reason}` marks seal as revoked
- [ ] Revocation is itself a forensic event in the hash chain
- [ ] Public verification endpoint returns `valid: false, revocation_reason` for revoked seals
- [ ] Revocation is irreversible
- [ ] Admin role required (RBAC)

### US-008: Pluggable Recognition Engine Interface
**Description:** As a developer, I want to swap the recognition engine without touching the forensic layer so that we can support production-grade models (ArcFace/AdaFace ONNX) alongside the demo PCA+SVM.

**Acceptance Criteria:**
- [ ] Define `RecognitionEngine` abstract base class with `train()`, `predict()`, `get_model_info()` methods
- [ ] `FallbackEngine` wraps existing PCA+SVM pipeline (current behavior)
- [ ] `ONNXEngine` loads ArcFace/AdaFace ONNX models for production inference
- [ ] Engine selection via `FF_RECOGNITION_ENGINE=fallback|onnx|proxy` env var
- [ ] `proxy` engine delegates to the compliance proxy (US-004) for external APIs
- [ ] All engines produce identical forensic events — the forensic layer is engine-agnostic
- [ ] Existing tests pass with both `fallback` and `onnx` engines

### US-009: Blockchain Merkle Root Anchoring
**Description:** As a platform operator, I want to periodically anchor Merkle roots to a public blockchain so that the forensic chain has immutable, externally verifiable timestamps.

**Acceptance Criteria:**
- [ ] New module `friendlyface/crypto/anchor.py` with `BlockchainAnchor` ABC
- [ ] `PolygonAnchor` implementation: publishes Merkle root hash to a Polygon smart contract
- [ ] `BaseAnchor` implementation: publishes to Base L2
- [ ] `NullAnchor` (default): no-op for local/demo deployments
- [ ] Anchoring triggered on configurable interval (default: every 100 events or 24 hours)
- [ ] Anchor transaction hash stored in DB alongside the Merkle root
- [ ] `GET /anchor/history` returns list of anchored roots with tx hashes and block numbers
- [ ] ForensicSeal includes latest anchor tx hash as external proof reference
- [ ] Configuration via `FF_ANCHOR_CHAIN=none|polygon|base` and `FF_ANCHOR_KEY`

### US-010: Python SDK for Enterprise Integration
**Description:** As an enterprise developer, I want a pip-installable SDK so that I can add forensic logging to my existing pipeline in 5 lines of code.

**Acceptance Criteria:**
- [ ] New package `friendlyface-sdk` (separate pyproject.toml in `sdk/` directory)
- [ ] Core class: `FriendlyFaceClient(base_url, api_key)`
- [ ] Methods: `log_event()`, `check_consent()`, `grant_consent()`, `create_bundle()`, `issue_seal()`, `verify_seal()`, `run_audit()`, `proxy_recognize()`
- [ ] Decorator: `@forensic_trace` wraps any function, logs inputs/outputs as forensic events
- [ ] Context manager: `with client.forensic_session() as session:` auto-bundles all events
- [ ] Returns typed dataclasses, not raw dicts
- [ ] Published to PyPI as `friendlyface-sdk`
- [ ] README with quickstart showing 5-line integration
- [ ] 90%+ test coverage on SDK

### US-011: ForensicSeal Landing Page
**Description:** As a potential customer, I want a compelling landing page explaining ForensicSeal so that I understand the value and can request a demo.

**Acceptance Criteria:**
- [ ] New route `/seal` in React frontend serves the landing page
- [ ] Hero: "The compliance certificate your AI system needs before August 2, 2026"
- [ ] Sections: What is ForensicSeal, How it works (3-step visual), Why you need it (EU AI Act), Competitive comparison (vs manual audits), Pricing placeholder, Demo request form
- [ ] Live verification demo: paste any seal credential, see verification result in real-time
- [ ] Responsive, dark theme consistent with existing dashboard
- [ ] SEO meta tags for "EU AI Act compliance", "AI compliance certificate", "forensic AI"

### US-012: ICDF2C 2026 Paper Submission Package
**Description:** As a researcher, I want to finalize the academic paper with real benchmarks from the activated pipeline so that we can submit to ICDF2C 2026.

**Acceptance Criteria:**
- [ ] Run `scripts/full_pipeline_demo.py` with timing instrumentation, collect latency metrics
- [ ] Benchmarks: event recording latency, bundle creation time, ZK proof generation time, seal issuance time, verification time, Merkle proof time
- [ ] Update `docs/PAPER_DRAFT.md` Table 3 (Performance) with real numbers
- [ ] Add ForensicSeal as a contribution in Section 4 (Implementation)
- [ ] Generate comparison table vs 5 commercial systems (with ForensicSeal column)
- [ ] Export paper as LaTeX (.tex) for conference submission
- [ ] Prepare supplementary materials: architecture diagrams, API documentation link, demo URL

## Functional Requirements

- FR-1: ForensicSeal must be a valid W3C Verifiable Credential (JSON-LD format)
- FR-2: Seal verification must work with only the credential JSON — no server round-trip required for cryptographic checks
- FR-3: The compliance proxy must add less than 5ms overhead (excluding upstream API latency)
- FR-4: Seal issuance must check all 6 layers and produce per-layer scores
- FR-5: Annex IV conformity document must cover all 10 required sections from EU AI Act Article 11
- FR-6: Blockchain anchoring must be optional — the system must work fully without it
- FR-7: The SDK must support Python 3.9+ with zero required dependencies beyond `requests`
- FR-8: All new endpoints must follow existing patterns: rate limiting, RBAC, forensic event emission, OpenAPI tags
- FR-9: Seal revocation must be irreversible and cryptographically recorded
- FR-10: Continuous compliance chain must be queryable: "show me this system's compliance history for the last 12 months"

## Non-Goals (Out of Scope)

- Building a production facial recognition model (we wrap existing engines)
- Mobile app or native desktop client
- Multi-tenant SaaS with billing (future phase)
- Real-time video stream processing (batch/request-response only)
- Replacing existing compliance frameworks (SOC 2, ISO 27001) — ForensicSeal complements them
- GDPR Data Subject Access Request (DSAR) automation
- Non-facial-recognition AI systems (future expansion, but not v1)

## Technical Considerations

- ForensicSeal issuance reuses existing `crypto/did.py` (Ed25519), `crypto/vc.py` (Verifiable Credentials), `crypto/schnorr.py` (ZK proofs), and `governance/compliance.py` (scoring)
- The compliance proxy should use `httpx.AsyncClient` for upstream forwarding (already a dependency)
- Blockchain anchoring requires ethers.js or web3.py — keep it in a separate optional dependency group
- The SDK should be a thin HTTP wrapper, not a copy of the server logic
- Seal verification must handle clock skew gracefully (allow 5-minute window)
- All seal operations should emit SSE events for real-time dashboard updates

## Success Metrics

- First ForensicSeal issued against real forensic evidence (not demo data)
- Public verification endpoint returns valid=true for issued seals
- Compliance proxy successfully wraps at least one external recognition API
- Annex IV document covers all 10 required sections with auto-populated data
- SDK installable via pip and functional in 5 lines of code
- Paper submitted to ICDF2C 2026 with real benchmark numbers
- At least one enterprise demo/pilot initiated before August 2, 2026

## Open Questions

- Should ForensicSeal be free to verify but paid to issue? (freemium model)
- What's the right default expiry: 30, 60, or 90 days?
- Should we pursue EU AI Act "notified body" status to make seals legally binding?
- Patent filing timeline — provisional before or after ICDF2C submission?
- Should the SDK also ship as a Docker sidecar for non-Python shops?
