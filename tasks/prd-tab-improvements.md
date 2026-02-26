[PRD]
# PRD: FriendlyFace — Tab-by-Tab UI Improvements

## Overview

Systematically improve all 17 existing frontend pages in the FriendlyFace forensic AI platform, one tab at a time. Each improvement targets: consistent loading skeletons, proper empty states, error handling, real data wiring, interactive polish, and visual refinement. The stack is React 19 + Vite + TailwindCSS (custom tokens: `glass-card`, `fg-muted`, `fg-secondary`, `surface`, `teal`, `cyan`, `amethyst`, `gold`, `rose-ember`). Backend is FastAPI with 84 endpoints at `/api/v1/`.

**Key constraint:** Dashboard is already polished — use it as the gold standard for all other pages.

---

## Goals

- Every tab has a skeleton loading state (not a blank screen or spinner-only)
- Every tab has a proper empty state (EmptyState component) when data is absent
- Every tab has consistent error handling with the rose-ember error banner pattern
- Replace raw JSON textareas with structured form inputs where applicable
- Add loading indicators to all async action buttons (LoadingButton component)
- Add auto-refresh or manual refresh where live data is shown
- Ensure every tab visually matches the Dashboard's glass-card design language

---

## Quality Gates

These commands must pass for every user story:

```bash
cd ~/projects/products/friendlyface/frontend && npm run build
cd ~/projects/products/friendlyface/frontend && npm run lint
```

For UI stories, also include:
- Screenshot verification: start dev server (`source .venv/bin/activate && python3 -m friendlyface`) and take Chrome DevTools screenshot of the tab

---

## User Stories

### US-001: EventStream — Live forensic event feed
**Description:** As a forensic analyst, I want to see a polished live event stream so that I can monitor activity in real time without staring at raw JSON.

**Acceptance Criteria:**
- [ ] Shows skeleton (3 pulsing rows) while SSE connection is establishing
- [ ] Empty state shown when no events have arrived yet (EmptyState with subtitle "Waiting for forensic events…")
- [ ] Each event row shows: event type badge (eventBadgeColor), actor, timestamp, event_id (truncated)
- [ ] Connection status indicator: green dot "Connected" / red dot "Disconnected" in header area
- [ ] Auto-scrolls to latest event, with a "Pause scroll" toggle button
- [ ] Error banner (rose-ember) if SSE connection fails with retry button
- [ ] File: `frontend/src/pages/EventStream.tsx`

### US-002: EventsTable — Paginated forensic event table
**Description:** As a forensic analyst, I want a paginated, filterable table of all events so that I can audit the chain history efficiently.

**Acceptance Criteria:**
- [ ] Skeleton table (5 rows) shown during initial fetch
- [ ] Empty state when no events exist
- [ ] Pagination controls (Prev / Next + "Page X of Y") wired to limit/offset API params
- [ ] Filter by event_type via dropdown (populated from unique types in data)
- [ ] Each row: event_type badge, actor, timestamp, event_id copy button
- [ ] Row click expands inline detail (content JSON in a pre block)
- [ ] Error banner if fetch fails
- [ ] File: `frontend/src/pages/EventsTable.tsx`

### US-003: Bundles — Forensic bundle list + creation
**Description:** As a forensic analyst, I want to view, create, and inspect bundles so that I can package and export forensic evidence sets.

**Acceptance Criteria:**
- [ ] Skeleton (3 pulsing cards) during initial load
- [ ] Empty state when no bundles exist with CTA "Create your first bundle"
- [ ] Bundle list shows: id (truncated+copy), event count, created_at, VC/DID badges if present
- [ ] "Create Bundle" button opens inline form: event_ids textarea, optional description
- [ ] LoadingButton on create action, success banner on completion with bundle ID
- [ ] Bundle row "View" expands: full JSON-LD export preview in scrollable pre block
- [ ] Download button exports bundle as JSON file
- [ ] Error banner on any failure
- [ ] File: `frontend/src/pages/Bundles.tsx`

### US-004: DID Management — Decentralized Identity registry
**Description:** As a system operator, I want to create and inspect DIDs so that I can manage cryptographic identities for the forensic chain.

**Acceptance Criteria:**
- [ ] Skeleton during load of DID list
- [ ] Empty state if no DIDs created yet
- [ ] DID list shows: did string (truncated+copy), created_at, public_key hint
- [ ] "Create DID" button → LoadingButton → shows new DID with full key material in a reveal card
- [ ] "Resolve" input: enter a DID, fetch and display resolution document in pre block
- [ ] Verify VC section: paste VC JSON, verify button, pass/fail result badge
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/DIDManagement.tsx`

### US-005: ZK Proofs — Zero-knowledge proof generation + verification
**Description:** As a forensic analyst, I want a clean interface for generating and verifying ZK proofs so that I can prove bundle integrity without exposing data.

**Acceptance Criteria:**
- [ ] Replace raw JSON textarea for proof verification with structured input: paste proof JSON in a code-editor-style pre+contenteditable or a simple textarea with monospace font and line numbers hint
- [ ] "Generate Proof" section: bundle ID input + LoadingButton, result shows proof fields in a structured breakdown (not raw JSON dump)
- [ ] Proof display shows: scheme, commitment, challenge, response as labelled rows
- [ ] Copy-to-clipboard button on generated proof
- [ ] "Verify Proof" section pre-filled with last generated proof, verify button, clear pass/fail badge (teal checkmark / rose-ember X)
- [ ] "Get Stored Proof" section: bundle ID lookup with skeleton while loading
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/ZKProofs.tsx`

### US-006: Merkle Explorer — Tree root + inclusion proofs
**Description:** As a forensic analyst, I want to visually explore the Merkle tree so that I can verify event inclusion with confidence.

**Acceptance Criteria:**
- [ ] Root card shows: merkle_root (truncated hash + copy), leaf_count, last updated timestamp
- [ ] Skeleton for root card during fetch
- [ ] Empty state if leaf_count is 0 with explanation "No events in the Merkle tree yet"
- [ ] Proof lookup: event ID input → lookup button (LoadingButton) → result card
- [ ] Proof result shows: leaf_hash, root match indicator, path length, leaf_index visually as "Leaf 3 of 47"
- [ ] Proof path rendered as vertical chain: each node shows hash (truncated) + direction badge (left/right)
- [ ] "Refresh Root" button re-fetches root
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/MerkleExplorer.tsx`

### US-007: Bias Audits — Fairness audit interface
**Description:** As a fairness analyst, I want a structured form to run bias audits so that I don't have to hand-craft raw JSON to test demographic groups.

**Acceptance Criteria:**
- [ ] Fairness status card at top: score ring (ProgressRing component), total/compliant audit counts, skeleton during load
- [ ] Replace raw JSON textarea for groups with a dynamic group builder: add/remove group rows with fields (group_name, true_positives, false_positives, true_negatives, false_negatives)
- [ ] Threshold sliders for demographic_parity and equalized_odds (0–0.5 range, 0.01 step)
- [ ] "Run Audit" → LoadingButton → result card shows: compliant badge, fairness_score ring, DP gap, EO gap
- [ ] Audit history table: audit_id, timestamp, compliant badge, fairness_score bar — row click fetches and shows detail in expandable section
- [ ] Demographics breakdown table (if data available)
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/BiasAudits.tsx`

### US-008: FL Simulations — Federated learning execution
**Description:** As an ML engineer, I want a clean simulation launcher so that I can run and compare standard vs DP-FedAvg experiments with clear visual results.

**Acceptance Criteria:**
- [ ] Two-tab layout (Standard / DP-FedAvg) preserved, tab switching is pill style
- [ ] Standard form: n_clients (number input, 2–20), n_rounds (1–10), enable_poisoning_detection toggle, seed input
- [ ] DP form: all standard params + epsilon slider (0.1–10), delta input, max_grad_norm input
- [ ] "Run Simulation" → LoadingButton with "Running…" state
- [ ] Results: timeline of rounds — each round as a row: round number, accuracy progress bar, model hash (truncated), poisoning flags if present
- [ ] Summary card: simulation_id, total rounds, final accuracy, DP budget if applicable
- [ ] FL Status card: fetched from `/api/v1/fl/status` with skeleton
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/FLSimulations.tsx`

### US-009: Explainability — LIME, SHAP, SDD saliency
**Description:** As an ML engineer, I want a structured explainability interface so that I can generate and compare LIME, SHAP, and SDD explanations for model decisions.

**Acceptance Criteria:**
- [ ] Three-section layout: LIME / SHAP / SDD — each collapsible or tab-switched
- [ ] Each section has a consistent form pattern: model_id input, subject_id or image input, generate button (LoadingButton)
- [ ] LIME results: feature importance as horizontal bar chart (inline CSS bars, no chart library needed)
- [ ] SHAP results: base value + feature contribution rows with +/- color coding (teal positive, rose-ember negative)
- [ ] SDD results: gradient magnitude display, class label, confidence score
- [ ] Skeleton placeholders in each result area while loading
- [ ] Copy-to-clipboard for raw result JSON in each section
- [ ] Error banner per section (not global — so one can fail without hiding others)
- [ ] File: `frontend/src/pages/Explainability.tsx`

### US-010: Compliance — EU AI Act compliance report
**Description:** As a compliance officer, I want a visual compliance dashboard so that I can quickly see pass/fail status per requirement and export reports.

**Acceptance Criteria:**
- [ ] Overall compliance ring (ProgressRing) with score % and compliant/non-compliant badge — skeleton during load
- [ ] Requirements broken into sections (data governance, transparency, human oversight, robustness) — each as a collapsible card
- [ ] Per-requirement row: name, status badge (pass/fail/partial), score bar, details text
- [ ] "Re-check Compliance" refresh button with LoadingButton
- [ ] OSCAL export button: calls `/api/v1/governance/oscal` and downloads JSON
- [ ] EU AI Act summary section: risk level badge, applicable articles list
- [ ] Empty state if no compliance data yet
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/Compliance.tsx`

### US-011: Consent Management — Subject consent registry
**Description:** As a data controller, I want to manage consent records so that I can grant, revoke, and audit subject consent in compliance with GDPR.

**Acceptance Criteria:**
- [ ] Consent list with skeleton during load, empty state if none
- [ ] Each consent row: subject_id, purpose, status badge (granted/revoked/expired), granted_at, expiry
- [ ] "Grant Consent" form: subject_id input, purpose input, expiry date picker, optional metadata — LoadingButton
- [ ] Revoke action per row with confirmation dialog
- [ ] Consent event audit trail: expandable list of all consent events for a subject (fetch on demand)
- [ ] Filter by status (all / granted / revoked)
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/ConsentManagement.tsx`

### US-012: Data Erasure — Right-to-erasure requests
**Description:** As a data controller, I want a structured erasure request interface so that I can comply with right-to-erasure requests with full forensic traceability.

**Acceptance Criteria:**
- [ ] Erasure request form: subject_id input, reason textarea, checkbox "I confirm this erasure is legally required" — submit disabled until checkbox checked
- [ ] "Submit Erasure Request" → LoadingButton → result shows: request_id, subject_id, erasure_event_id, timestamp in a success card
- [ ] Erasure history list (from `/api/v1/governance/erasure/requests`): skeleton, empty state, rows with subject_id + timestamp + event_id
- [ ] Warning banner explaining the forensic chain records the erasure event (data removed but event logged)
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/DataErasure.tsx`

### US-013: Retention Policies — Automated data lifecycle
**Description:** As a system operator, I want to configure and run retention policies so that I can enforce data lifecycle rules without manual cleanup.

**Acceptance Criteria:**
- [ ] Active policy card: shows current policy config (max_age_days, max_events, action) — skeleton during load
- [ ] Edit policy form: inline edit with max_age_days number input, max_events number input, action select (archive/delete/flag)
- [ ] "Apply Policy Now" → LoadingButton → result shows: events_processed, events_retained, events_expired counts in a summary card
- [ ] Policy history: last N runs with timestamp + counts as compact rows
- [ ] Dry-run toggle: apply policy in preview mode without executing
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/RetentionPolicies.tsx`

### US-014: Provenance Explorer — DAG lineage browser
**Description:** As a forensic analyst, I want to explore the provenance graph so that I can trace any model or inference back to its training data.

**Acceptance Criteria:**
- [ ] Stats header: total_nodes count, total_edges count — skeleton during load
- [ ] Node search: event_id or node_id input → fetch node detail → display in card
- [ ] Node card shows: node_id, node_type badge (training/inference/explanation/bundle), timestamp, parent IDs as links
- [ ] Lineage path: "trace from root" button — fetches path and renders as vertical chain of node cards
- [ ] Recent nodes list: last 10 nodes as compact rows (type badge + id + timestamp)
- [ ] Empty state if no provenance nodes
- [ ] Error banner on failures
- [ ] File: `frontend/src/pages/ProvenanceExplorer.tsx`

### US-015: Recognition — Face recognition model management
**Description:** As an ML engineer, I want a complete recognition interface so that I can train models, run inference, and manage the face gallery with full forensic traceability.

**Acceptance Criteria:**
- [ ] Tab layout: Gallery / Train / Recognize / Models (existing structure preserved, pill tabs)
- [ ] Gallery tab: subject grid with skeleton, empty state, each subject card shows subject_id + image_count, delete subject button with confirmation
- [ ] Train tab: class JSON builder (dynamic add/remove rows), epochs/components inputs, LoadingButton, result shows model_id + accuracy + event_id
- [ ] Recognize tab: subject_id + model_id inputs, optional confidence threshold slider, LoadingButton, result shows top-N predictions as ranked list with confidence bars
- [ ] Models tab: list with skeleton, each model row shows model_id, type, accuracy bar, trained_at, delete button
- [ ] Voice biometrics section (if endpoint available): enroll/verify with audio file upload input
- [ ] Error banners per-tab (not global)
- [ ] File: `frontend/src/pages/Recognition.tsx`

### US-016: Admin Ops — Backup, migrations, system ops
**Description:** As a system operator, I want a clean admin panel so that I can manage backups, run migrations, and perform system operations safely.

**Acceptance Criteria:**
- [ ] Three sections: Backups / Migrations / System — each as distinct card
- [ ] Backups: list with skeleton, each backup row shows filename + size + created_at + restore button (with confirmation dialog); "Create Backup" LoadingButton
- [ ] Migrations: current version display, available migrations list, "Run Migrations" LoadingButton with confirmation, rollback button per migration
- [ ] System: storage backend indicator, DB stats (event count, bundle count), "Rebuild Merkle Tree" LoadingButton, chain integrity re-check button
- [ ] All destructive actions (restore, rollback) require confirmation dialog
- [ ] Success/error banners per section
- [ ] File: `frontend/src/pages/AdminOps.tsx`

### US-017: Dashboard — Polish pass
**Description:** As a user, I want a refined dashboard so that the entry point feels production-grade.

**Acceptance Criteria:**
- [ ] Add manual "Refresh" button next to auto-refresh interval indicator (shows "Last updated X seconds ago")
- [ ] Format uptime as "Xh Ym Zs" when uptime > 3600s
- [ ] Chain integrity card shows last verified timestamp (not just valid/invalid)
- [ ] Events by type bar chart: truncate long event type names with title tooltip
- [ ] Recent events table: add a "View all" link to /events route
- [ ] Compliance card: show specific failing requirements count if non-compliant
- [ ] File: `frontend/src/pages/Dashboard.tsx`

---

## Functional Requirements

- FR-1: All pages must use the existing EmptyState component (`components/EmptyState.tsx`) for zero-data states
- FR-2: All async button actions must use LoadingButton component (`components/LoadingButton.tsx`) — no manual disabled/spinner patterns
- FR-3: Skeleton loading must use existing Skeleton components where available, fallback to `animate-pulse bg-surface rounded` divs
- FR-4: All error states must use the standard rose-ember banner: `bg-rose-ember/10 border border-rose-ember/20 rounded-lg px-4 py-2 text-rose-ember text-sm`
- FR-5: All card containers must use `glass-card p-4` className (existing pattern)
- FR-6: No new dependencies — use only existing: React 19, TailwindCSS, existing components
- FR-7: Each story must leave the build passing (`npm run build` exits 0)
- FR-8: Raw JSON textareas for structured user input must be replaced with proper form fields

## Non-Goals

- Backend API changes (frontend only, unless a critical endpoint is missing)
- New pages or routes
- Chart library integration (use CSS-only bars as Dashboard does)
- Authentication changes
- Mobile-first redesign (responsive is fine, mobile-first is not in scope)
- Dark/light theme toggle (existing theme is dark, stay as-is)

## Technical Considerations

- Component inventory: `EmptyState`, `LoadingButton`, `ProgressRing`, `Skeleton*`, `ErrorBoundary`, `Toast`, `ConfirmDialog` — use these, don't recreate
- API base: all endpoints at `/api/v1/` (e.g. `/api/v1/fairness/audits`)
- Pagination pattern: `?limit=20&offset=0` → response `{items, total, limit, offset}`
- Auth: dev mode has no auth; production uses `X-API-Key` header
- Color tokens in TailwindCSS: `teal`, `cyan`, `amethyst`, `gold`, `rose-ember`, `fg-muted`, `fg-secondary`, `fg-faint`, `surface`, `surface-light`, `border-theme`

## Success Metrics

- Every tab passes `npm run build` and `npm run lint`
- Every tab has a visible loading state (no blank flash)
- Every tab has an empty state message
- No raw JSON textareas remain for structured input (BiasAudits groups, ZKProofs verify)
- Visual screenshot of each improved tab matches Dashboard's glass-card quality

## Open Questions

- Should Recognition's voice biometrics section be enabled? (Depends on endpoint availability at `/api/v1/recognition/voice/*`)
- Should RetentionPolicies show a dry-run diff preview or just counts?
[/PRD]
