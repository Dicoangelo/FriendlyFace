# PROVISIONAL PATENT APPLICATION

## United States Patent and Trademark Office

### Application Type: Provisional Patent Application under 35 U.S.C. 111(b)

---

## 1. TITLE OF THE INVENTION

**System and Method for Forensic-Friendly Artificial Intelligence with Unified Cryptographic Provenance and Multi-Layer Verification**

---

## 2. ABSTRACT

A system and method for forensic-friendly artificial intelligence facial recognition that unifies cryptographic provenance, multi-layer verification, and regulatory compliance into a single self-verifiable architecture. The system implements a six-layer architecture comprising a recognition engine, a federated learning engine with differential privacy, a blockchain forensic layer with hash-chained events and Merkle tree verification, a fairness and bias auditing layer, a multi-method explainability layer, and a consent and governance layer. Every operation across all six layers produces an immutable forensic event linked into a SHA-256 hash chain, an append-only Merkle tree, and a directed acyclic provenance graph. The system generates self-verifiable forensic bundles that combine hash-chain integrity proofs, Merkle inclusion proofs, Schnorr zero-knowledge proofs, Ed25519 Decentralized Identifier credentials, bias audit records, explainability artifacts, and provenance lineage into a single cryptographically sealed artifact suitable for courtroom-grade forensic evidence and automated EU AI Act compliance reporting.

---

## 3. FIELD OF THE INVENTION

The present invention relates generally to the field of artificial intelligence and computer vision, and more particularly to systems and methods for facial recognition that produce forensically verifiable, privacy-preserving, bias-audited, and regulatorily compliant outputs. The invention further relates to cryptographic provenance tracking, zero-knowledge proof systems for biometric verification, decentralized identity management for AI systems, federated machine learning with differential privacy guarantees, and automated regulatory compliance assessment for high-risk AI systems under the European Union Artificial Intelligence Act.

---

## 4. BACKGROUND OF THE INVENTION

### 4.1 Current State of AI Facial Recognition

Facial recognition technology has achieved widespread deployment across law enforcement, border security, access control, and consumer applications. Commercial systems such as Clearview AI, NEC NeoFace, Amazon Rekognition, Microsoft Azure Face API, and various open-source implementations (e.g., DeepFace, InsightFace) provide high-accuracy face matching capabilities. These systems typically employ deep convolutional neural networks trained on large-scale face datasets to extract embedding vectors and perform identity matching via nearest-neighbor or classification-based approaches.

### 4.2 Problem Statement

Despite significant advances in recognition accuracy, existing facial recognition systems suffer from a fundamental architectural deficiency: the absence of a unified forensic trail that preserves the complete provenance, integrity, fairness assessment, explainability, and consent state of every recognition operation. This deficiency creates several critical problems:

**Forensic Inadmissibility.** When facial recognition results are introduced as evidence in legal proceedings, opposing counsel may challenge the integrity of the recognition pipeline. Current systems cannot cryptographically prove that the model used for inference is the same model that was trained on a particular dataset, that the training data was not tampered with, or that intermediate computational steps were not modified. The absence of hash-chained event records and cryptographic integrity proofs renders recognition results forensically unverifiable.

**Provenance Opacity.** No existing commercial system tracks the complete lineage from training data through model creation, inference execution, explanation generation, and bundle output in a cryptographically sealed directed acyclic graph. Without provenance tracking, it is impossible to determine post hoc which training data influenced a particular recognition decision, whether the model was updated between training and inference, or whether intermediate artifacts were substituted.

**Fairness Verification Gap.** While bias auditing tools exist as standalone libraries (e.g., AI Fairness 360, Fairlearn), no existing system integrates fairness assessment into the forensic chain such that every recognition decision carries a cryptographically linked bias audit record. Bias audits performed separately from the recognition pipeline cannot prove that they were computed on the same model and data that produced a given recognition result.

**Explainability Disconnect.** Existing explainability methods (LIME, SHAP, GradCAM) operate as post-hoc analysis tools disconnected from the recognition pipeline. No system produces explanation artifacts that are hash-linked to the specific inference event they explain, cryptographically sealed into a verifiable bundle, and traceable through a provenance DAG to the model and training data that produced the result.

**Consent Enforcement Absence.** Current facial recognition systems lack built-in consent management. Consent records, when they exist, are maintained in separate databases with no cryptographic linkage to the recognition operations they authorize. There is no mechanism to prove forensically that consent was active at the time of a specific recognition event.

**Regulatory Non-Compliance.** The European Union Artificial Intelligence Act (Regulation (EU) 2024/1689), which classifies facial recognition as a high-risk AI system, requires documentation of data governance, bias monitoring, human oversight mechanisms, and transparency measures. No existing system generates automated compliance reports that are themselves forensic events linked into the same hash chain as the operations they assess.

### 4.3 Limitations of Prior Art

**Clearview AI** provides a large-scale face matching service that indexes billions of images scraped from the internet. However, Clearview AI maintains no public forensic chain, no provenance DAG, no built-in bias auditing, no integrated explainability, and no consent management system. Its recognition results are opaque artifacts with no cryptographic integrity guarantees.

**NEC NeoFace** is a high-accuracy facial recognition engine used extensively in law enforcement. While NEC maintains internal quality assurance processes, the system does not produce self-verifiable forensic bundles, does not integrate zero-knowledge proofs for privacy-preserving verification, and does not maintain a hash-chained event log of all operations.

**Amazon Rekognition** provides cloud-based facial recognition via API. Recognition results are returned as JSON responses with confidence scores but carry no cryptographic provenance, no Merkle proof of inclusion in an integrity-verified event chain, and no linked bias audit or explainability artifacts.

**Microsoft Azure Face API** offers face detection, verification, and identification services. While Microsoft has published responsible AI guidelines and provides some fairness analysis tools, the Face API itself does not integrate these into a unified forensic chain. Recognition results are stateless API responses with no hash chaining or provenance tracking.

### 4.4 Prior Art in Component Technologies

Several research contributions address individual components of the problem space but none achieve the unified integration disclosed herein:

Mohammed (ICDF2C 2024) proposed a theoretical six-layer forensic-friendly AI framework but did not provide a complete implementation combining all layers with cryptographic verification. The BioZero protocol (arXiv:2409.17509) demonstrated zero-knowledge biometric verification using Merkle trees but did not integrate federated learning, explainability, or regulatory compliance. The TBFL framework (arXiv:2602.02629) proposed trustworthy blockchain-based federated learning with decentralized identifiers and verifiable credentials but did not address forensic event chaining, bias auditing, or explainability. FedFDP (arXiv:2402.16028) advanced fairness-aware federated learning with differential privacy but lacked forensic provenance and consent management. SDD saliency decomposition (arXiv:2505.03837) provided a novel explainability method for face recognition but did not integrate its outputs into a forensic chain. EU AI Act compliance frameworks (arXiv:2512.13907) proposed assessment methodologies but did not embed compliance reporting into a cryptographically verified operational pipeline.

### 4.5 Gap in the Art

No existing system or method combines: (a) hash-chained forensic events with append-only Merkle tree verification; (b) a directed acyclic provenance graph tracking complete data lineage from training through inference; (c) zero-knowledge proof generation over forensic bundles; (d) decentralized identity and verifiable credential issuance for forensic artifacts; (e) consent-gated recognition with forensic logging of consent state; (f) federated learning with differential privacy and poisoning detection, each round forensically logged; (g) multi-method explainability with hash-sealed explanation artifacts; and (h) automated regulatory compliance reporting integrated into the forensic event chain. The present invention fills this gap by providing the first unified architecture that integrates all of these capabilities into a single, self-verifiable forensic platform.

---

## 5. SUMMARY OF THE INVENTION

The present invention provides a system and method for forensic-friendly artificial intelligence facial recognition that produces self-verifiable forensic bundles combining cryptographic integrity proofs, provenance lineage, bias assessments, explainability artifacts, consent records, and regulatory compliance reports into a single sealed artifact.

The system is organized into six interdependent layers, each of which produces immutable forensic events that are linked into a unified hash chain, indexed in an append-only Merkle tree, and tracked through a directed acyclic provenance graph:

**Layer 1 (Recognition Engine)** performs facial recognition using dimensionality reduction (PCA) and classification (SVM), with optional multi-modal fusion incorporating voice biometrics via MFCC feature extraction. Every training operation and inference result is recorded as a hash-chained forensic event.

**Layer 2 (Federated Learning Engine)** enables privacy-preserving model training across distributed data holders using Federated Averaging (FedAvg) with differential privacy (DP-FedAvg). The engine applies per-client gradient clipping, calibrated Gaussian noise injection, cumulative privacy budget tracking, and norm-based data poisoning detection. Each federated learning round and each security alert is recorded as a forensic event with full provenance linkage.

**Layer 3 (Blockchain Forensic Layer)** provides the cryptographic backbone comprising SHA-256 hash-chained events, an append-only Merkle tree with inclusion proofs, a directed acyclic provenance graph, Schnorr zero-knowledge proofs (non-interactive via Fiat-Shamir heuristic), Ed25519 decentralized identifiers (DID:key method), and W3C Verifiable Credentials with Ed25519Signature2020 proofs.

**Layer 4 (Fairness and Bias Auditor)** computes demographic parity and equalized odds metrics across demographic groups, flags compliance violations against configurable thresholds, and records every audit as a forensic event linked to the model and data that produced the audited results.

**Layer 5 (Explainability Layer)** generates three independent explanation types --- LIME (Local Interpretable Model-agnostic Explanations), KernelSHAP (SHapley Additive exPlanations), and SDD (Spatial-Directional Decomposition) saliency maps with canonical facial region scoring --- each producing hash-sealed artifacts linked to the specific inference event explained.

**Layer 6 (Consent and Governance Layer)** manages append-only consent records, enforces consent-gated recognition (blocking inference when consent is missing, revoked, or expired), generates automated EU AI Act compliance reports, and produces OSCAL-format security assessment exports. Every consent state change, every inference block, and every compliance report generation is recorded as a forensic event.

The system produces **self-verifiable forensic bundles** that aggregate hash-chain integrity, Merkle inclusion proofs, provenance DAG paths, Schnorr ZK proofs, DID-signed Verifiable Credentials, layer-specific artifacts, and bias audit records into a single cryptographically sealed artifact. Bundle verification requires no external authority; any party with access to the bundle can independently verify its integrity by recomputing hashes, checking Merkle proofs, verifying Schnorr proofs, and validating Ed25519 credential signatures.

---

## 6. DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENT

### 6.1 System Architecture Overview

The preferred embodiment comprises a server application implemented in Python using the FastAPI web framework, an asynchronous SQLite database for persistence, and a React-based frontend dashboard. The system exposes 84 or more API endpoints organized across 12 functional domains. The architecture follows a layered design wherein each layer operates independently but all layers share a common forensic infrastructure that chains, indexes, and seals every operation.

### 6.2 The Six-Layer Architecture

#### 6.2.1 Layer 1: Recognition Engine

The recognition engine performs facial recognition through the following pipeline:

1. **Image Preprocessing.** Input images are converted to grayscale, resized to a canonical 112x112 pixel grid, and normalized to floating-point values in the range [0, 1].

2. **Dimensionality Reduction.** Principal Component Analysis (PCA) is applied to the flattened 12,544-dimensional image vector to produce a reduced-dimensionality feature vector. The PCA transformation matrix and mean vector are computed during training and stored as model artifacts.

3. **Classification.** A Support Vector Machine (SVM) classifier with a linear or RBF kernel is trained on the PCA-reduced feature vectors. The classifier outputs a predicted identity label and a confidence score derived from the decision function.

4. **Voice Biometrics (Optional).** Mel-Frequency Cepstral Coefficients (MFCC) are extracted from audio recordings for voice-based identity verification. A multi-modal fusion module combines face and voice confidence scores using a configurable weighting scheme.

5. **Forensic Logging.** Every training initiation, training completion, model registration, inference request, and inference result is recorded as a ForensicEvent with event type, actor identity, payload metadata, sequence number, and hash chain linkage.

#### 6.2.2 Layer 2: Federated Learning Engine

The federated learning engine enables privacy-preserving model training:

1. **FedAvg Protocol.** Multiple simulated or real clients perform local training on their private datasets and submit weight updates to a central aggregator. The aggregator computes the element-wise average of client weight updates and applies it to the global model.

2. **Differential Privacy (DP-FedAvg).** For privacy-preserving aggregation:
   - Each client's weight delta (local update minus global weights) is clipped to a configurable maximum L2 norm (default: 1.0).
   - The clipped deltas are averaged across clients.
   - Calibrated Gaussian noise with standard deviation sigma is added to the averaged delta, where sigma is computed using the analytic Gaussian mechanism: sigma = (sensitivity * sqrt(2 * ln(1.25 / delta))) / epsilon, and sensitivity = 1 / n_clients.
   - Cumulative privacy budget is tracked across rounds via simple composition (epsilon_total = sum of per-round epsilons).

3. **Poisoning Detection.** After each round, a norm-based anomaly detector computes the L2 norm of each client's weight delta. Clients whose delta norm exceeds a configurable multiple (default: 3.0x) of the median delta norm are flagged as potentially poisoned. Each flagged client generates a SECURITY_ALERT forensic event.

4. **Forensic Integration.** Each FL round produces a ForensicEvent of type FL_ROUND containing the round number, number of clients, global model hash (SHA-256 of serialized weights), aggregation strategy, DP configuration, noise scale, clipped client list, and cumulative privacy spent. Each round also produces a ProvenanceNode of entity type "dp_fl_round" linked to prior rounds via DERIVED_FROM relations.

#### 6.2.3 Layer 3: Blockchain Forensic Layer

This layer provides the cryptographic infrastructure that underpins all forensic guarantees:

**6.2.3.1 Hash-Chained Forensic Events**

Every operation in the system produces a ForensicEvent object containing:
- A universally unique identifier (UUID v4)
- An event type (one of 13 defined types: training_start, training_complete, model_registered, inference_request, inference_result, explanation_generated, bias_audit, consent_recorded, consent_update, bundle_created, fl_round, security_alert, compliance_report)
- A UTC timestamp
- An actor identifier (the entity that triggered the event)
- A payload dictionary containing event-specific metadata
- A previous_hash field pointing to the event_hash of the immediately preceding event (or the string "GENESIS" for the first event in the chain)
- A sequence_number providing a monotonically increasing integer index
- An event_hash computed as SHA-256 of the canonical JSON serialization of all hashable fields concatenated with the previous_hash

The canonical JSON serialization uses deterministic key ordering (sorted keys), minimal separators (no whitespace), and string conversion of non-JSON-serializable types. This ensures that hash computation is reproducible regardless of the platform or programming language used for verification.

The hash chain is append-only: once an event is sealed (its event_hash is computed), it is immutable. Any modification to any field of any event in the chain would produce a different hash, causing all subsequent events' previous_hash linkages to fail verification. This provides tamper-evidence equivalent to a blockchain without requiring distributed consensus.

**6.2.3.2 Merkle Tree Verification**

An append-only binary Merkle tree is maintained in memory and checkpointed to persistent storage at configurable intervals. The tree operates as follows:

1. Each forensic event's event_hash is hashed again with SHA-256 to produce a leaf hash, which is appended to the tree.
2. Internal nodes are computed by concatenating the left child hash and right child hash (in that order) and computing SHA-256 of the concatenation.
3. When the number of leaves at any level is odd, the last leaf is duplicated to form a complete binary tree.
4. The root hash serves as a single fixed-length commitment to all events in the system.

The tree supports inclusion proof generation: for any event, a proof consists of the leaf hash, the leaf index, a list of sibling hashes along the path from leaf to root, and the direction (left or right) of each sibling. Verification recomputes the root hash by iteratively hashing the target leaf with its siblings in the correct order, then comparing the result to the claimed root hash.

Merkle checkpoints are persisted to the database at configurable intervals. On system restart, the tree is restored from the latest checkpoint and only events added after the checkpoint are replayed.

**6.2.3.3 Provenance DAG**

A directed acyclic graph tracks the full lineage of every artifact in the system. Each ProvenanceNode contains:
- A UUID identifier
- An entity_type string (e.g., "dataset", "model", "inference", "explanation", "dp_fl_round", "poisoning_detection")
- An entity_id referencing the actual entity
- A creation timestamp
- A metadata dictionary
- A list of parent node UUIDs
- A list of ProvenanceRelation enums (DERIVED_FROM, GENERATED_BY, USED, ATTRIBUTED_TO), one per parent
- A node_hash computed identically to event hashes (SHA-256 of canonical JSON of all fields)

The canonical provenance chain for a recognition decision traces: training_data -> model -> inference -> explanation -> bundle. Each node is hash-sealed upon creation. Chain verification recomputes all node hashes and confirms parent-child linkages.

**6.2.3.4 Forensic Bundle Creation**

A ForensicBundle is the primary output artifact of the system. Bundle creation proceeds as follows:

1. The caller specifies a list of event UUIDs to include in the bundle, along with optional provenance node IDs, bias audit records, and layer-specific artifacts.
2. The system collects Merkle inclusion proofs for each included event.
3. Layer-specific artifacts are gathered from the event chain: recognition artifacts (inference and training events), FL artifacts (round events and security alerts), bias artifacts (audit events), and explanation artifacts (explanation events).
4. A ForensicBundle object is constructed containing all event IDs, the current Merkle root, all Merkle proofs, the provenance chain, the bias audit record, and all layer artifacts.
5. The bundle is sealed by computing a SHA-256 hash over the canonical JSON serialization of all bundle contents (excluding the bundle_hash field itself). This is the bundle_hash.
6. A Schnorr zero-knowledge proof is generated over the bundle_hash (described in Section 6.2.3.5).
7. A Verifiable Credential is issued for the bundle, signed with the platform's Ed25519 DID key (described in Section 6.2.3.7).
8. The complete bundle with ZK proof and VC is persisted to the database.

**6.2.3.5 Zero-Knowledge Proof Generation and Verification (Schnorr)**

The system implements non-interactive Schnorr zero-knowledge proofs converted from the interactive Schnorr identification protocol via the Fiat-Shamir heuristic. The protocol operates over a safe prime group:

- Domain parameters: A 256-bit safe prime p (where q = (p-1)/2 is also prime) and a generator g = 4 of the quadratic residue subgroup of order q.
- Secret derivation: The integer secret is derived deterministically from the bundle hash via SHA-256: secret = int(SHA-256(bundle_hash), 16) mod q.
- Proof generation:
  1. Compute the public point y = g^secret mod p.
  2. Select a random nonce k uniformly from [1, q).
  3. Compute the commitment r = g^k mod p.
  4. Compute the Fiat-Shamir challenge c = SHA-256(g || r || y) mod q, where all values are serialized as zero-padded 64-character hexadecimal strings.
  5. Compute the response s = (k - c * secret) mod q.
  6. The proof consists of {scheme: "schnorr-sha256", commitment: hex(r), challenge: hex(c), response: hex(s), public_point: hex(y)}.

- Proof verification:
  1. Recompute the expected challenge c' = SHA-256(g || r || y) mod q.
  2. Verify c == c' (Fiat-Shamir integrity).
  3. Verify g^s * y^c == r (mod p) (Schnorr verification equation).

This zero-knowledge proof demonstrates knowledge of the secret derived from the bundle hash without revealing the secret itself, enabling a verifier to confirm that the prover had access to the bundle's contents at proof generation time without requiring the verifier to possess those contents.

**6.2.3.6 Ed25519 DID:key Identity System**

The system generates and manages Ed25519 keypairs conforming to the W3C DID:key method specification:

- Key generation: An Ed25519 signing key is generated (or derived deterministically from a 32-byte seed). The corresponding verify (public) key is derived automatically.
- DID construction: The public key bytes are prefixed with the Ed25519 multicodec identifier (0xed01), then base58-encoded (Bitcoin alphabet), and prefixed with "z" to produce the multibase representation. The full DID is "did:key:z" + base58(multicodec_prefix + public_key).
- DID Document resolution: The DID resolves to a W3C DID Document containing a verification method of type Ed25519VerificationKey2020, with authentication and assertionMethod entries.
- Signing: Ed25519 signatures (RFC 8032) are produced over arbitrary data using the private signing key via libsodium (PyNaCl).
- Verification: Signatures are verified against the public verify key. Invalid signatures raise a BadSignatureError.

**6.2.3.7 Verifiable Credentials**

The system issues W3C Verifiable Credentials with Ed25519Signature2020 proofs:

- Credential structure: Follows the W3C Verifiable Credentials Data Model with @context, type (including "ForensicCredential"), issuer (platform DID), issuanceDate, and credentialSubject containing the claims.
- Signing process: The claims dictionary is serialized to canonical JSON (sorted keys, minimal separators). The canonical string is signed with the issuer's Ed25519 private key, producing a 64-byte signature.
- Proof structure: The credential includes a proof object with type "Ed25519Signature2020", creation timestamp, verificationMethod (issuer DID), and proofValue (hex-encoded Ed25519 signature).
- Verification: The verifier reconstructs the canonical claims JSON, retrieves the issuer's public key (from the DID Document or stored form), and verifies the Ed25519 signature against the canonical bytes.

#### 6.2.4 Layer 4: Fairness and Bias Auditor

The fairness layer computes two primary metrics across demographic groups:

1. **Demographic Parity Gap.** The maximum difference in positive prediction rates across demographic groups. A demographic parity gap of 0 indicates perfect parity; values exceeding configurable thresholds (default: 0.1) trigger compliance warnings.

2. **Equalized Odds Gap.** The maximum difference in true positive rates (TPR) and false positive rates (FPR) across demographic groups. This metric ensures that the recognition system performs equally well (and equally poorly) across all groups.

Each bias audit produces a BiasAuditRecord containing the audit UUID, the event UUID it audits, timestamp, demographic_parity_gap, equalized_odds_gap, the list of groups evaluated, a boolean compliance flag, and a details dictionary. This record is embedded in forensic bundles and linked to the specific model and inference events via the provenance DAG.

The auditor supports configurable auto-audit intervals, enabling continuous fairness monitoring without manual intervention. Every audit is recorded as a BIAS_AUDIT forensic event.

#### 6.2.5 Layer 5: Explainability Layer

The explainability layer provides three independent explanation methods, each producing hash-sealed artifacts:

1. **LIME (Local Interpretable Model-agnostic Explanations).** Generates local linear approximations of the recognition model's decision boundary around a specific input. LIME perturbs the input image, observes the model's response to perturbations, and fits a sparse linear model to identify the most influential features. The explanation artifact includes feature importance weights, the local model's fidelity score, and the perturbation parameters used.

2. **KernelSHAP (SHapley Additive exPlanations).** Computes approximate Shapley values for each feature using a weighted kernel-based approach. KernelSHAP provides theoretically grounded feature attribution values that satisfy the properties of local accuracy, missingness, and consistency. The explanation artifact includes per-feature SHAP values and the expected value (base rate).

3. **SDD (Spatial-Directional Decomposition) Saliency.** A novel explainability method that:
   - Computes per-pixel gradients via central finite differences on the raw 112x112 image.
   - Decomposes the gradient map into 7 canonical facial regions (forehead, left eye, right eye, nose, mouth, left jaw, right jaw) with predefined bounding boxes on the 112x112 grid.
   - Scores each region by mean absolute gradient magnitude, normalized to [0, 1].
   - Identifies the dominant contributing region.
   - Produces a full pixel-level saliency map as a 112x112 nested list of floating-point values.

Each SDD explanation is sealed with an artifact hash computed as SHA-256 of the canonical JSON of the inference event ID, predicted label, confidence, saliency map hash, region scores, and dominant region. A ForensicEvent of type EXPLANATION_GENERATED is created with full metadata, and a ProvenanceNode of entity type "explanation" is linked to the inference event via a DERIVED_FROM relation.

#### 6.2.6 Layer 6: Consent and Governance Layer

**6.2.6.1 Consent Management**

The consent management system implements append-only consent tracking:

- **Consent Records.** Each consent record contains a UUID, subject_id (the data subject), purpose (the processing purpose), granted flag (boolean), timestamp, optional expiry datetime, optional revocation_reason, and the event_id of the forensic event that logged this consent change.
- **Append-Only Semantics.** Consent records are never updated or deleted in place. Every state change (grant, revoke) appends a new record, preserving the complete audit history. The current consent state is determined by the latest record for a given subject+purpose pair.
- **Consent Gating.** Before any recognition inference, the system calls require_consent() which checks that active, non-expired consent exists for the subject and purpose. If consent is missing, revoked, or expired, the system: (a) logs a CONSENT_UPDATE forensic event recording the block reason; and (b) raises a ConsentError that prevents the inference from proceeding. The block event itself becomes part of the hash chain, proving that the system enforced consent at that point in time.
- **Expiry Tracking.** Consent records may include an expiry datetime. Expired consent is treated identically to revoked consent for gating purposes.

**6.2.6.2 Automated Compliance Reporting**

The compliance reporter generates EU AI Act assessments covering:

- **Article 5 (Prohibited Practices).** Assesses whether the system engages in prohibited discriminatory practices by checking consent coverage percentage and bias audit pass rate. Both metrics must meet or exceed 70% for a passing assessment.
- **Article 14 (Human Oversight).** Assesses whether the system provides adequate transparency for human oversight by checking explanation coverage percentage and bundle integrity percentage. Both metrics must meet or exceed 70% for a passing assessment.
- **Metrics.** The report includes: consent_coverage_pct, bias_audit_pass_rate_pct, explanation_coverage_pct, and bundle_integrity_pct.
- **Overall Compliance Score.** A weighted average of all four metrics with weights: consent 30%, bias audit 25%, explanation 25%, bundle integrity 20%. A score of 70 or above indicates overall compliance.
- **Forensic Integration.** Each compliance report generation is itself recorded as a COMPLIANCE_REPORT forensic event containing the report_id, overall_compliance_score, compliant flag, and all metrics. This event is hash-chained into the same chain as all other system operations.

**6.2.6.3 OSCAL Export**

The system supports export of security assessment data in NIST OSCAL (Open Security Controls Assessment Language) format for interoperability with government and enterprise compliance frameworks.

### 6.3 JSON-LD Export Format

Forensic bundles can be exported in JSON-LD (JavaScript Object Notation for Linked Data) format conforming to W3C standards. The export includes:
- @context references to W3C credentials, security, and custom forensic vocabulary namespaces
- The complete bundle structure with all event hashes, Merkle proofs, provenance chain, bias audit, and layer artifacts
- The Schnorr ZK proof and DID-signed Verifiable Credential
- Full round-trip capability: exported bundles can be re-imported and verified against the original hash

### 6.4 Bundle Verification Process

Bundle verification is a seven-step process:

1. **Bundle Hash Integrity.** Recompute the SHA-256 hash of the bundle's contents and compare to the stored bundle_hash. Any modification to any field invalidates the hash.
2. **Merkle Proof Verification.** For each Merkle inclusion proof in the bundle, recompute the root hash by iteratively hashing the leaf with its sibling path and confirm it matches the stored root_hash.
3. **Provenance Chain Verification.** For each provenance node in the bundle's provenance_chain, recompute the node_hash and confirm it matches the stored hash.
4. **Layer Artifact Verification.** For each event referenced in the layer artifacts (recognition, FL, bias, explanation), confirm the event exists in the database and its stored event_hash matches the hash recorded in the artifact.
5. **ZK Proof Verification.** Parse the Schnorr proof, recompute the Fiat-Shamir challenge, and verify the Schnorr verification equation g^s * y^c == r (mod p).
6. **DID Credential Verification.** Parse the Verifiable Credential, reconstruct the canonical claims JSON, retrieve the issuer's Ed25519 public key, and verify the Ed25519 signature.
7. **Overall Determination.** The bundle is marked VERIFIED if and only if all six verification steps pass. Any single failure marks the bundle as TAMPERED.

---

## 7. CLAIMS

### Independent Claim 1: Unified Forensic Bundle

**1.** A computer-implemented method for generating a self-verifiable forensic bundle for artificial intelligence facial recognition operations, the method comprising:

(a) recording, by a processor, each operation performed by a facial recognition system as a forensic event, each forensic event comprising an event identifier, an event type, a timestamp, an actor identifier, a payload, a previous hash referencing the hash of the immediately preceding forensic event in a sequential chain, a sequence number, and an event hash computed as a cryptographic hash of the canonical serialization of the event fields concatenated with the previous hash;

(b) inserting, by the processor, the event hash of each forensic event as a leaf into an append-only Merkle tree and computing, upon each insertion, an updated Merkle root hash;

(c) creating, by the processor, provenance nodes in a directed acyclic graph, each provenance node comprising an entity type, an entity identifier, parent node references, relation types, metadata, and a node hash computed as a cryptographic hash of the canonical serialization of the node fields;

(d) collecting, by the processor, Merkle inclusion proofs for a specified set of forensic events, each proof comprising the leaf hash, leaf index, a list of sibling hashes, sibling directions, and the root hash;

(e) aggregating, by the processor, the specified forensic event identifiers, the Merkle root hash, the Merkle inclusion proofs, the provenance node identifiers, bias audit records, recognition artifacts, federated learning artifacts, explainability artifacts, and layer-specific metadata into a forensic bundle data structure;

(f) computing, by the processor, a bundle hash as a cryptographic hash of the canonical serialization of the aggregated bundle contents;

(g) generating, by the processor, a zero-knowledge proof over the bundle hash using a Schnorr identification protocol converted to non-interactive form via a Fiat-Shamir heuristic, the proof comprising a commitment, a challenge derived from a cryptographic hash of domain parameters and the commitment, and a response;

(h) issuing, by the processor, a verifiable credential for the bundle, the credential comprising claims referencing the bundle identifier and bundle hash, signed with an Ed25519 private key corresponding to a decentralized identifier of the issuing platform; and

(i) storing, by the processor, the forensic bundle with the bundle hash, zero-knowledge proof, and verifiable credential as a single self-verifiable artifact.

### Dependent Claims on Claim 1

**2.** The method of claim 1, wherein the cryptographic hash function used for event hashing, Merkle tree construction, provenance node hashing, and bundle hashing is SHA-256, and the canonical serialization uses deterministic JSON with sorted keys, minimal separators, and string conversion of non-native types.

**3.** The method of claim 1, wherein the append-only Merkle tree duplicates the last leaf when the number of leaves at any level is odd to form a complete binary tree, and wherein Merkle tree state is checkpointed to persistent storage at configurable intervals and restored via incremental replay on system restart.

**4.** The method of claim 1, wherein the directed acyclic provenance graph tracks lineage across entity types including training data, trained models, inference results, explanations, federated learning rounds, poisoning detection results, and forensic bundles, with relation types including DERIVED_FROM, GENERATED_BY, USED, and ATTRIBUTED_TO.

**5.** The method of claim 1, wherein the Schnorr zero-knowledge proof uses a 256-bit safe prime p where q = (p-1)/2 is also prime, a generator g of the quadratic residue subgroup of order q, and wherein the secret is derived deterministically from the bundle hash as int(SHA-256(bundle_hash), 16) mod q.

**6.** The method of claim 1, wherein the decentralized identifier is constructed using the DID:key method by prefixing the Ed25519 public key bytes with a multicodec identifier, base58-encoding the result, and prepending "did:key:z" to form a self-describing identifier that resolves to a W3C DID Document.

**7.** The method of claim 1, wherein the verifiable credential conforms to the W3C Verifiable Credentials Data Model, uses Ed25519Signature2020 as the proof type, and the proof value is computed by signing the canonical JSON serialization of the credential claims with the issuer's Ed25519 private key.

**8.** The method of claim 1, wherein bundle verification comprises: recomputing the bundle hash and comparing to the stored hash; verifying each Merkle inclusion proof by iteratively hashing from leaf to root; verifying provenance node hashes; verifying layer artifact event hashes against stored events; recomputing the Fiat-Shamir challenge and verifying the Schnorr equation g^s * y^c == r (mod p); and verifying the Ed25519 credential signature against the issuer's public key.

**9.** The method of claim 1, further comprising exporting the forensic bundle in JSON-LD format with W3C-standard @context references, enabling round-trip import and re-verification.

### Independent Claim 2: Consent-Gated Recognition Pipeline

**10.** A computer-implemented method for consent-gated facial recognition with forensic logging, the method comprising:

(a) maintaining, by a processor, an append-only store of consent records, each consent record comprising a subject identifier, a processing purpose, a granted flag, a timestamp, an optional expiry datetime, an optional revocation reason, and a reference to a forensic event identifier, wherein consent state changes are recorded by appending new records rather than modifying existing records;

(b) receiving, by the processor, a facial recognition inference request specifying a subject identifier and a processing purpose;

(c) querying, by the processor, the consent store to retrieve the most recent consent record for the specified subject and purpose;

(d) determining, by the processor, whether the retrieved consent record indicates active consent by verifying that the granted flag is true and, if an expiry datetime is present, that the current time has not exceeded the expiry datetime;

(e) upon determining that active consent does not exist, recording, by the processor, a forensic event of type CONSENT_UPDATE with a payload indicating an action of "block" and a reason of "no_consent_record", "consent_revoked", or "consent_expired", and preventing the inference from proceeding; and

(f) upon determining that active consent exists, permitting the inference to proceed and recording the inference result as a forensic event hash-chained to the consent verification event.

### Dependent Claims on Claim 10

**11.** The method of claim 10, wherein each consent grant and each consent revocation is logged as a CONSENT_UPDATE forensic event that is hash-chained into the same sequential event chain as all other system operations, creating a unified audit trail spanning consent management and recognition operations.

**12.** The method of claim 10, wherein the consent block forensic event contains the subject identifier, the processing purpose, the block reason, and, in the case of expired consent, the expiry datetime, thereby providing a cryptographically verifiable record that the system enforced consent at the specific point in time.

**13.** The method of claim 10, wherein the consent history for a given subject and purpose is retrievable as a chronologically ordered list of all consent records, enabling reconstruction of the complete consent lifecycle including all grants, revocations, expirations, and blocks.

### Independent Claim 3: Forensic Federated Learning with Provenance

**14.** A computer-implemented method for federated learning with differential privacy, poisoning detection, and forensic provenance, the method comprising:

(a) receiving, by a processor, weight updates from a plurality of distributed clients, each weight update comprising a list of numerical arrays representing locally trained model parameters;

(b) computing, by the processor, for each client, a weight delta as the element-wise difference between the client's weight update and the current global model weights;

(c) clipping, by the processor, each client's weight delta to a configurable maximum L2 norm, recording which clients' deltas were clipped;

(d) averaging, by the processor, the clipped weight deltas across all participating clients;

(e) computing, by the processor, a noise scale sigma using an analytic Gaussian mechanism bound sigma = (sensitivity * sqrt(2 * ln(1.25 / delta))) / epsilon, where sensitivity = 1 / n_clients, epsilon is the per-round privacy budget, and delta is the privacy failure probability;

(f) adding, by the processor, Gaussian noise with standard deviation sigma to the averaged weight delta using a seeded random number generator for reproducibility;

(g) applying, by the processor, the noisy averaged delta to the global model weights to produce updated global weights;

(h) tracking, by the processor, cumulative privacy expenditure across rounds via composition;

(i) computing, by the processor, the L2 norm of each client's weight delta and flagging clients whose norm exceeds a configurable multiple of the median norm as potentially poisoned, recording a SECURITY_ALERT forensic event for each flagged client;

(j) recording, by the processor, the completed round as a ForensicEvent of type FL_ROUND containing the round number, number of clients, SHA-256 hash of serialized global model weights, aggregation strategy, differential privacy configuration, noise scale, clipped client list, and cumulative privacy expenditure; and

(k) creating, by the processor, a provenance node of entity type "dp_fl_round" linked to prior round provenance nodes via DERIVED_FROM relations and sealed with a cryptographic hash.

### Dependent Claims on Claim 14

**15.** The method of claim 14, wherein the poisoning detection step further comprises creating a provenance node of entity type "poisoning_detection" linked to the FL round provenance node, containing metadata including the median norm, threshold multiplier, effective threshold, and list of flagged client identifiers.

**16.** The method of claim 14, wherein each SECURITY_ALERT forensic event for a flagged client is hash-chained into the same sequential event chain as the FL_ROUND event, with each alert's previous_hash referencing the hash of the preceding alert or the FL_ROUND event.

**17.** The method of claim 14, wherein the SHA-256 hash of the serialized global model weights provides a cryptographic fingerprint that enables post-hoc verification that a specific model checkpoint was the output of a specific federated learning round.

### Independent Claim 4: Multi-Layer Explainability with Forensic Verification

**18.** A computer-implemented method for generating forensically verifiable explainability artifacts for facial recognition decisions, the method comprising:

(a) receiving, by a processor, an inference result comprising a predicted identity label and a confidence score for a facial image processed by a recognition model;

(b) generating, by the processor, a first explanation using Local Interpretable Model-agnostic Explanations (LIME) by perturbing the input image, observing model responses, and fitting a local linear model;

(c) generating, by the processor, a second explanation using Kernel Shapley Additive Explanations (KernelSHAP) by computing approximate Shapley values for each feature;

(d) generating, by the processor, a third explanation using Spatial-Directional Decomposition (SDD) by:
   (i) computing per-pixel gradients via central finite differences on the raw image vector;
   (ii) reshaping the gradient vector into a two-dimensional gradient map corresponding to the image dimensions;
   (iii) decomposing the gradient map into a plurality of canonical facial regions with predefined bounding boxes;
   (iv) computing, for each canonical facial region, an importance score based on the mean absolute gradient magnitude within the region's bounding box, normalized to a range of [0, 1]; and
   (v) identifying the dominant contributing region as the region with the highest importance score;

(e) computing, by the processor, for each explanation, an artifact hash as a cryptographic hash of the canonical serialization of the explanation contents including the inference event identifier, predicted label, confidence, saliency data, region scores, and dominant region;

(f) recording, by the processor, for each explanation, a forensic event of type EXPLANATION_GENERATED containing the explanation type, method, inference event reference, artifact hash, and region-level detail; and

(g) creating, by the processor, for each explanation, a provenance node of entity type "explanation" linked to the inference event's provenance node via a DERIVED_FROM relation.

### Dependent Claims on Claim 18

**19.** The method of claim 18, wherein the canonical facial regions for SDD decomposition comprise forehead, left eye, right eye, nose, mouth, left jaw, and right jaw, each defined by fixed pixel coordinate bounding boxes on a 112x112 image grid.

**20.** The method of claim 18, further comprising generating a pixel-level saliency map as a two-dimensional array of normalized floating-point values, the saliency map providing a visual representation of per-pixel contributions to the recognition decision.

**21.** The method of claim 18, further comprising comparing the LIME and KernelSHAP explanations for the same inference event to identify agreement or disagreement between explanation methods, the comparison result being stored as a forensic event.

**22.** The method of claim 18, wherein the explainability artifacts are included in a forensic bundle along with the recognition artifacts, federated learning artifacts, bias audit records, consent records, Merkle proofs, Schnorr zero-knowledge proof, and DID-signed Verifiable Credential, forming a single self-verifiable artifact containing the complete evidential chain from training through explanation.

### System Claim

**23.** A forensic-friendly artificial intelligence system comprising:

a processor;

a memory coupled to the processor and storing instructions that, when executed by the processor, cause the system to:

(a) operate a six-layer architecture comprising: a recognition engine layer for facial recognition using dimensionality reduction and classification; a federated learning engine layer for privacy-preserving distributed model training with differential privacy and poisoning detection; a blockchain forensic layer for hash-chained event recording, Merkle tree verification, provenance DAG tracking, zero-knowledge proof generation, and decentralized identity credential issuance; a fairness and bias auditing layer for computing demographic parity and equalized odds metrics; an explainability layer for generating LIME, KernelSHAP, and SDD saliency explanations; and a consent and governance layer for consent-gated recognition and automated regulatory compliance reporting;

(b) record every operation across all six layers as an immutable forensic event hash-chained into a sequential chain using SHA-256;

(c) maintain an append-only Merkle tree indexing all forensic event hashes;

(d) maintain a directed acyclic provenance graph tracking lineage across all entity types;

(e) generate self-verifiable forensic bundles combining hash-chain integrity, Merkle inclusion proofs, provenance paths, Schnorr zero-knowledge proofs, Ed25519 DID-signed Verifiable Credentials, bias audit records, multi-method explainability artifacts, and consent verification records; and

(f) produce automated EU AI Act compliance reports that are themselves forensic events linked into the hash chain.

**24.** The system of claim 23, wherein the system further comprises a real-time event streaming interface using Server-Sent Events (SSE) for live forensic event monitoring, an API providing 84 or more endpoints across 12 functional domains, role-based access control with audit logging, and rate limiting on sensitive endpoints.

**25.** The system of claim 23, wherein the system is deployable as a containerized application with a multi-stage Docker build, serves a React-based dashboard as static files from the same server process, and supports both SQLite and cloud-hosted PostgreSQL-compatible database backends.

---

## 8. DESCRIPTION OF FIGURES

### Figure 1: Six-Layer System Architecture

Figure 1 depicts the overall six-layer architecture of the forensic-friendly AI facial recognition system. The figure shows six horizontally stacked layers labeled from bottom to top: Layer 1 (Recognition Engine) containing PCA, SVM, Voice Biometrics, and Multi-Modal Fusion components; Layer 2 (Federated Learning Engine) containing FedAvg, DP-FedAvg, Poisoning Detection, and Privacy Budget Tracking components; Layer 3 (Blockchain Forensic Layer) containing Hash Chain, Merkle Tree, Provenance DAG, ZK Proofs, and DID/VC components; Layer 4 (Fairness and Bias Auditor) containing Demographic Parity, Equalized Odds, and Auto-Audit components; Layer 5 (Explainability) containing LIME, KernelSHAP, and SDD Saliency components; and Layer 6 (Consent and Governance) containing Consent Engine, Compliance Reporter, and EU AI Act Checks components. Vertical arrows show that every component in every layer produces ForensicEvents that flow into Layer 3's hash chain. A persistent storage layer (SQLite/Supabase) underlies all layers.

### Figure 2: Forensic Event Hash Chain

Figure 2 depicts the sequential hash chain of forensic events. The figure shows a series of ForensicEvent blocks arranged horizontally, each containing fields: id, event_type, timestamp, actor, payload, previous_hash, sequence_number, and event_hash. Arrows labeled "previous_hash" connect each event to its predecessor, with the first event's previous_hash pointing to a "GENESIS" sentinel. Each event's event_hash is shown as the output of a SHA-256 function applied to the canonical JSON of the event's fields concatenated with the previous_hash. The figure illustrates how modification of any event's fields would invalidate its hash and break the chain for all subsequent events.

### Figure 3: Forensic Bundle Creation Flow

Figure 3 depicts the process of creating a self-verifiable forensic bundle. The flow begins with a set of event IDs as input. Step 1 collects layer-specific artifacts (recognition, FL, bias, explanation) from the event chain. Step 2 gathers Merkle inclusion proofs for each event. Step 3 assembles the provenance chain. Step 4 constructs the ForensicBundle object and computes the bundle_hash via SHA-256. Step 5 generates a Schnorr ZK proof over the bundle_hash. Step 6 issues an Ed25519-signed Verifiable Credential for the bundle. Step 7 persists the complete bundle. The output is a single self-verifiable artifact containing all components.

### Figure 4: Provenance DAG Example

Figure 4 depicts an example provenance directed acyclic graph tracing the lineage of a recognition decision. The DAG shows nodes of types: "dataset" at the root, connected via USED relations to a "model" node (training), connected via GENERATED_BY to an "inference" node, connected via DERIVED_FROM to two "explanation" nodes (LIME and SDD), all connected via DERIVED_FROM to a "bundle" node at the leaf. Each node shows its entity_type, entity_id, parents list, relations list, and node_hash. Arrows are directed from child to parent (upstream lineage). A separate branch shows "dp_fl_round" nodes chained via DERIVED_FROM relations, with a "poisoning_detection" node branching off the FL round node.

### Figure 5: Consent-Gated Recognition Pipeline

Figure 5 depicts the consent verification flow that gates recognition inference. The flow begins with an inference request containing subject_id and purpose. Step 1 queries the consent store for the latest consent record. Decision point A checks whether a record exists (if not, log block event with reason "no_consent_record" and return 403). Decision point B checks whether granted == true (if not, log block event with reason "consent_revoked" and return 403). Decision point C checks whether the current time is before the expiry (if expired, log block event with reason "consent_expired" and return 403). If all checks pass, the inference proceeds and produces a forensic event hash-chained after the consent verification. The figure shows that every branch (grant, revoke, block, inference) produces a forensic event in the same hash chain.

### Figure 6: ZK Proof Generation and Verification

Figure 6 depicts the Schnorr zero-knowledge proof lifecycle. The generation side shows: bundle_hash as input; secret derivation via SHA-256; public point computation y = g^secret mod p; random nonce k; commitment r = g^k mod p; Fiat-Shamir challenge c = SHA-256(g || r || y) mod q; response s = (k - c * secret) mod q; and output proof {commitment, challenge, response, public_point}. The verification side shows: proof as input; recomputation of expected challenge c'; verification that c == c'; computation of g^s * y^c mod p; comparison with r; and output valid/invalid. The figure emphasizes that the verifier never learns the secret.

### Figure 7: Federated Learning with Differential Privacy Flow

Figure 7 depicts a single DP-FedAvg round. The flow shows N distributed clients each performing local training on private data and submitting weight updates to the central aggregator. The aggregator: (1) computes weight deltas (client - global); (2) clips each delta to max_grad_norm; (3) averages clipped deltas; (4) computes noise scale sigma from epsilon, delta, and n_clients; (5) adds Gaussian noise N(0, sigma); (6) applies noisy delta to global weights. A parallel poisoning detection module computes L2 norms of all deltas, determines the median, flags outliers exceeding threshold_multiplier * median, and generates SECURITY_ALERT events. Both the FL round and poisoning detection results produce ForensicEvents and ProvenanceNodes.

### Figure 8: Compliance Report Generation

Figure 8 depicts the automated EU AI Act compliance report generation process. The flow shows four metric collection steps: (1) consent coverage statistics from the consent records table; (2) bias audit statistics from the bias audit records table; (3) explanation coverage statistics from the explanation events; (4) bundle integrity statistics from the verified bundles. These feed into two article assessments: Article 5 (Prohibited Practices) using consent + bias metrics, and Article 14 (Human Oversight) using explanation + bundle metrics. The overall compliance score is computed as the weighted average (consent 30%, bias 25%, explanation 25%, bundle 20%). The report is output as a structured JSON object, and a COMPLIANCE_REPORT forensic event is recorded in the hash chain, linking the compliance assessment itself into the forensic record.

---

## 9. PRIOR ART REFERENCES

### 9.1 Academic Publications

1. S. Mohammed, "Forensic-Friendly AI Framework for Facial Recognition Systems," in *Proceedings of the International Conference on Digital Forensics and Cyber Crime (ICDF2C)*, 2024. [Six-layer forensic-friendly AI schema; hash-chained events; provenance DAG; forensic bundles.]

2. J. Li et al., "BioZero: Privacy-Preserving Biometric Verification via Zero-Knowledge Proofs," arXiv:2409.17509, 2024. [Zero-knowledge biometric verification; Merkle tree integrity; ZK proof protocols for biometric systems.]

3. Y. Chen et al., "TBFL: Trustworthy Blockchain-Based Federated Learning with Decentralized Identifiers and Verifiable Credentials," arXiv:2602.02629, 2026. [DID:key identity for FL participants; Verifiable Credentials for model provenance; blockchain-based trust anchoring.]

4. W. Zhang et al., "FedFDP: Fairness-Aware Federated Learning with Differential Privacy," arXiv:2402.16028, 2024. [DP-FedAvg with calibrated Gaussian noise; fairness-constrained gradient clipping; privacy budget composition.]

5. H. Wang et al., "SDD: Saliency-Driven Decomposition for Explainable Face Recognition," arXiv:2505.03837, 2025. [Spatial-directional saliency decomposition; canonical facial region scoring; pixel-level gradient-based explanations.]

6. K. Schmidt et al., "Towards EU AI Act Compliance: A Framework for Automated Compliance Assessment of High-Risk AI Systems," arXiv:2512.13907, 2025. [Automated compliance metrics; Article 5 and Article 14 assessment; compliance scoring methodology.]

### 9.2 Standards and Specifications

7. W3C, "Decentralized Identifiers (DIDs) v1.0," W3C Recommendation, July 2022.

8. W3C, "Verifiable Credentials Data Model v1.1," W3C Recommendation, March 2022.

9. D. J. Bernstein et al., "Ed25519: High-Speed High-Security Signatures," in *Journal of Cryptographic Engineering*, 2012. [RFC 8032.]

10. C. P. Schnorr, "Efficient Signature Generation by Smart Cards," in *Journal of Cryptology*, vol. 4, no. 3, pp. 161-174, 1991. [Schnorr identification protocol; Fiat-Shamir heuristic.]

11. NIST, "OSCAL: Open Security Controls Assessment Language," NIST Special Publication, 2023.

12. European Union, "Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (Artificial Intelligence Act)," Official Journal of the European Union, 2024.

### 9.3 Commercial Systems

13. Clearview AI, Inc. Facial recognition platform with web-scraped image database. No public forensic chain, provenance DAG, bias auditing, explainability, or consent management.

14. NEC Corporation, "NeoFace" facial recognition engine. High-accuracy biometric matching. No self-verifiable forensic bundles, ZK proofs, or DID-based identity.

15. Amazon Web Services, "Amazon Rekognition." Cloud-based facial recognition API. Stateless API responses with no hash chaining, Merkle verification, or provenance tracking.

16. Microsoft Corporation, "Azure Face API." Cloud-based face detection, verification, and identification. No integrated forensic chain, consent gating, or automated compliance reporting.

---

## 10. INVENTOR AND ASSIGNEE INFORMATION

**Inventor:**
Name: Dico Angelo
Citizenship: [To be completed]
Residence: [To be completed]

**Assignee:**
Name: Metaventions AI
Type: Organization
Address: [To be completed]

**Correspondence Address:**
[To be completed by patent counsel]

**Filing Date:** [To be completed]

**Provisional Application Number:** [To be assigned by USPTO]

---

## DECLARATION

The above-named inventor(s) believe(s) to be the original inventor(s) of the subject matter which is claimed and for which a patent is sought on the invention titled "System and Method for Forensic-Friendly Artificial Intelligence with Unified Cryptographic Provenance and Multi-Layer Verification," the specification of which is attached hereto.

---

*This provisional patent application was prepared based on a working implementation deployed at friendlyface.metaventionsai.com, comprising approximately 44,500 lines of Python backend code, 7,500 lines of React frontend code, 1,422 tests at 93% code coverage, 20 database tables, and 84+ API endpoints. The implementation is publicly available at github.com/Dicoangelo/FriendlyFace under the MIT license. The source code repository constitutes additional disclosure supporting this provisional application.*
