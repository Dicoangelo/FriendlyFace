"""Tests for EU AI Act Annex IV Conformity Assessment (US-006).

Covers:
  - Document generation returns all 10 sections
  - Completeness scoring is correct
  - Sections have correct structure (section_id, title, status, content, evidence, gaps)
  - With seeded data: sections populate from real forensic data
  - With empty database: all sections are "missing" or "partial"
  - HTML output is valid (contains key sections)
  - API endpoint returns 201
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import friendlyface.api.app as app_module
from friendlyface.api.app import _db, _service, app, limiter
from friendlyface.core.models import BiasAuditRecord, EventType
from friendlyface.core.service import ForensicService
from friendlyface.governance.conformity import ConformityAssessmentGenerator
from friendlyface.recognition.gallery import FaceGallery
from friendlyface.storage.database import Database


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db(tmp_path):
    """Fresh database for each test."""
    database = Database(tmp_path / "conformity_test.db")
    await database.connect()
    yield database
    await database.close()


@pytest_asyncio.fixture
async def service(db):
    """Fresh forensic service for each test."""
    svc = ForensicService(db)
    await svc.initialize()
    return svc


@pytest_asyncio.fixture
async def generator(db, service):
    """ConformityAssessmentGenerator wired to fresh db + forensic service."""
    return ConformityAssessmentGenerator(db, service)


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database."""
    _db.db_path = tmp_path / "conformity_api_test.db"
    await _db.connect()
    await _db.run_migrations()
    await _service.initialize()

    _service.merkle = __import__(
        "friendlyface.core.merkle", fromlist=["MerkleTree"]
    ).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()
    app_module._dashboard_cache["data"] = None
    app_module._dashboard_cache["timestamp"] = 0.0
    app_module._explanations.clear()
    app_module._model_registry.clear()
    app_module._fl_simulations.clear()
    app_module._latest_compliance_report = None
    app_module._auto_audit_interval = 50
    app_module._recognition_event_count = 0
    app_module._gallery = FaceGallery(_db)
    from friendlyface.recognition.pipeline import RecognitionPipeline

    app_module._recognition_pipeline = RecognitionPipeline(gallery=app_module._gallery)

    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


# ---------------------------------------------------------------------------
# Helper to seed data
# ---------------------------------------------------------------------------


async def _seed_forensic_data(db: Database, service: ForensicService) -> None:
    """Seed the database with realistic forensic events and audits."""
    # Training events
    await service.record_event(
        event_type=EventType.TRAINING_START,
        actor="trainer",
        payload={"dataset": "lfw", "size": 13000},
    )
    await service.record_event(
        event_type=EventType.TRAINING_COMPLETE,
        actor="trainer",
        payload={"accuracy": 0.94, "n_subjects": 50},
    )

    # Model registration
    from datetime import datetime, timezone

    await db.insert_model(
        "model-001",
        datetime.now(timezone.utc).isoformat(),
        {"accuracy": 0.94, "n_components": 50, "n_subjects": 50},
    )

    # Inference
    await service.record_event(
        event_type=EventType.INFERENCE_RESULT,
        actor="recognizer",
        payload={"subject_id": "subj-001", "confidence": 0.92},
    )

    # Explanation
    await service.record_event(
        event_type=EventType.EXPLANATION_GENERATED,
        actor="explainer",
        payload={"method": "lime", "event_id": "inference-001"},
    )

    # Consent
    from uuid import uuid4 as _uuid4

    await db.insert_consent_record(
        record_id=str(_uuid4()),
        subject_id="subj-001",
        purpose="identification",
        granted=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Bias audit
    audit = BiasAuditRecord(
        demographic_parity_gap=0.05,
        equalized_odds_gap=0.03,
        groups_evaluated=["male", "female", "other"],
        compliant=True,
    )
    await db.insert_bias_audit(audit)

    # FL round
    await service.record_event(
        event_type=EventType.FL_ROUND,
        actor="fl_coordinator",
        payload={"round": 1, "clients": 3, "accuracy": 0.88},
    )

    # Bundle
    events = await db.get_all_events()
    event_ids = [e.id for e in events[:2]]
    await service.create_bundle(event_ids=event_ids)

    # Compliance report
    await db.insert_compliance_report(
        "report-001",
        datetime.now(timezone.utc).isoformat(),
        {"overall_compliance_score": 85.0, "compliant": True},
    )

    # Provenance node
    service.add_provenance_node(
        entity_type="dataset",
        entity_id="lfw-v1",
        metadata={"size": 13000, "source": "LFW"},
    )


# ---------------------------------------------------------------------------
# Unit tests — ConformityAssessmentGenerator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_10_sections(generator):
    """Document generation returns exactly 10 sections."""
    doc = await generator.generate()
    assert len(doc["sections"]) == 10


@pytest.mark.asyncio
async def test_section_structure(generator):
    """Each section has the required keys."""
    doc = await generator.generate()
    required_keys = {"section_id", "title", "status", "content", "evidence", "gaps"}
    for section in doc["sections"]:
        assert required_keys.issubset(section.keys()), (
            f"Section {section.get('section_id')} missing keys: "
            f"{required_keys - section.keys()}"
        )


@pytest.mark.asyncio
async def test_section_status_values(generator):
    """Section status must be one of: complete, partial, missing."""
    doc = await generator.generate()
    valid_statuses = {"complete", "partial", "missing"}
    for section in doc["sections"]:
        assert section["status"] in valid_statuses, (
            f"Section {section['section_id']} has invalid status: {section['status']}"
        )


@pytest.mark.asyncio
async def test_empty_database_sections_not_complete(generator):
    """With an empty database, no section should be 'complete'."""
    doc = await generator.generate()
    for section in doc["sections"]:
        assert section["status"] != "complete", (
            f"Section {section['section_id']} should not be complete on empty DB"
        )


@pytest.mark.asyncio
async def test_empty_database_has_gaps(generator):
    """With an empty database, gaps should be populated."""
    doc = await generator.generate()
    assert len(doc["gaps"]) > 0, "Empty database should produce gaps"


@pytest.mark.asyncio
async def test_empty_database_completeness_score(generator):
    """With empty database, completeness score should be low."""
    doc = await generator.generate()
    # Section 3 (risk classification) will be partial since it's mostly documentary
    # so score won't be 0, but should be well below 100
    assert doc["completeness"]["score_pct"] < 50.0
    assert doc["completeness"]["complete"] == 0


@pytest.mark.asyncio
async def test_document_metadata(generator):
    """Document metadata fields are correctly populated."""
    doc = await generator.generate()
    assert doc["document_type"] == "eu_ai_act_annex_iv"
    assert doc["system_id"] == "friendlyface"
    assert doc["system_name"] == "FriendlyFace"
    assert doc["regulation"] == "EU AI Act (Regulation 2024/1689)"
    assert doc["risk_classification"] == "high-risk"
    assert doc["enforcement_deadline"] == "2026-08-02"
    assert "Article 11" in doc["applicable_articles"]
    assert doc["completeness"]["total_sections"] == 10


@pytest.mark.asyncio
async def test_custom_system_id(generator):
    """Custom system_id and system_name are used."""
    doc = await generator.generate(system_id="my-system", system_name="My System")
    assert doc["system_id"] == "my-system"
    assert doc["system_name"] == "My System"


@pytest.mark.asyncio
async def test_section_ids_are_unique(generator):
    """All section_ids should be unique."""
    doc = await generator.generate()
    ids = [s["section_id"] for s in doc["sections"]]
    assert len(ids) == len(set(ids)), "Duplicate section_ids found"


@pytest.mark.asyncio
async def test_section_ids_ordered(generator):
    """Sections should be numbered annex_iv_1 through annex_iv_10."""
    doc = await generator.generate()
    expected_ids = [f"annex_iv_{i}" for i in range(1, 11)]
    actual_ids = [s["section_id"] for s in doc["sections"]]
    assert actual_ids == expected_ids


# ---------------------------------------------------------------------------
# Tests with seeded data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seeded_data_improves_completeness(db, service):
    """Seeding data should improve completeness score."""
    gen = ConformityAssessmentGenerator(db, service)

    # Empty
    doc_empty = await gen.generate()
    empty_score = doc_empty["completeness"]["score_pct"]

    # Seed
    await _seed_forensic_data(db, service)

    # Re-generate
    doc_seeded = await gen.generate()
    seeded_score = doc_seeded["completeness"]["score_pct"]

    assert seeded_score > empty_score, (
        f"Seeded score {seeded_score} should be > empty score {empty_score}"
    )


@pytest.mark.asyncio
async def test_seeded_section_1_has_data(db, service):
    """Section 1 should contain event counts and model info after seeding."""
    await _seed_forensic_data(db, service)
    gen = ConformityAssessmentGenerator(db, service)
    doc = await gen.generate()
    s1 = doc["sections"][0]
    assert s1["section_id"] == "annex_iv_1"
    assert s1["content"]["total_forensic_events"] > 0
    assert s1["content"]["registered_models"] > 0
    assert s1["status"] == "complete"


@pytest.mark.asyncio
async def test_seeded_section_4_has_training_data(db, service):
    """Section 4 should reflect training events after seeding."""
    await _seed_forensic_data(db, service)
    gen = ConformityAssessmentGenerator(db, service)
    doc = await gen.generate()
    s4 = doc["sections"][3]
    assert s4["section_id"] == "annex_iv_4"
    assert s4["content"]["training_events_started"] > 0
    assert s4["content"]["training_events_completed"] > 0
    assert len(s4["evidence"]) > 0


@pytest.mark.asyncio
async def test_seeded_section_5_has_bias_data(db, service):
    """Section 5 should reflect bias audits after seeding."""
    await _seed_forensic_data(db, service)
    gen = ConformityAssessmentGenerator(db, service)
    doc = await gen.generate()
    s5 = doc["sections"][4]
    assert s5["section_id"] == "annex_iv_5"
    assert s5["content"]["total_audits"] > 0
    assert len(s5["content"]["demographic_groups_tested"]) >= 2
    assert s5["status"] == "complete"


@pytest.mark.asyncio
async def test_seeded_section_6_has_model_data(db, service):
    """Section 6 should have model and inference data after seeding."""
    await _seed_forensic_data(db, service)
    gen = ConformityAssessmentGenerator(db, service)
    doc = await gen.generate()
    s6 = doc["sections"][5]
    assert s6["section_id"] == "annex_iv_6"
    assert s6["content"]["total_models"] > 0
    assert s6["content"]["total_inferences"] > 0
    assert s6["status"] == "complete"


@pytest.mark.asyncio
async def test_seeded_section_7_has_oversight_data(db, service):
    """Section 7 should have consent and explanation data after seeding."""
    await _seed_forensic_data(db, service)
    gen = ConformityAssessmentGenerator(db, service)
    doc = await gen.generate()
    s7 = doc["sections"][6]
    assert s7["section_id"] == "annex_iv_7"
    assert s7["content"]["consent_coverage"]["total_subjects"] > 0
    assert s7["content"]["explanation_coverage"]["total_explanations"] > 0
    assert s7["status"] == "complete"


@pytest.mark.asyncio
async def test_seeded_section_8_has_integrity_data(db, service):
    """Section 8 should show valid chain integrity after seeding."""
    await _seed_forensic_data(db, service)
    gen = ConformityAssessmentGenerator(db, service)
    doc = await gen.generate()
    s8 = doc["sections"][7]
    assert s8["section_id"] == "annex_iv_8"
    assert s8["content"]["hash_chain_integrity"]["valid"] is True
    assert s8["content"]["hash_chain_integrity"]["event_count"] > 0
    assert s8["status"] == "complete"


@pytest.mark.asyncio
async def test_seeded_section_9_has_quality_data(db, service):
    """Section 9 should have events, bundles, and reports after seeding."""
    await _seed_forensic_data(db, service)
    gen = ConformityAssessmentGenerator(db, service)
    doc = await gen.generate()
    s9 = doc["sections"][8]
    assert s9["section_id"] == "annex_iv_9"
    assert s9["content"]["total_forensic_events"] > 0
    assert s9["content"]["total_bundles"] > 0
    assert s9["content"]["compliance_reports_generated"] > 0
    assert s9["status"] == "complete"


@pytest.mark.asyncio
async def test_completeness_scoring_math(generator):
    """Verify completeness scoring arithmetic."""
    doc = await generator.generate()
    c = doc["completeness"]
    assert c["complete"] + c["partial"] + c["missing"] == 10
    expected_score = round((c["complete"] + c["partial"] * 0.5) / 10 * 100, 1)
    assert c["score_pct"] == expected_score


# ---------------------------------------------------------------------------
# HTML rendering tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_html_output_contains_key_elements(generator):
    """HTML output should contain key document elements."""
    doc = await generator.generate()
    html = generator.render_html(doc)

    assert "<!DOCTYPE html>" in html
    assert "EU AI Act Annex IV" in html
    assert "FriendlyFace" in html
    assert "annex_iv_1" in html
    assert "annex_iv_10" in html
    assert "General Description" in html
    assert "Post-Market Monitoring" in html
    assert "enforcement_deadline" in html or "2026-08-02" in html


@pytest.mark.asyncio
async def test_html_output_has_status_badges(generator):
    """HTML should contain status badges for each section."""
    doc = await generator.generate()
    html = generator.render_html(doc)
    # At least one status badge should appear
    assert "Missing" in html or "Partial" in html or "Complete" in html


@pytest.mark.asyncio
async def test_html_output_escapes_special_chars(generator):
    """HTML renderer should escape special characters."""
    doc = await generator.generate(
        system_id="test<script>", system_name='Test "System"'
    )
    html = generator.render_html(doc)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&quot;System&quot;" in html


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_conformity_assessment_json(client):
    """POST /governance/conformity-assessment returns 201 with JSON."""
    resp = await client.post("/governance/conformity-assessment")
    assert resp.status_code == 201
    data = resp.json()
    assert data["document_type"] == "eu_ai_act_annex_iv"
    assert len(data["sections"]) == 10
    assert "completeness" in data
    assert "gaps" in data


@pytest.mark.asyncio
async def test_api_conformity_assessment_html(client):
    """POST /governance/conformity-assessment?format=html returns HTML."""
    resp = await client.post("/governance/conformity-assessment?format=html")
    assert resp.status_code == 201
    assert "text/html" in resp.headers.get("content-type", "")
    assert "EU AI Act Annex IV" in resp.text
    assert "annex_iv_1" in resp.text


@pytest.mark.asyncio
async def test_api_conformity_assessment_custom_params(client):
    """Custom system_id and system_name are respected."""
    resp = await client.post(
        "/governance/conformity-assessment?system_id=custom&system_name=CustomSystem"
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["system_id"] == "custom"
    assert data["system_name"] == "CustomSystem"


@pytest.mark.asyncio
async def test_api_v1_conformity_assessment(client):
    """POST /api/v1/governance/conformity-assessment also works."""
    resp = await client.post("/api/v1/governance/conformity-assessment")
    assert resp.status_code == 201
    data = resp.json()
    assert data["document_type"] == "eu_ai_act_annex_iv"
    assert len(data["sections"]) == 10
