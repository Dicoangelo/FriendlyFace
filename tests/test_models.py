"""Tests for core domain models."""

from friendlyface.core.models import (
    BiasAuditRecord,
    EventType,
    ForensicBundle,
    ForensicEvent,
    ProvenanceNode,
    canonical_json,
    sha256_hex,
)


def test_canonical_json_deterministic():
    """canonical_json must produce identical output regardless of key order."""
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b


def test_sha256_hex_known_value():
    """SHA-256 of empty string is a known constant."""
    assert sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestForensicEvent:
    def test_seal_and_verify(self):
        event = ForensicEvent(
            event_type=EventType.TRAINING_START,
            actor="test_agent",
            payload={"model": "resnet50"},
        ).seal()

        assert event.event_hash != ""
        assert event.verify()

    def test_tamper_detection(self):
        event = ForensicEvent(
            event_type=EventType.INFERENCE_RESULT,
            actor="test_agent",
            payload={"score": 0.95},
        ).seal()

        # Tamper with payload
        event.payload["score"] = 0.5
        assert not event.verify()

    def test_hash_chain(self):
        e1 = ForensicEvent(
            event_type=EventType.TRAINING_START,
            actor="agent",
            previous_hash="GENESIS",
            sequence_number=0,
        ).seal()

        e2 = ForensicEvent(
            event_type=EventType.TRAINING_COMPLETE,
            actor="agent",
            previous_hash=e1.event_hash,
            sequence_number=1,
        ).seal()

        assert e2.previous_hash == e1.event_hash
        assert e1.verify()
        assert e2.verify()

    def test_genesis_event(self):
        event = ForensicEvent(
            event_type=EventType.TRAINING_START,
            actor="genesis",
        ).seal()
        assert event.previous_hash == "GENESIS"


class TestProvenanceNode:
    def test_seal_and_verify(self):
        node = ProvenanceNode(
            entity_type="model",
            entity_id="resnet50-v1",
            metadata={"accuracy": 0.97},
        ).seal()

        assert node.node_hash != ""
        assert node.node_hash == node.compute_hash()

    def test_tamper_detection(self):
        node = ProvenanceNode(
            entity_type="model",
            entity_id="resnet50-v1",
        ).seal()

        node.entity_id = "tampered"
        assert node.node_hash != node.compute_hash()


class TestForensicBundle:
    def test_seal_and_verify(self):
        from uuid import uuid4

        bundle = ForensicBundle(
            event_ids=[uuid4(), uuid4()],
            merkle_root="abc123",
        ).seal()

        assert bundle.bundle_hash != ""
        assert bundle.verify()
        assert bundle.status.value == "complete"

    def test_tamper_detection(self):
        from uuid import uuid4

        bundle = ForensicBundle(
            event_ids=[uuid4()],
            merkle_root="abc123",
        ).seal()

        bundle.merkle_root = "tampered"
        assert not bundle.verify()


class TestBiasAuditRecord:
    def test_creation(self):
        audit = BiasAuditRecord(
            demographic_parity_gap=0.05,
            equalized_odds_gap=0.03,
            groups_evaluated=["group_a", "group_b"],
            compliant=True,
        )
        assert audit.compliant
        assert audit.demographic_parity_gap == 0.05
