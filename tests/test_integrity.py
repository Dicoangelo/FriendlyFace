"""Integrity and tamper detection tests.

Verification requirements from the plan:
1. Hash chain integrity across 100+ events
2. Tamper detection: modify any field → verification fails
3. Merkle proof verification for all events
"""

import pytest

from friendlyface.core.merkle import MerkleTree
from friendlyface.core.models import EventType, ForensicEvent


class TestHashChainIntegrity:
    """Test hash chain integrity across 100+ events."""

    def test_chain_100_events(self):
        events: list[ForensicEvent] = []
        prev_hash = "GENESIS"

        for i in range(120):
            event = ForensicEvent(
                event_type=EventType.INFERENCE_REQUEST,
                actor=f"agent_{i}",
                payload={"request_id": i},
                previous_hash=prev_hash,
                sequence_number=i,
            ).seal()
            events.append(event)
            prev_hash = event.event_hash

        # All events should verify
        for event in events:
            assert event.verify(), f"Event {event.sequence_number} failed verification"

        # Chain links should be intact
        for i in range(1, len(events)):
            assert events[i].previous_hash == events[i - 1].event_hash, (
                f"Chain broken at event {i}"
            )

    def test_tamper_any_field_detected(self):
        """Modify any single field in the chain → verification fails."""
        events: list[ForensicEvent] = []
        prev_hash = "GENESIS"

        for i in range(10):
            event = ForensicEvent(
                event_type=EventType.TRAINING_START,
                actor="agent",
                payload={"step": i},
                previous_hash=prev_hash,
                sequence_number=i,
            ).seal()
            events.append(event)
            prev_hash = event.event_hash

        # Tamper with event 5's actor
        events[5].actor = "tampered_agent"
        assert not events[5].verify()

        # Tamper with event 3's payload
        events[3].payload["injected"] = True
        assert not events[3].verify()

        # Tamper with event 7's timestamp
        from datetime import datetime, timezone

        events[7].timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
        assert not events[7].verify()

        # Tamper with event 0's previous_hash
        events[0].previous_hash = "fake_genesis"
        assert not events[0].verify()

    def test_chain_break_detected(self):
        """If someone replaces an event mid-chain, the break is detectable."""
        events: list[ForensicEvent] = []
        prev_hash = "GENESIS"

        for i in range(5):
            event = ForensicEvent(
                event_type=EventType.INFERENCE_RESULT,
                actor="agent",
                payload={"score": 0.9 + i * 0.01},
                previous_hash=prev_hash,
                sequence_number=i,
            ).seal()
            events.append(event)
            prev_hash = event.event_hash

        # Replace event 2 with a forged event
        forged = ForensicEvent(
            event_type=EventType.INFERENCE_RESULT,
            actor="forger",
            payload={"score": 1.0},
            previous_hash=events[1].event_hash,
            sequence_number=2,
        ).seal()
        events[2] = forged

        # Event 3 should now have a broken chain (its previous_hash doesn't match)
        assert events[3].previous_hash != events[2].event_hash


class TestMerkleIntegrity:
    def test_proof_all_100_events(self):
        """Verify Merkle proofs for all 100+ events."""
        tree = MerkleTree()
        for i in range(110):
            tree.add_leaf(f"event_hash_{i}")

        for i in range(110):
            proof = tree.get_proof(i)
            assert proof.verify(), f"Merkle proof failed for leaf {i}"
            assert tree.verify_proof(proof), f"Tree verification failed for leaf {i}"

    def test_invalid_proof_rejected(self):
        """A proof generated from a different tree state should fail."""
        tree = MerkleTree()
        tree.add_leaf("a")
        tree.add_leaf("b")
        proof = tree.get_proof(0)

        # Add more leaves (changes root)
        tree.add_leaf("c")

        # Old proof's root no longer matches
        assert not tree.verify_proof(proof)

    def test_tampered_leaf_proof_fails(self):
        """Modifying the leaf hash should invalidate the proof."""
        tree = MerkleTree()
        tree.add_leaf("original")
        proof = tree.get_proof(0)

        # Tamper with the proof's leaf hash
        proof.leaf_hash = "tampered_hash"
        assert not proof.verify()
