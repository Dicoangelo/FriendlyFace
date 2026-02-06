"""Tests for the async SQLite storage layer."""

import pytest

from friendlyface.core.models import (
    BundleStatus,
    EventType,
    ForensicBundle,
    ForensicEvent,
    ProvenanceNode,
)
from friendlyface.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


class TestEventStorage:
    async def test_insert_and_get(self, db):
        event = ForensicEvent(
            event_type=EventType.TRAINING_START,
            actor="test",
            payload={"lr": 0.001},
            sequence_number=0,
        ).seal()

        await db.insert_event(event)
        loaded = await db.get_event(event.id)

        assert loaded is not None
        assert loaded.id == event.id
        assert loaded.event_hash == event.event_hash
        assert loaded.payload == {"lr": 0.001}

    async def test_get_latest(self, db):
        for i in range(3):
            e = ForensicEvent(
                event_type=EventType.TRAINING_START,
                actor="test",
                sequence_number=i,
            ).seal()
            await db.insert_event(e)

        latest = await db.get_latest_event()
        assert latest is not None
        assert latest.sequence_number == 2

    async def test_get_all_ordered(self, db):
        for i in range(5):
            e = ForensicEvent(
                event_type=EventType.INFERENCE_REQUEST,
                actor="test",
                sequence_number=i,
            ).seal()
            await db.insert_event(e)

        events = await db.get_all_events()
        assert len(events) == 5
        assert [e.sequence_number for e in events] == [0, 1, 2, 3, 4]

    async def test_event_count(self, db):
        assert await db.get_event_count() == 0
        e = ForensicEvent(
            event_type=EventType.BIAS_AUDIT,
            actor="test",
            sequence_number=0,
        ).seal()
        await db.insert_event(e)
        assert await db.get_event_count() == 1


class TestProvenanceStorage:
    async def test_insert_and_get(self, db):
        node = ProvenanceNode(
            entity_type="dataset",
            entity_id="faces_v1",
            metadata={"size": 10000},
        ).seal()

        await db.insert_provenance_node(node)
        loaded = await db.get_provenance_node(node.id)

        assert loaded is not None
        assert loaded.entity_type == "dataset"
        assert loaded.node_hash == node.node_hash


class TestBundleStorage:
    async def test_insert_and_get(self, db):
        from uuid import uuid4

        bundle = ForensicBundle(
            event_ids=[uuid4()],
            merkle_root="test_root",
        ).seal()

        await db.insert_bundle(bundle)
        loaded = await db.get_bundle(bundle.id)

        assert loaded is not None
        assert loaded.bundle_hash == bundle.bundle_hash
        assert loaded.status == BundleStatus.COMPLETE

    async def test_update_status(self, db):
        from uuid import uuid4

        bundle = ForensicBundle(event_ids=[uuid4()], merkle_root="r").seal()
        await db.insert_bundle(bundle)

        await db.update_bundle_status(bundle.id, BundleStatus.VERIFIED)
        loaded = await db.get_bundle(bundle.id)
        assert loaded.status == BundleStatus.VERIFIED
