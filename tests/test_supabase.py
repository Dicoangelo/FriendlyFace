"""Tests for the Supabase storage adapter using mocked Supabase client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from friendlyface.core.models import (
    BiasAuditRecord,
    BundleStatus,
    EventType,
    ForensicBundle,
    ForensicEvent,
    ProvenanceNode,
    ProvenanceRelation,
)
from friendlyface.storage.supabase_db import SupabaseDatabase


# ---------------------------------------------------------------------------
# Helpers: build a mock Supabase async client
# ---------------------------------------------------------------------------


class MockResponse:
    """Mimics a Supabase API response."""

    def __init__(self, data=None, count=None):
        self.data = data or []
        self.count = count


def _make_mock_table():
    """Create a mock table builder that supports chaining."""
    table = MagicMock()

    # .insert(data).execute() -> coroutine
    insert_chain = MagicMock()
    insert_chain.execute = AsyncMock(return_value=MockResponse())
    table.insert = MagicMock(return_value=insert_chain)

    # .select(...).eq(...).execute() -> coroutine
    select_chain = MagicMock()
    select_chain.eq = MagicMock(return_value=select_chain)
    select_chain.order = MagicMock(return_value=select_chain)
    select_chain.limit = MagicMock(return_value=select_chain)
    select_chain.execute = AsyncMock(return_value=MockResponse())
    table.select = MagicMock(return_value=select_chain)

    # .update(data).eq(...).execute()
    update_chain = MagicMock()
    update_chain.eq = MagicMock(return_value=update_chain)
    update_chain.execute = AsyncMock(return_value=MockResponse())
    table.update = MagicMock(return_value=update_chain)

    return table


def _make_mock_client():
    """Create a mock Supabase async client with per-table mocks."""
    client = MagicMock()
    tables: dict[str, MagicMock] = {}

    def get_table(name):
        if name not in tables:
            tables[name] = _make_mock_table()
        return tables[name]

    client.table = MagicMock(side_effect=get_table)
    client._tables = tables  # expose for test assertions
    return client


@pytest.fixture
def mock_client():
    return _make_mock_client()


@pytest.fixture
def db(mock_client):
    sdb = SupabaseDatabase(url="https://test.supabase.co", key="test-key")
    sdb._client = mock_client
    return sdb


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnection:
    async def test_connect_creates_client(self):
        sdb = SupabaseDatabase(url="https://x.supabase.co", key="k")
        with patch(
            "friendlyface.storage.supabase_db.SupabaseDatabase.connect",
            new_callable=AsyncMock,
        ) as mock_connect:
            await sdb.connect()
            mock_connect.assert_called_once()

    async def test_close_clears_client(self, db):
        assert db._client is not None
        await db.close()
        assert db._client is None

    def test_client_property_raises_when_not_connected(self):
        sdb = SupabaseDatabase()
        with pytest.raises(RuntimeError, match="not connected"):
            _ = sdb.client

    def test_client_property_returns_client_when_connected(self, db):
        assert db.client is not None


# ---------------------------------------------------------------------------
# ForensicEvent CRUD
# ---------------------------------------------------------------------------


class TestEventStorage:
    async def test_insert_event_calls_table_insert(self, db, mock_client):
        event = ForensicEvent(
            event_type=EventType.TRAINING_START,
            actor="test",
            payload={"lr": 0.001},
            sequence_number=0,
        ).seal()

        await db.insert_event(event)

        mock_client.table.assert_called_with("forensic_events")
        tbl = mock_client._tables["forensic_events"]
        tbl.insert.assert_called_once()
        args = tbl.insert.call_args[0][0]
        assert args["id"] == str(event.id)
        assert args["event_type"] == "training_start"
        assert args["actor"] == "test"

    async def test_get_event_returns_none_when_missing(self, db, mock_client):
        result = await db.get_event(uuid4())
        assert result is None

    async def test_get_event_returns_event_from_row(self, db, mock_client):
        eid = uuid4()
        event = ForensicEvent(
            id=eid,
            event_type=EventType.INFERENCE_RESULT,
            actor="tester",
            payload={"score": 0.95},
            sequence_number=3,
        ).seal()

        row = {
            "id": str(eid),
            "event_type": "inference_result",
            "timestamp": event.timestamp.isoformat(),
            "actor": "tester",
            "payload": '{"score": 0.95}',
            "previous_hash": event.previous_hash,
            "event_hash": event.event_hash,
            "sequence_number": 3,
        }

        tbl = mock_client._tables.setdefault("forensic_events", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.eq.return_value = select_chain
        select_chain.execute = AsyncMock(return_value=MockResponse(data=[row]))

        loaded = await db.get_event(eid)
        assert loaded is not None
        assert loaded.id == eid
        assert loaded.event_hash == event.event_hash
        assert loaded.payload == {"score": 0.95}

    async def test_get_latest_event_empty(self, db):
        result = await db.get_latest_event()
        assert result is None

    async def test_get_latest_event_returns_latest(self, db, mock_client):
        event = ForensicEvent(
            event_type=EventType.TRAINING_START,
            actor="t",
            sequence_number=5,
        ).seal()

        row = {
            "id": str(event.id),
            "event_type": "training_start",
            "timestamp": event.timestamp.isoformat(),
            "actor": "t",
            "payload": "{}",
            "previous_hash": "GENESIS",
            "event_hash": event.event_hash,
            "sequence_number": 5,
        }

        tbl = mock_client._tables.setdefault("forensic_events", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.order.return_value = select_chain
        select_chain.limit.return_value = select_chain
        select_chain.execute = AsyncMock(return_value=MockResponse(data=[row]))

        latest = await db.get_latest_event()
        assert latest is not None
        assert latest.sequence_number == 5

    async def test_get_all_events(self, db, mock_client):
        rows = []
        for i in range(3):
            e = ForensicEvent(
                event_type=EventType.BIAS_AUDIT,
                actor="t",
                sequence_number=i,
            ).seal()
            rows.append(
                {
                    "id": str(e.id),
                    "event_type": "bias_audit",
                    "timestamp": e.timestamp.isoformat(),
                    "actor": "t",
                    "payload": "{}",
                    "previous_hash": "GENESIS",
                    "event_hash": e.event_hash,
                    "sequence_number": i,
                }
            )

        tbl = mock_client._tables.setdefault("forensic_events", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.order.return_value = select_chain
        select_chain.execute = AsyncMock(return_value=MockResponse(data=rows))

        events = await db.get_all_events()
        assert len(events) == 3
        assert [e.sequence_number for e in events] == [0, 1, 2]

    async def test_get_event_count(self, db, mock_client):
        tbl = mock_client._tables.setdefault("forensic_events", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.execute = AsyncMock(return_value=MockResponse(data=[], count=42))

        count = await db.get_event_count()
        assert count == 42

    async def test_get_event_count_none_returns_zero(self, db, mock_client):
        tbl = mock_client._tables.setdefault("forensic_events", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.execute = AsyncMock(return_value=MockResponse(data=[], count=None))

        count = await db.get_event_count()
        assert count == 0


# ---------------------------------------------------------------------------
# ProvenanceNode CRUD
# ---------------------------------------------------------------------------


class TestProvenanceStorage:
    async def test_insert_provenance_node(self, db, mock_client):
        node = ProvenanceNode(
            entity_type="dataset",
            entity_id="faces_v1",
            metadata={"size": 10000},
        ).seal()

        await db.insert_provenance_node(node)

        mock_client.table.assert_called_with("provenance_nodes")
        tbl = mock_client._tables["provenance_nodes"]
        tbl.insert.assert_called_once()

    async def test_get_provenance_node_missing(self, db):
        result = await db.get_provenance_node(uuid4())
        assert result is None

    async def test_get_provenance_node_returns_node(self, db, mock_client):
        nid = uuid4()
        node = ProvenanceNode(
            id=nid,
            entity_type="model",
            entity_id="svm_v1",
            metadata={"accuracy": 0.99},
            parents=[],
            relations=[],
        ).seal()

        row = {
            "id": str(nid),
            "entity_type": "model",
            "entity_id": "svm_v1",
            "created_at": node.created_at.isoformat(),
            "metadata": '{"accuracy": 0.99}',
            "parents": "[]",
            "relations": "[]",
            "node_hash": node.node_hash,
        }

        tbl = mock_client._tables.setdefault("provenance_nodes", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.eq.return_value = select_chain
        select_chain.execute = AsyncMock(return_value=MockResponse(data=[row]))

        loaded = await db.get_provenance_node(nid)
        assert loaded is not None
        assert loaded.entity_type == "model"
        assert loaded.node_hash == node.node_hash
        assert loaded.metadata == {"accuracy": 0.99}

    async def test_provenance_with_parents_and_relations(self, db, mock_client):
        parent_id = uuid4()
        nid = uuid4()
        node = ProvenanceNode(
            id=nid,
            entity_type="inference",
            entity_id="inf_1",
            parents=[parent_id],
            relations=[ProvenanceRelation.DERIVED_FROM],
        ).seal()

        row = {
            "id": str(nid),
            "entity_type": "inference",
            "entity_id": "inf_1",
            "created_at": node.created_at.isoformat(),
            "metadata": "{}",
            "parents": f'["{parent_id}"]',
            "relations": '["derived_from"]',
            "node_hash": node.node_hash,
        }

        tbl = mock_client._tables.setdefault("provenance_nodes", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.eq.return_value = select_chain
        select_chain.execute = AsyncMock(return_value=MockResponse(data=[row]))

        loaded = await db.get_provenance_node(nid)
        assert loaded.parents == [parent_id]
        assert loaded.relations == [ProvenanceRelation.DERIVED_FROM]


# ---------------------------------------------------------------------------
# ForensicBundle CRUD
# ---------------------------------------------------------------------------


class TestBundleStorage:
    async def test_insert_bundle(self, db, mock_client):
        bundle = ForensicBundle(
            event_ids=[uuid4()],
            merkle_root="test_root",
        ).seal()

        await db.insert_bundle(bundle)

        mock_client.table.assert_called_with("forensic_bundles")
        tbl = mock_client._tables["forensic_bundles"]
        tbl.insert.assert_called_once()

    async def test_get_bundle_missing(self, db):
        result = await db.get_bundle(uuid4())
        assert result is None

    async def test_get_bundle_returns_bundle(self, db, mock_client):
        bid = uuid4()
        eid = uuid4()
        bundle = ForensicBundle(
            id=bid,
            event_ids=[eid],
            merkle_root="root123",
        ).seal()

        row = {
            "id": str(bid),
            "created_at": bundle.created_at.isoformat(),
            "status": "complete",
            "event_ids": f'["{eid}"]',
            "merkle_root": "root123",
            "merkle_proofs": "[]",
            "provenance_chain": "[]",
            "bias_audit": None,
            "zk_proof_placeholder": None,
            "did_credential_placeholder": None,
            "bundle_hash": bundle.bundle_hash,
        }

        tbl = mock_client._tables.setdefault("forensic_bundles", _make_mock_table())
        select_chain = tbl.select.return_value
        select_chain.eq.return_value = select_chain
        select_chain.execute = AsyncMock(return_value=MockResponse(data=[row]))

        loaded = await db.get_bundle(bid)
        assert loaded is not None
        assert loaded.bundle_hash == bundle.bundle_hash
        assert loaded.status == BundleStatus.COMPLETE
        assert loaded.event_ids == [eid]

    async def test_update_bundle_status(self, db, mock_client):
        bid = uuid4()
        await db.update_bundle_status(bid, BundleStatus.VERIFIED)

        mock_client.table.assert_called_with("forensic_bundles")
        tbl = mock_client._tables["forensic_bundles"]
        tbl.update.assert_called_once_with({"status": "verified"})


# ---------------------------------------------------------------------------
# BiasAudit CRUD
# ---------------------------------------------------------------------------


class TestBiasAuditStorage:
    async def test_insert_bias_audit(self, db, mock_client):
        audit = BiasAuditRecord(
            demographic_parity_gap=0.05,
            equalized_odds_gap=0.03,
            groups_evaluated=["age", "gender"],
            compliant=True,
            details={"note": "pass"},
        )

        await db.insert_bias_audit(audit)

        mock_client.table.assert_called_with("bias_audits")
        tbl = mock_client._tables["bias_audits"]
        tbl.insert.assert_called_once()
        args = tbl.insert.call_args[0][0]
        assert args["demographic_parity_gap"] == 0.05
        assert args["compliant"] is True

    async def test_insert_bias_audit_with_event_id(self, db, mock_client):
        eid = uuid4()
        audit = BiasAuditRecord(
            event_id=eid,
            demographic_parity_gap=0.1,
            equalized_odds_gap=0.08,
        )

        await db.insert_bias_audit(audit)

        tbl = mock_client._tables["bias_audits"]
        args = tbl.insert.call_args[0][0]
        assert args["event_id"] == str(eid)


# ---------------------------------------------------------------------------
# Backend selection via environment variable
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_default_is_sqlite(self):
        import os

        from friendlyface.api.app import _create_database

        os.environ.pop("FF_STORAGE", None)
        db = _create_database()
        from friendlyface.storage.database import Database

        assert isinstance(db, Database)

    def test_sqlite_explicit(self):
        import os

        from friendlyface.api.app import _create_database

        os.environ["FF_STORAGE"] = "sqlite"
        try:
            db = _create_database()
            from friendlyface.storage.database import Database

            assert isinstance(db, Database)
        finally:
            os.environ.pop("FF_STORAGE", None)

    def test_supabase_backend(self):
        import os

        from friendlyface.api.app import _create_database

        os.environ["FF_STORAGE"] = "supabase"
        os.environ["SUPABASE_URL"] = "https://test.supabase.co"
        os.environ["SUPABASE_KEY"] = "test-key"
        try:
            db = _create_database()
            assert isinstance(db, SupabaseDatabase)
            assert db.url == "https://test.supabase.co"
            assert db.key == "test-key"
        finally:
            os.environ.pop("FF_STORAGE", None)
            os.environ.pop("SUPABASE_URL", None)
            os.environ.pop("SUPABASE_KEY", None)

    def test_supabase_case_insensitive(self):
        import os

        from friendlyface.api.app import _create_database

        os.environ["FF_STORAGE"] = "Supabase"
        try:
            db = _create_database()
            assert isinstance(db, SupabaseDatabase)
        finally:
            os.environ.pop("FF_STORAGE", None)


# ---------------------------------------------------------------------------
# Row deserialization edge cases
# ---------------------------------------------------------------------------


class TestRowDeserialization:
    def test_event_payload_as_dict(self, db):
        """Supabase may return JSONB as dict instead of string."""
        row = {
            "id": str(uuid4()),
            "event_type": "training_start",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "actor": "test",
            "payload": {"key": "value"},
            "previous_hash": "GENESIS",
            "event_hash": "abc123",
            "sequence_number": 0,
        }
        event = db._row_to_event(row)
        assert event.payload == {"key": "value"}

    def test_event_payload_as_string(self, db):
        row = {
            "id": str(uuid4()),
            "event_type": "training_start",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "actor": "test",
            "payload": '{"key": "value"}',
            "previous_hash": "GENESIS",
            "event_hash": "abc123",
            "sequence_number": 0,
        }
        event = db._row_to_event(row)
        assert event.payload == {"key": "value"}

    def test_provenance_metadata_as_dict(self, db):
        row = {
            "id": str(uuid4()),
            "entity_type": "dataset",
            "entity_id": "d1",
            "created_at": "2025-01-01T00:00:00+00:00",
            "metadata": {"size": 100},
            "parents": [],
            "relations": [],
            "node_hash": "h",
        }
        node = db._row_to_provenance(row)
        assert node.metadata == {"size": 100}

    def test_bundle_event_ids_as_list(self, db):
        eid = uuid4()
        row = {
            "id": str(uuid4()),
            "created_at": "2025-01-01T00:00:00+00:00",
            "status": "pending",
            "event_ids": [str(eid)],
            "merkle_root": "",
            "merkle_proofs": [],
            "provenance_chain": [],
            "bias_audit": None,
            "zk_proof_placeholder": None,
            "did_credential_placeholder": None,
            "bundle_hash": "",
        }
        bundle = db._row_to_bundle(row)
        assert bundle.event_ids == [eid]
