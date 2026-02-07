"""Tests for US-044: Merkle Tree Persistence.

Tests checkpoint serialization, restoration, and incremental rebuild.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from friendlyface.core.merkle import MerkleTree
from friendlyface.core.models import EventType
from friendlyface.storage.database import Database


class TestMerkleCheckpoint:
    def test_to_checkpoint(self):
        tree = MerkleTree()
        tree.add_leaf("hash1")
        tree.add_leaf("hash2")
        event_index = {uuid4(): 0, uuid4(): 1}
        cp = tree.to_checkpoint(event_index)
        assert cp["leaf_count"] == 2
        assert cp["root_hash"] == tree.root
        assert len(cp["leaves"]) == 2
        assert len(cp["event_index"]) == 2

    def test_from_checkpoint_roundtrip(self):
        tree = MerkleTree()
        tree.add_leaf("hash1")
        tree.add_leaf("hash2")
        tree.add_leaf("hash3")
        eid1, eid2, eid3 = uuid4(), uuid4(), uuid4()
        event_index = {eid1: 0, eid2: 1, eid3: 2}

        cp = tree.to_checkpoint(event_index)
        restored_tree, restored_index = MerkleTree.from_checkpoint(cp)

        assert restored_tree.leaf_count == 3
        assert restored_tree.root == tree.root
        assert restored_index[eid1] == 0
        assert restored_index[eid2] == 1
        assert restored_index[eid3] == 2

    def test_from_checkpoint_empty(self):
        tree, index = MerkleTree.from_checkpoint({"leaves": [], "event_index": {}})
        assert tree.leaf_count == 0
        assert tree.root is None
        assert len(index) == 0

    def test_checkpoint_proofs_valid(self):
        tree = MerkleTree()
        for i in range(10):
            tree.add_leaf(f"event_{i}")
        cp = tree.to_checkpoint()
        restored, _ = MerkleTree.from_checkpoint(cp)
        # Proofs should work on restored tree
        proof = restored.get_proof(5)
        assert restored.verify_proof(proof)


class TestMerkleCheckpointDB:
    @pytest.fixture
    async def mig_db(self, tmp_path):
        db = Database(tmp_path / "merkle_test.db")
        await db.connect()
        await db.run_migrations()
        yield db
        await db.close()

    @pytest.mark.asyncio
    async def test_insert_and_get_checkpoint(self, mig_db):
        tree = MerkleTree()
        tree.add_leaf("data1")
        tree.add_leaf("data2")
        cp = tree.to_checkpoint({uuid4(): 0, uuid4(): 1})
        await mig_db.insert_merkle_checkpoint(cp)

        latest = await mig_db.get_latest_merkle_checkpoint()
        assert latest is not None
        assert latest["leaf_count"] == 2
        assert latest["root_hash"] == tree.root

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, mig_db):
        for i in range(3):
            tree = MerkleTree()
            for j in range(i + 1):
                tree.add_leaf(f"data_{i}_{j}")
            cp = tree.to_checkpoint()
            await mig_db.insert_merkle_checkpoint(cp)

        latest = await mig_db.get_latest_merkle_checkpoint()
        assert latest["leaf_count"] == 3

    @pytest.mark.asyncio
    async def test_no_checkpoint(self, mig_db):
        result = await mig_db.get_latest_merkle_checkpoint()
        assert result is None


class TestServiceCheckpointIntegration:
    """Test that ForensicService uses checkpoints correctly."""

    @pytest.fixture
    async def svc_db(self, tmp_path):
        db = Database(tmp_path / "svc_ckpt_test.db")
        await db.connect()
        await db.run_migrations()
        yield db
        await db.close()

    @pytest.mark.asyncio
    async def test_service_initialize_without_checkpoint(self, svc_db):
        from friendlyface.core.service import ForensicService

        svc = ForensicService(svc_db)
        await svc.initialize()
        assert svc.merkle.leaf_count == 0

    @pytest.mark.asyncio
    async def test_service_initialize_with_checkpoint(self, svc_db):
        from friendlyface.core.service import ForensicService

        # Create events and a checkpoint
        svc1 = ForensicService(svc_db)
        await svc1.initialize()
        await svc1.record_event(EventType.TRAINING_START, "test", {"step": 1})
        await svc1.record_event(EventType.TRAINING_COMPLETE, "test", {"step": 2})

        cp = svc1.merkle.to_checkpoint(svc1._event_index)
        await svc_db.insert_merkle_checkpoint(cp)

        # Create a new service and restore
        svc2 = ForensicService(svc_db)
        await svc2.initialize()
        assert svc2.merkle.leaf_count == 2
        assert svc2.merkle.root == svc1.merkle.root

    @pytest.mark.asyncio
    async def test_incremental_rebuild_from_checkpoint(self, svc_db):
        from friendlyface.core.service import ForensicService

        svc1 = ForensicService(svc_db)
        await svc1.initialize()
        await svc1.record_event(EventType.TRAINING_START, "test")
        await svc1.record_event(EventType.TRAINING_COMPLETE, "test")

        # Save checkpoint at 2 events
        cp = svc1.merkle.to_checkpoint(svc1._event_index)
        await svc_db.insert_merkle_checkpoint(cp)

        # Add more events
        await svc1.record_event(EventType.INFERENCE_RESULT, "test")

        # New service should restore checkpoint + replay 1 event
        svc2 = ForensicService(svc_db)
        await svc2.initialize()
        assert svc2.merkle.leaf_count == 3
        assert svc2.merkle.root == svc1.merkle.root
