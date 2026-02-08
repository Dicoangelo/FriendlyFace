"""Tests for face gallery enrollment and search (US-046)."""

from __future__ import annotations

import numpy as np
import pytest
import pytest_asyncio

from friendlyface.recognition.embeddings import EMBEDDING_DIM, FaceEmbedding
from friendlyface.recognition.gallery import FaceGallery, GalleryMatch
from friendlyface.storage.database import Database


def _make_embedding(seed: int = 0, model_version: str = "test-v1") -> FaceEmbedding:
    """Create a random unit-norm embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return FaceEmbedding(vector=vec, model_version=model_version, input_hash=f"hash-{seed}")


@pytest_asyncio.fixture
async def gallery(tmp_path):
    db = Database(tmp_path / "gallery_test.db")
    await db.connect()
    await db.run_migrations()
    g = FaceGallery(db, similarity_threshold=0.5)
    yield g
    await db.close()


class TestEnroll:
    async def test_enroll_returns_entry(self, gallery):
        emb = _make_embedding(seed=1)
        result = await gallery.enroll("alice", emb, quality_score=0.9)
        assert result["subject_id"] == "alice"
        assert result["entry_id"]
        assert result["embedding_dim"] == EMBEDDING_DIM
        assert result["quality_score"] == 0.9

    async def test_enroll_multiple_same_subject(self, gallery):
        for i in range(3):
            await gallery.enroll("bob", _make_embedding(seed=i + 10))
        count = await gallery.count()
        assert count == 3

    async def test_enroll_with_metadata(self, gallery):
        emb = _make_embedding(seed=5)
        result = await gallery.enroll("carol", emb, metadata={"camera": "front"})
        assert result["subject_id"] == "carol"


class TestSearch:
    async def test_search_finds_enrolled(self, gallery):
        emb = _make_embedding(seed=42)
        await gallery.enroll("alice", emb)
        matches = await gallery.search(emb, top_k=5)
        assert len(matches) >= 1
        assert matches[0].subject_id == "alice"
        assert matches[0].similarity == pytest.approx(1.0, abs=1e-4)

    async def test_search_empty_gallery(self, gallery):
        matches = await gallery.search(_make_embedding(seed=1))
        assert matches == []

    async def test_search_respects_threshold(self, gallery):
        await gallery.enroll("alice", _make_embedding(seed=1))
        # Different embedding should have low similarity
        query = _make_embedding(seed=999)
        matches = await gallery.search(query, top_k=5)
        # May or may not find match depending on random similarity
        for m in matches:
            assert m.similarity >= gallery.similarity_threshold

    async def test_search_top_k(self, gallery):
        for i in range(10):
            await gallery.enroll(f"subject-{i}", _make_embedding(seed=i))
        matches = await gallery.search(_make_embedding(seed=0), top_k=3)
        assert len(matches) <= 3

    async def test_search_sorted_descending(self, gallery):
        for i in range(5):
            await gallery.enroll(f"s{i}", _make_embedding(seed=i))
        matches = await gallery.search(_make_embedding(seed=0), top_k=5)
        for i in range(len(matches) - 1):
            assert matches[i].similarity >= matches[i + 1].similarity

    async def test_search_zero_query_returns_empty(self, gallery):
        await gallery.enroll("alice", _make_embedding(seed=1))
        zero_emb = FaceEmbedding(
            vector=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            model_version="v1",
            input_hash="zero",
        )
        matches = await gallery.search(zero_emb)
        assert matches == []

    async def test_gallery_match_fields(self, gallery):
        emb = _make_embedding(seed=50)
        await gallery.enroll("test-subject", emb, quality_score=0.8)
        matches = await gallery.search(emb)
        m = matches[0]
        assert isinstance(m, GalleryMatch)
        assert m.subject_id == "test-subject"
        assert m.entry_id
        assert m.model_version == "test-v1"


class TestListSubjects:
    async def test_empty_gallery(self, gallery):
        subjects = await gallery.list_subjects()
        assert subjects == []

    async def test_lists_enrolled_subjects(self, gallery):
        await gallery.enroll("alice", _make_embedding(seed=1))
        await gallery.enroll("bob", _make_embedding(seed=2))
        subjects = await gallery.list_subjects()
        ids = [s["subject_id"] for s in subjects]
        assert "alice" in ids
        assert "bob" in ids

    async def test_entry_count(self, gallery):
        await gallery.enroll("alice", _make_embedding(seed=1))
        await gallery.enroll("alice", _make_embedding(seed=2))
        subjects = await gallery.list_subjects()
        alice = [s for s in subjects if s["subject_id"] == "alice"][0]
        assert alice["entry_count"] == 2


class TestDeleteSubject:
    async def test_delete_removes_entries(self, gallery):
        await gallery.enroll("alice", _make_embedding(seed=1))
        await gallery.enroll("alice", _make_embedding(seed=2))
        deleted = await gallery.delete_subject("alice")
        assert deleted == 2
        assert await gallery.count() == 0

    async def test_delete_nonexistent_returns_zero(self, gallery):
        deleted = await gallery.delete_subject("nobody")
        assert deleted == 0


class TestCount:
    async def test_empty(self, gallery):
        assert await gallery.count() == 0

    async def test_after_enrollment(self, gallery):
        await gallery.enroll("a", _make_embedding(seed=1))
        await gallery.enroll("b", _make_embedding(seed=2))
        assert await gallery.count() == 2
