"""Tests for gallery API endpoints (US-085)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


def _make_face_bytes(seed: int = 42, size: tuple[int, int] = (200, 200)) -> bytes:
    """Create a synthetic face image as PNG bytes."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(60, 200, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gallery_enroll_success(client):
    """Enroll a face and get back entry metadata."""
    face_bytes = _make_face_bytes(seed=1)
    resp = await client.post(
        "/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "alice"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["subject_id"] == "alice"
    assert "entry_id" in data
    assert data["embedding_dim"] == 512
    assert "model_version" in data


@pytest.mark.asyncio
async def test_gallery_enroll_with_quality(client):
    """Explicit quality_score is stored."""
    face_bytes = _make_face_bytes(seed=2)
    resp = await client.post(
        "/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "bob", "quality_score": 0.95},
    )
    assert resp.status_code == 201
    assert resp.json()["quality_score"] == 0.95


@pytest.mark.asyncio
async def test_gallery_enroll_empty_image(client):
    """Empty image returns 400."""
    resp = await client.post(
        "/gallery/enroll",
        files={"image": ("empty.png", b"", "image/png")},
        params={"subject_id": "nobody"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_gallery_enroll_creates_forensic_event(client):
    """Enrollment records a forensic event."""
    face_bytes = _make_face_bytes(seed=3)
    await client.post(
        "/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "charlie"},
    )
    events_resp = await client.get("/events")
    assert events_resp.status_code == 200
    events = events_resp.json()["items"]
    enroll_events = [e for e in events if e.get("payload", {}).get("action") == "gallery_enroll"]
    assert len(enroll_events) >= 1


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gallery_search_empty(client):
    """Search on empty gallery returns no matches."""
    face_bytes = _make_face_bytes(seed=10)
    resp = await client.post(
        "/gallery/search",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["matches"] == []
    assert data["total_matches"] == 0


@pytest.mark.asyncio
async def test_gallery_search_finds_enrolled(client):
    """Search finds a previously enrolled face."""
    face_bytes = _make_face_bytes(seed=20)

    # Enroll
    enroll_resp = await client.post(
        "/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "dana"},
    )
    assert enroll_resp.status_code == 201

    # Search with same face
    search_resp = await client.post(
        "/gallery/search",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"top_k": 5},
    )
    assert search_resp.status_code == 200
    data = search_resp.json()
    assert data["total_matches"] >= 1
    assert data["matches"][0]["subject_id"] == "dana"
    assert data["matches"][0]["similarity"] > 0.9


@pytest.mark.asyncio
async def test_gallery_search_top_k(client):
    """top_k limits results."""
    face_bytes = _make_face_bytes(seed=30)

    # Enroll same face 5 times with different subjects
    for i in range(5):
        await client.post(
            "/gallery/enroll",
            files={"image": ("face.png", face_bytes, "image/png")},
            params={"subject_id": f"user_{i}"},
        )

    resp = await client.post(
        "/gallery/search",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"top_k": 2},
    )
    assert resp.status_code == 200
    assert len(resp.json()["matches"]) <= 2


# ---------------------------------------------------------------------------
# List subjects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gallery_list_subjects_empty(client):
    """Empty gallery returns empty subject list."""
    resp = await client.get("/gallery/subjects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["subjects"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_gallery_list_subjects_after_enroll(client):
    """Subjects appear after enrollment."""
    face_bytes = _make_face_bytes(seed=40)
    await client.post(
        "/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "eve"},
    )
    resp = await client.get("/gallery/subjects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    subject_ids = [s["subject_id"] for s in data["subjects"]]
    assert "eve" in subject_ids


# ---------------------------------------------------------------------------
# Delete subject
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gallery_delete_subject(client):
    """Delete removes all entries for a subject."""
    face_bytes = _make_face_bytes(seed=50)
    await client.post(
        "/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "frank"},
    )

    resp = await client.delete("/gallery/subjects/frank")
    assert resp.status_code == 200
    assert resp.json()["entries_deleted"] >= 1

    # Verify gone
    subjects_resp = await client.get("/gallery/subjects")
    subject_ids = [s["subject_id"] for s in subjects_resp.json()["subjects"]]
    assert "frank" not in subject_ids


@pytest.mark.asyncio
async def test_gallery_delete_nonexistent(client):
    """Deleting a non-existent subject returns 404."""
    resp = await client.delete("/gallery/subjects/nonexistent_person")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gallery_count_empty(client):
    """Empty gallery returns count 0."""
    resp = await client.get("/gallery/count")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


@pytest.mark.asyncio
async def test_gallery_count_after_enroll(client):
    """Count increases after enrollment."""
    face_bytes = _make_face_bytes(seed=60)
    await client.post(
        "/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "grace"},
    )
    resp = await client.get("/gallery/count")
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


# ---------------------------------------------------------------------------
# v1 prefix routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gallery_v1_enroll(client):
    """Gallery enroll works via /api/v1/ prefix."""
    face_bytes = _make_face_bytes(seed=70)
    resp = await client.post(
        "/api/v1/gallery/enroll",
        files={"image": ("face.png", face_bytes, "image/png")},
        params={"subject_id": "v1_test"},
    )
    assert resp.status_code == 201


@pytest.mark.asyncio
async def test_gallery_v1_count(client):
    """Gallery count works via /api/v1/ prefix."""
    resp = await client.get("/api/v1/gallery/count")
    assert resp.status_code == 200
    assert "total" in resp.json()
