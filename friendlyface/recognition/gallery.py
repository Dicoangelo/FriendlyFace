"""Persistent face gallery for embedding-based recognition (US-046).

Stores enrolled face embeddings in the database and provides
nearest-neighbor search for 1:N identification via cosine similarity.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from friendlyface.recognition.embeddings import FaceEmbedding
from friendlyface.storage.database import Database

logger = logging.getLogger("friendlyface.recognition.gallery")


@dataclass
class GalleryMatch:
    """A match result from gallery search."""

    subject_id: str
    entry_id: str
    similarity: float
    model_version: str


class FaceGallery:
    """Database-backed face embedding gallery.

    Supports enrollment (insert), search (1:N cosine similarity),
    and deletion of gallery entries.
    """

    def __init__(self, db: Database, similarity_threshold: float = 0.5) -> None:
        self.db = db
        self.similarity_threshold = similarity_threshold

    async def enroll(
        self,
        subject_id: str,
        embedding: FaceEmbedding,
        quality_score: float | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Enroll a face embedding for a subject.

        Returns the gallery entry record.
        """
        entry_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        await self.db.db.execute(
            "INSERT INTO face_gallery "
            "(id, subject_id, embedding, embedding_dim, model_version, "
            "quality_score, created_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry_id,
                subject_id,
                embedding.vector.tobytes(),
                embedding.dim,
                embedding.model_version,
                quality_score,
                now,
                json.dumps(metadata or {}),
            ),
        )
        await self.db.db.commit()

        logger.info("Enrolled face for subject %s (entry %s)", subject_id, entry_id)
        return {
            "entry_id": entry_id,
            "subject_id": subject_id,
            "embedding_dim": embedding.dim,
            "model_version": embedding.model_version,
            "quality_score": quality_score,
            "created_at": now,
        }

    async def search(
        self,
        query: FaceEmbedding,
        top_k: int = 5,
    ) -> list[GalleryMatch]:
        """Search the gallery for nearest matches to a query embedding.

        Uses cosine similarity. Returns up to ``top_k`` matches above
        the similarity threshold, sorted by similarity descending.
        """
        cursor = await self.db.db.execute(
            "SELECT id, subject_id, embedding, embedding_dim, model_version "
            "FROM face_gallery ORDER BY created_at ASC"
        )
        rows = await cursor.fetchall()

        if not rows:
            return []

        q = query.vector.astype(np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        matches: list[GalleryMatch] = []
        for row in rows:
            stored_vec = np.frombuffer(row[2], dtype=np.float32).astype(np.float64)
            if len(stored_vec) != len(q):
                continue
            s_norm = np.linalg.norm(stored_vec)
            if s_norm == 0:
                continue
            sim = float(np.dot(q, stored_vec) / (q_norm * s_norm))

            if sim >= self.similarity_threshold:
                matches.append(
                    GalleryMatch(
                        subject_id=row[1],
                        entry_id=row[0],
                        similarity=round(sim, 6),
                        model_version=row[4],
                    )
                )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]

    async def list_subjects(self) -> list[dict]:
        """List all enrolled subjects with entry counts."""
        cursor = await self.db.db.execute(
            "SELECT subject_id, COUNT(*) as cnt, MAX(created_at) as latest "
            "FROM face_gallery GROUP BY subject_id ORDER BY latest DESC"
        )
        rows = await cursor.fetchall()
        return [{"subject_id": r[0], "entry_count": r[1], "latest_enrollment": r[2]} for r in rows]

    async def delete_subject(self, subject_id: str) -> int:
        """Delete all gallery entries for a subject. Returns count deleted."""
        cursor = await self.db.db.execute(
            "DELETE FROM face_gallery WHERE subject_id = ?", (subject_id,)
        )
        await self.db.db.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("Deleted %d gallery entries for subject %s", deleted, subject_id)
        return deleted

    async def count(self) -> int:
        """Return total number of gallery entries."""
        cursor = await self.db.db.execute("SELECT COUNT(*) FROM face_gallery")
        row = await cursor.fetchone()
        return row[0] if row else 0
