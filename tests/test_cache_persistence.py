"""Tests for write-through cache persistence (US-088 through US-091)."""

from __future__ import annotations

import pytest_asyncio

from friendlyface.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    """Fresh database for persistence tests."""
    database = Database(tmp_path / "cache_test.db")
    await database.connect()
    yield database
    await database.close()


# ---------------------------------------------------------------------------
# Explanation Records (US-088)
# ---------------------------------------------------------------------------


class TestExplanationPersistence:
    async def test_insert_and_get(self, db):
        """Insert explanation and retrieve by ID."""
        await db.insert_explanation("e1", "ev1", "lime", "2026-01-01T00:00:00Z", {"top_k": 5})
        record = await db.get_explanation("e1")
        assert record is not None
        assert record["id"] == "e1"
        assert record["event_id"] == "ev1"
        assert record["method"] == "lime"
        assert record["data"]["top_k"] == 5

    async def test_get_nonexistent(self, db):
        """Non-existent explanation returns None."""
        assert await db.get_explanation("nope") is None

    async def test_list_explanations(self, db):
        """List returns all explanations with pagination."""
        for i in range(5):
            await db.insert_explanation(
                f"e{i}", f"ev{i}", "shap", f"2026-01-0{i + 1}T00:00:00Z", {"i": i}
            )
        items, total = await db.list_explanations(limit=3, offset=0)
        assert total == 5
        assert len(items) == 3

    async def test_list_explanations_offset(self, db):
        """Pagination offset works."""
        for i in range(5):
            await db.insert_explanation(f"e{i}", f"ev{i}", "sdd", f"2026-01-0{i + 1}T00:00:00Z", {})
        items, total = await db.list_explanations(limit=10, offset=3)
        assert len(items) == 2
        assert total == 5

    async def test_upsert_explanation(self, db):
        """INSERT OR REPLACE updates existing record."""
        await db.insert_explanation("e1", "ev1", "lime", "2026-01-01T00:00:00Z", {"v": 1})
        await db.insert_explanation("e1", "ev1", "lime", "2026-01-01T00:00:00Z", {"v": 2})
        record = await db.get_explanation("e1")
        assert record["data"]["v"] == 2

    async def test_empty_list(self, db):
        """Empty table returns empty list and zero total."""
        items, total = await db.list_explanations()
        assert items == []
        assert total == 0


# ---------------------------------------------------------------------------
# Model Registry (US-089)
# ---------------------------------------------------------------------------


class TestModelPersistence:
    async def test_insert_and_get(self, db):
        """Insert model and retrieve by ID."""
        await db.insert_model("m1", "2026-01-01T00:00:00Z", {"n_components": 128})
        model = await db.get_model("m1")
        assert model is not None
        assert model["id"] == "m1"
        assert model["n_components"] == 128

    async def test_get_nonexistent(self, db):
        """Non-existent model returns None."""
        assert await db.get_model("nope") is None

    async def test_list_models(self, db):
        """List returns all models."""
        await db.insert_model("m1", "2026-01-01T00:00:00Z", {"a": 1})
        await db.insert_model("m2", "2026-01-02T00:00:00Z", {"a": 2})
        models = await db.list_models()
        assert len(models) == 2

    async def test_upsert_model(self, db):
        """INSERT OR REPLACE updates existing model."""
        await db.insert_model("m1", "2026-01-01T00:00:00Z", {"v": 1})
        await db.insert_model("m1", "2026-01-01T00:00:00Z", {"v": 2})
        model = await db.get_model("m1")
        assert model["v"] == 2

    async def test_empty_list(self, db):
        """Empty table returns empty list."""
        assert await db.list_models() == []


# ---------------------------------------------------------------------------
# FL Simulations (US-090)
# ---------------------------------------------------------------------------


class TestFLSimulationPersistence:
    async def test_insert_and_get(self, db):
        """Insert FL simulation and retrieve by ID."""
        await db.insert_fl_simulation("s1", "2026-01-01T00:00:00Z", {"n_rounds": 10})
        sim = await db.get_fl_simulation("s1")
        assert sim is not None
        assert sim["id"] == "s1"
        assert sim["n_rounds"] == 10

    async def test_get_nonexistent(self, db):
        """Non-existent simulation returns None."""
        assert await db.get_fl_simulation("nope") is None

    async def test_list_simulations(self, db):
        """List returns all simulations."""
        await db.insert_fl_simulation("s1", "2026-01-01T00:00:00Z", {"n": 1})
        await db.insert_fl_simulation("s2", "2026-01-02T00:00:00Z", {"n": 2})
        sims = await db.list_fl_simulations()
        assert len(sims) == 2

    async def test_upsert_simulation(self, db):
        """INSERT OR REPLACE updates existing simulation."""
        await db.insert_fl_simulation("s1", "2026-01-01T00:00:00Z", {"v": 1})
        await db.insert_fl_simulation("s1", "2026-01-01T00:00:00Z", {"v": 2})
        sim = await db.get_fl_simulation("s1")
        assert sim["v"] == 2

    async def test_empty_list(self, db):
        """Empty table returns empty list."""
        assert await db.list_fl_simulations() == []


# ---------------------------------------------------------------------------
# Compliance Reports (US-091)
# ---------------------------------------------------------------------------


class TestCompliancePersistence:
    async def test_insert_and_get_latest(self, db):
        """Insert compliance report and retrieve latest."""
        await db.insert_compliance_report("r1", "2026-01-01T00:00:00Z", {"status": "ok"})
        report = await db.get_latest_compliance_report()
        assert report is not None
        assert report["id"] == "r1"
        assert report["status"] == "ok"

    async def test_latest_returns_most_recent(self, db):
        """Latest returns the most recently created report."""
        await db.insert_compliance_report("r1", "2026-01-01T00:00:00Z", {"v": 1})
        await db.insert_compliance_report("r2", "2026-01-02T00:00:00Z", {"v": 2})
        report = await db.get_latest_compliance_report()
        assert report["id"] == "r2"
        assert report["v"] == 2

    async def test_get_latest_empty(self, db):
        """No reports returns None."""
        assert await db.get_latest_compliance_report() is None

    async def test_list_reports(self, db):
        """List returns all reports."""
        await db.insert_compliance_report("r1", "2026-01-01T00:00:00Z", {"a": 1})
        await db.insert_compliance_report("r2", "2026-01-02T00:00:00Z", {"a": 2})
        reports = await db.list_compliance_reports()
        assert len(reports) == 2

    async def test_upsert_report(self, db):
        """INSERT OR REPLACE updates existing report."""
        await db.insert_compliance_report("r1", "2026-01-01T00:00:00Z", {"v": 1})
        await db.insert_compliance_report("r1", "2026-01-01T00:00:00Z", {"v": 2})
        report = await db.get_latest_compliance_report()
        assert report["v"] == 2

    async def test_empty_list(self, db):
        """Empty table returns empty list."""
        assert await db.list_compliance_reports() == []


# ---------------------------------------------------------------------------
# API write-through integration
# ---------------------------------------------------------------------------


class TestWriteThroughAPI:
    async def test_explanation_persisted_via_api(self, client):
        """LIME explanation is persisted to DB via API."""
        import friendlyface.api.app as app_module

        # Create an inference event first
        event_resp = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        event_id = event_resp.json()["id"]

        # Trigger LIME explanation
        resp = await client.post(
            "/explainability/lime",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        explanation_id = resp.json()["explanation_id"]

        # Verify in memory
        assert explanation_id in app_module._explanations

        # Verify in DB
        from friendlyface.api.app import _db

        record = await _db.get_explanation(explanation_id)
        assert record is not None
        assert record["method"] == "lime"

    async def test_shap_explanation_persisted(self, client):
        """SHAP explanation is persisted to DB via API."""
        event_resp = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        event_id = event_resp.json()["id"]

        resp = await client.post(
            "/explainability/shap",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        explanation_id = resp.json()["explanation_id"]

        from friendlyface.api.app import _db

        record = await _db.get_explanation(explanation_id)
        assert record is not None
        assert record["method"] == "shap"

    async def test_sdd_explanation_persisted(self, client):
        """SDD explanation is persisted to DB via API."""
        event_resp = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        event_id = event_resp.json()["id"]

        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        explanation_id = resp.json()["explanation_id"]

        from friendlyface.api.app import _db

        record = await _db.get_explanation(explanation_id)
        assert record is not None
        assert record["method"] == "sdd"

    async def test_model_persisted_via_train(self, client, tmp_path):
        """Model registry entry is persisted to DB after training."""
        import numpy as np
        from PIL import Image as PILImage

        from friendlyface.recognition.pca import IMAGE_SIZE

        rng = np.random.default_rng(42)
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        labels = []
        for i in range(12):
            pixels = rng.integers(0, 256, size=IMAGE_SIZE, dtype=np.uint8)
            img = PILImage.fromarray(pixels, mode="L")
            img.save(dataset_dir / f"face_{i:04d}.png")
            labels.append(i % 3)

        output_dir = tmp_path / "models"

        resp = await client.post(
            "/recognition/train",
            json={
                "dataset_path": str(dataset_dir),
                "output_dir": str(output_dir),
                "n_components": 5,
                "labels": labels,
            },
        )
        assert resp.status_code == 201
        model_id = resp.json()["model_id"]

        from friendlyface.api.app import _db

        model = await _db.get_model(model_id)
        assert model is not None
        assert model["id"] == model_id
