"""Tests for bundle export/import endpoints (US-035)."""

from __future__ import annotations

from uuid import uuid4


class TestExportStructure:
    """Verify JSON-LD export structure."""

    async def test_export_structure(self, client):
        """Create events + bundle, export, verify JSON-LD structure."""
        r1 = await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test", "payload": {"k": "v1"}},
        )
        r2 = await client.post(
            "/events",
            json={"event_type": "training_complete", "actor": "test", "payload": {"k": "v2"}},
        )
        e1 = r1.json()
        e2 = r2.json()

        br = await client.post(
            "/bundles",
            json={"event_ids": [e1["id"], e2["id"]]},
        )
        assert br.status_code == 201
        bundle = br.json()

        resp = await client.get(f"/bundles/{bundle['id']}/export")
        assert resp.status_code == 200
        doc = resp.json()

        assert doc["@context"] == [
            "https://www.w3.org/2018/credentials/v1",
            "https://friendlyface.dev/forensic/v1",
        ]
        assert doc["@type"] == "ForensicBundle"
        assert doc["id"] == bundle["id"]
        assert "created_at" in doc
        assert "status" in doc
        assert "bundle_hash" in doc
        assert isinstance(doc["events"], list)
        assert isinstance(doc["merkle_proofs"], list)
        assert isinstance(doc["provenance_chain"], list)
        assert "bias_audit" in doc
        assert "recognition_artifacts" in doc
        assert "fl_artifacts" in doc
        assert "bias_report" in doc
        assert "explanation_artifacts" in doc
        assert "zk_proof" in doc
        assert "did_credential" in doc

    async def test_export_not_found(self, client):
        """404 for nonexistent bundle."""
        fake_id = str(uuid4())
        resp = await client.get(f"/bundles/{fake_id}/export")
        assert resp.status_code == 404

    async def test_export_includes_events(self, client):
        """Verify that exported events array contains full event data."""
        r1 = await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "agent_a", "payload": {"x": 1}},
        )
        e1 = r1.json()

        br = await client.post("/bundles", json={"event_ids": [e1["id"]]})
        bundle = br.json()

        resp = await client.get(f"/bundles/{bundle['id']}/export")
        doc = resp.json()

        assert len(doc["events"]) == 1
        exported_event = doc["events"][0]
        assert exported_event["id"] == e1["id"]
        assert exported_event["event_type"] == "training_start"
        assert exported_event["actor"] == "agent_a"
        assert exported_event["payload"] == {"x": 1}
        assert "event_hash" in exported_event
        assert "previous_hash" in exported_event


class TestRoundTrip:
    """Verify export -> import round trip."""

    async def test_round_trip(self, client):
        """Export a bundle, import it, verify imported=True and all checks True."""
        r1 = await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test", "payload": {"a": 1}},
        )
        r2 = await client.post(
            "/events",
            json={"event_type": "training_complete", "actor": "test", "payload": {"b": 2}},
        )
        e1 = r1.json()
        e2 = r2.json()

        br = await client.post("/bundles", json={"event_ids": [e1["id"], e2["id"]]})
        bundle = br.json()

        export_resp = await client.get(f"/bundles/{bundle['id']}/export")
        assert export_resp.status_code == 200
        doc = export_resp.json()

        import_resp = await client.post("/bundles/import", json={"document": doc})
        assert import_resp.status_code == 201
        result = import_resp.json()

        assert result["imported"] is True
        assert result["bundle_id"] is not None
        assert result["verification"]["hash_valid"] is True
        assert result["verification"]["merkle_valid"] is True
        assert result["verification"]["zk_valid"] is True
        assert result["verification"]["did_valid"] is True
        assert result["verification"]["chain_valid"] is True


class TestImportValidation:
    """Verify import validation logic."""

    async def test_import_tampered_hash(self, client):
        """Modify the bundle_hash in exported JSON, import should return 422."""
        r1 = await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test"},
        )
        e1 = r1.json()

        br = await client.post("/bundles", json={"event_ids": [e1["id"]]})
        bundle = br.json()

        export_resp = await client.get(f"/bundles/{bundle['id']}/export")
        doc = export_resp.json()

        doc["bundle_hash"] = "deadbeef" * 8

        import_resp = await client.post("/bundles/import", json={"document": doc})
        assert import_resp.status_code == 422
        detail = import_resp.json()["detail"]
        assert detail["imported"] is False
        assert detail["verification"]["hash_valid"] is False

    async def test_import_with_no_crypto(self, client):
        """Export a minimal bundle (no ZK/DID), import should succeed."""
        r1 = await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test"},
        )
        e1 = r1.json()

        br = await client.post("/bundles", json={"event_ids": [e1["id"]]})
        bundle = br.json()

        export_resp = await client.get(f"/bundles/{bundle['id']}/export")
        doc = export_resp.json()

        doc["zk_proof"] = None
        doc["did_credential"] = None

        import_resp = await client.post("/bundles/import", json={"document": doc})
        assert import_resp.status_code == 201
        result = import_resp.json()
        assert result["imported"] is True
        assert result["verification"]["zk_valid"] is True
        assert result["verification"]["did_valid"] is True
