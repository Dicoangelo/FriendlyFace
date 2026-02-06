"""Integration tests for the FastAPI endpoints."""


class TestHealth:
    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestEvents:
    async def test_record_event(self, client):
        resp = await client.post("/events", json={
            "event_type": "training_start",
            "actor": "test_agent",
            "payload": {"model": "resnet50"},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["event_type"] == "training_start"
        assert data["event_hash"] != ""
        assert data["previous_hash"] == "GENESIS"
        assert data["sequence_number"] == 0

    async def test_hash_chain(self, client):
        r1 = await client.post("/events", json={
            "event_type": "training_start",
            "actor": "agent",
        })
        r2 = await client.post("/events", json={
            "event_type": "training_complete",
            "actor": "agent",
        })

        e1 = r1.json()
        e2 = r2.json()
        assert e2["previous_hash"] == e1["event_hash"]
        assert e2["sequence_number"] == 1

    async def test_list_events(self, client):
        await client.post("/events", json={
            "event_type": "inference_request",
            "actor": "agent",
        })
        resp = await client.get("/events")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    async def test_get_event_by_id(self, client):
        r = await client.post("/events", json={
            "event_type": "bias_audit",
            "actor": "auditor",
        })
        eid = r.json()["id"]
        resp = await client.get(f"/events/{eid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == eid

    async def test_get_event_not_found(self, client):
        resp = await client.get("/events/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


class TestMerkle:
    async def test_merkle_root_empty(self, client):
        resp = await client.get("/merkle/root")
        assert resp.status_code == 200
        assert resp.json()["merkle_root"] is None

    async def test_merkle_proof_after_events(self, client):
        r = await client.post("/events", json={
            "event_type": "inference_result",
            "actor": "model",
        })
        eid = r.json()["id"]

        resp = await client.get(f"/merkle/proof/{eid}")
        assert resp.status_code == 200
        proof = resp.json()
        assert proof["leaf_hash"] != ""
        assert proof["root_hash"] != ""

    async def test_merkle_root_updates(self, client):
        await client.post("/events", json={
            "event_type": "training_start",
            "actor": "a",
        })
        r1 = await client.get("/merkle/root")
        root1 = r1.json()["merkle_root"]

        await client.post("/events", json={
            "event_type": "training_complete",
            "actor": "a",
        })
        r2 = await client.get("/merkle/root")
        root2 = r2.json()["merkle_root"]

        assert root1 != root2


class TestBundles:
    async def test_create_and_verify_bundle(self, client):
        # Create events
        r1 = await client.post("/events", json={
            "event_type": "inference_request",
            "actor": "user",
        })
        r2 = await client.post("/events", json={
            "event_type": "inference_result",
            "actor": "model",
            "payload": {"score": 0.95},
        })

        eid1 = r1.json()["id"]
        eid2 = r2.json()["id"]

        # Create bundle
        br = await client.post("/bundles", json={
            "event_ids": [eid1, eid2],
        })
        assert br.status_code == 201
        bundle = br.json()
        assert bundle["status"] == "complete"
        assert bundle["bundle_hash"] != ""

        # Verify bundle
        vr = await client.post(f"/verify/{bundle['id']}")
        assert vr.status_code == 200
        result = vr.json()
        assert result["valid"]
        assert result["status"] == "verified"

    async def test_get_bundle(self, client):
        r = await client.post("/events", json={
            "event_type": "consent_recorded",
            "actor": "subject",
        })
        br = await client.post("/bundles", json={
            "event_ids": [r.json()["id"]],
        })
        bid = br.json()["id"]

        resp = await client.get(f"/bundles/{bid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == bid


class TestChainIntegrity:
    async def test_chain_integrity_empty(self, client):
        resp = await client.get("/chain/integrity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"]
        assert data["count"] == 0

    async def test_chain_integrity_with_events(self, client):
        for i in range(5):
            await client.post("/events", json={
                "event_type": "inference_request",
                "actor": f"agent_{i}",
            })
        resp = await client.get("/chain/integrity")
        data = resp.json()
        assert data["valid"]
        assert data["count"] == 5


class TestProvenance:
    async def test_add_and_get_provenance(self, client):
        r = await client.post("/provenance", json={
            "entity_type": "dataset",
            "entity_id": "faces_v1",
            "metadata": {"size": 10000},
        })
        assert r.status_code == 201
        node = r.json()
        assert node["entity_type"] == "dataset"

        resp = await client.get(f"/provenance/{node['id']}")
        assert resp.status_code == 200
        chain = resp.json()
        assert len(chain) == 1

    async def test_provenance_chain(self, client):
        r1 = await client.post("/provenance", json={
            "entity_type": "dataset",
            "entity_id": "d1",
        })
        r2 = await client.post("/provenance", json={
            "entity_type": "model",
            "entity_id": "m1",
            "parents": [r1.json()["id"]],
            "relations": ["derived_from"],
        })
        r3 = await client.post("/provenance", json={
            "entity_type": "inference",
            "entity_id": "inf1",
            "parents": [r2.json()["id"]],
            "relations": ["generated_by"],
        })

        resp = await client.get(f"/provenance/{r3.json()['id']}")
        chain = resp.json()
        assert len(chain) == 3
        assert chain[0]["entity_type"] == "dataset"
        assert chain[2]["entity_type"] == "inference"
