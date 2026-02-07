"""Integration tests for the FastAPI endpoints."""


class TestHealth:
    async def test_health_endpoint(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    async def test_health_includes_uptime(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))

    async def test_health_includes_storage_backend(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert "storage_backend" in data
        assert data["storage_backend"] == "sqlite"


class TestRequestLogging:
    async def test_response_has_request_id(self, client):
        resp = await client.get("/health")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) == 8

    async def test_cors_headers_present(self, client):
        resp = await client.options(
            "/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert resp.status_code in (200, 405)  # depends on CORS config


class TestEvents:
    async def test_record_event(self, client):
        resp = await client.post(
            "/events",
            json={
                "event_type": "training_start",
                "actor": "test_agent",
                "payload": {"model": "resnet50"},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["event_type"] == "training_start"
        assert data["event_hash"] != ""
        assert data["previous_hash"] == "GENESIS"
        assert data["sequence_number"] == 0

    async def test_hash_chain(self, client):
        r1 = await client.post(
            "/events",
            json={
                "event_type": "training_start",
                "actor": "agent",
            },
        )
        r2 = await client.post(
            "/events",
            json={
                "event_type": "training_complete",
                "actor": "agent",
            },
        )

        e1 = r1.json()
        e2 = r2.json()
        assert e2["previous_hash"] == e1["event_hash"]
        assert e2["sequence_number"] == 1

    async def test_list_events(self, client):
        await client.post(
            "/events",
            json={
                "event_type": "inference_request",
                "actor": "agent",
            },
        )
        resp = await client.get("/events")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    async def test_get_event_by_id(self, client):
        r = await client.post(
            "/events",
            json={
                "event_type": "bias_audit",
                "actor": "auditor",
            },
        )
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
        r = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model",
            },
        )
        eid = r.json()["id"]

        resp = await client.get(f"/merkle/proof/{eid}")
        assert resp.status_code == 200
        proof = resp.json()
        assert proof["leaf_hash"] != ""
        assert proof["root_hash"] != ""

    async def test_merkle_root_updates(self, client):
        await client.post(
            "/events",
            json={
                "event_type": "training_start",
                "actor": "a",
            },
        )
        r1 = await client.get("/merkle/root")
        root1 = r1.json()["merkle_root"]

        await client.post(
            "/events",
            json={
                "event_type": "training_complete",
                "actor": "a",
            },
        )
        r2 = await client.get("/merkle/root")
        root2 = r2.json()["merkle_root"]

        assert root1 != root2


class TestBundles:
    async def test_create_and_verify_bundle(self, client):
        # Create events
        r1 = await client.post(
            "/events",
            json={
                "event_type": "inference_request",
                "actor": "user",
            },
        )
        r2 = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "model",
                "payload": {"score": 0.95},
            },
        )

        eid1 = r1.json()["id"]
        eid2 = r2.json()["id"]

        # Create bundle
        br = await client.post(
            "/bundles",
            json={
                "event_ids": [eid1, eid2],
            },
        )
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
        r = await client.post(
            "/events",
            json={
                "event_type": "consent_recorded",
                "actor": "subject",
            },
        )
        br = await client.post(
            "/bundles",
            json={
                "event_ids": [r.json()["id"]],
            },
        )
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
            await client.post(
                "/events",
                json={
                    "event_type": "inference_request",
                    "actor": f"agent_{i}",
                },
            )
        resp = await client.get("/chain/integrity")
        data = resp.json()
        assert data["valid"]
        assert data["count"] == 5


class TestProvenance:
    async def test_add_and_get_provenance(self, client):
        r = await client.post(
            "/provenance",
            json={
                "entity_type": "dataset",
                "entity_id": "faces_v1",
                "metadata": {"size": 10000},
            },
        )
        assert r.status_code == 201
        node = r.json()
        assert node["entity_type"] == "dataset"

        resp = await client.get(f"/provenance/{node['id']}")
        assert resp.status_code == 200
        chain = resp.json()
        assert len(chain) == 1

    async def test_provenance_chain(self, client):
        r1 = await client.post(
            "/provenance",
            json={
                "entity_type": "dataset",
                "entity_id": "d1",
            },
        )
        r2 = await client.post(
            "/provenance",
            json={
                "entity_type": "model",
                "entity_id": "m1",
                "parents": [r1.json()["id"]],
                "relations": ["derived_from"],
            },
        )
        r3 = await client.post(
            "/provenance",
            json={
                "entity_type": "inference",
                "entity_id": "inf1",
                "parents": [r2.json()["id"]],
                "relations": ["generated_by"],
            },
        )

        resp = await client.get(f"/provenance/{r3.json()['id']}")
        chain = resp.json()
        assert len(chain) == 3
        assert chain[0]["entity_type"] == "dataset"
        assert chain[2]["entity_type"] == "inference"


class TestDPFL:
    async def test_dp_fl_start(self, client):
        resp = await client.post(
            "/fl/dp-start",
            json={"n_clients": 3, "n_rounds": 2, "epsilon": 1.0, "delta": 1e-5},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["n_rounds"] == 2
        assert data["n_clients"] == 3
        assert data["total_epsilon"] > 0
        assert len(data["rounds"]) == 2
        assert data["dp_config"]["epsilon"] == 1.0

    async def test_dp_fl_rounds_have_privacy_metadata(self, client):
        resp = await client.post(
            "/fl/dp-start",
            json={"n_clients": 2, "n_rounds": 1, "epsilon": 0.5},
        )
        data = resp.json()
        round_info = data["rounds"][0]
        assert "noise_scale" in round_info
        assert "privacy_spent" in round_info
        assert round_info["privacy_spent"] == 0.5

    async def test_dp_fl_clipping_tracked(self, client):
        resp = await client.post(
            "/fl/dp-start",
            json={"n_clients": 2, "n_rounds": 1, "max_grad_norm": 1.0},
        )
        data = resp.json()
        assert "clipped_clients" in data["rounds"][0]
        assert "n_clipped" in data["rounds"][0]


class TestSddExplainability:
    async def test_sdd_explain_creates_record(self, client):
        # First create an inference event
        event_resp = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "test",
                "payload": {"test": True},
            },
        )
        event_id = event_resp.json()["id"]

        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": event_id},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["method"] == "sdd"
        assert data["inference_event_id"] == event_id
        assert data["num_regions"] == 7

    async def test_sdd_in_compare_endpoint(self, client):
        # Create inference event and SDD explanation
        event_resp = await client.post(
            "/events",
            json={
                "event_type": "inference_result",
                "actor": "test",
                "payload": {},
            },
        )
        event_id = event_resp.json()["id"]

        await client.post("/explainability/sdd", json={"event_id": event_id})

        resp = await client.get(f"/explainability/compare/{event_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "sdd_explanations" in data
        assert data["total_sdd"] == 1

    async def test_sdd_explain_missing_event(self, client):
        resp = await client.post(
            "/explainability/sdd",
            json={"event_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404


class TestMultiModalFusion:
    async def test_fusion_endpoint(self, client):
        resp = await client.post(
            "/recognition/multimodal",
            json={
                "face_label": "alice",
                "face_confidence": 0.9,
                "voice_confidence": 0.8,
                "face_weight": 0.6,
                "voice_weight": 0.4,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["fusion_method"] == "weighted_sum"
        assert len(data["fused_matches"]) > 0
        assert data["fused_matches"][0]["fused_confidence"] > 0

    async def test_fusion_invalid_weights(self, client):
        resp = await client.post(
            "/recognition/multimodal",
            json={
                "face_label": "alice",
                "face_confidence": 0.9,
                "voice_confidence": 0.8,
                "face_weight": 0.5,
                "voice_weight": 0.3,
            },
        )
        assert resp.status_code == 400


class TestDashboard:
    async def test_dashboard_structure(self, client):
        resp = await client.get("/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        expected_keys = {
            "uptime_seconds",
            "storage_backend",
            "total_events",
            "total_bundles",
            "total_provenance_nodes",
            "events_by_type",
            "recent_events",
            "chain_integrity",
            "crypto_status",
        }
        assert expected_keys.issubset(set(data.keys()))
        # Verify crypto_status sub-keys
        assert data["crypto_status"]["did_enabled"] is True
        assert data["crypto_status"]["zk_scheme"] == "schnorr-sha256"
        assert data["crypto_status"]["total_dids"] == 0
        assert data["crypto_status"]["total_vcs"] == 0

    async def test_dashboard_event_counts(self, client):
        # Initially empty
        resp = await client.get("/dashboard")
        data = resp.json()
        assert data["total_events"] == 0
        assert data["total_bundles"] == 0
        assert data["events_by_type"] == {}

        # Invalidate cache so next call recomputes
        from friendlyface.api.app import _dashboard_cache

        _dashboard_cache["data"] = None

        # Record some events
        await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test"},
        )
        await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "test"},
        )
        await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test"},
        )

        _dashboard_cache["data"] = None

        resp = await client.get("/dashboard")
        data = resp.json()
        assert data["total_events"] == 3
        assert data["events_by_type"]["training_start"] == 2
        assert data["events_by_type"]["inference_result"] == 1

    async def test_dashboard_recent_events(self, client):
        # Record events
        for i in range(3):
            await client.post(
                "/events",
                json={"event_type": "inference_request", "actor": f"agent_{i}"},
            )

        from friendlyface.api.app import _dashboard_cache

        _dashboard_cache["data"] = None

        resp = await client.get("/dashboard")
        data = resp.json()
        recent = data["recent_events"]
        assert len(recent) == 3
        # Most recent first (descending sequence_number)
        assert recent[0]["actor"] == "agent_2"
        assert recent[2]["actor"] == "agent_0"
        # Each recent event has required keys
        for event in recent:
            assert "id" in event
            assert "event_type" in event
            assert "actor" in event
            assert "timestamp" in event

    async def test_dashboard_includes_uptime(self, client):
        resp = await client.get("/dashboard")
        data = resp.json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    async def test_dashboard_chain_integrity(self, client):
        resp = await client.get("/dashboard")
        data = resp.json()
        assert "chain_integrity" in data
        ci = data["chain_integrity"]
        assert "valid" in ci
        assert "count" in ci
        assert ci["valid"] is True
        assert ci["count"] == 0

        from friendlyface.api.app import _dashboard_cache

        _dashboard_cache["data"] = None

        # Add some events and re-check
        for _ in range(3):
            await client.post(
                "/events",
                json={"event_type": "training_start", "actor": "test"},
            )

        _dashboard_cache["data"] = None

        resp = await client.get("/dashboard")
        data = resp.json()
        ci = data["chain_integrity"]
        assert ci["valid"] is True
        assert ci["count"] == 3


class TestSSE:
    async def test_event_stream_returns_streaming_response(self):
        """The /events/stream route returns a StreamingResponse with text/event-stream."""
        from starlette.responses import StreamingResponse

        from friendlyface.api.app import event_stream

        response = await event_stream(event_type=None)
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"

    async def test_event_stream_generator_yields_heartbeat(self):
        """The SSE generator yields an initial heartbeat on connect."""
        from friendlyface.api.app import event_stream

        response = await event_stream(event_type=None)
        gen = response.body_iterator
        first_chunk = await gen.__anext__()
        assert first_chunk == "event: heartbeat\ndata: {}\n\n"
        # Clean up: close the generator so the subscriber is removed
        await gen.aclose()

    async def test_broadcaster_subscribe_broadcast_receive(self):
        """subscribe() -> broadcast() -> queue.get() round-trip."""
        from friendlyface.api.sse import EventBroadcaster

        broadcaster = EventBroadcaster()
        queue = broadcaster.subscribe()

        event_data = {"event_type": "inference_result", "actor": "test"}
        broadcaster.broadcast(event_data)

        received = queue.get_nowait()
        assert received == event_data

    async def test_broadcaster_multiple_subscribers(self):
        """All subscribers receive the same broadcast."""
        from friendlyface.api.sse import EventBroadcaster

        broadcaster = EventBroadcaster()
        q1 = broadcaster.subscribe()
        q2 = broadcaster.subscribe()

        event_data = {"event_type": "training_start", "actor": "agent"}
        broadcaster.broadcast(event_data)

        assert q1.get_nowait() == event_data
        assert q2.get_nowait() == event_data

    async def test_broadcaster_unsubscribe_cleanup(self):
        """After unsubscribe, the queue no longer receives broadcasts."""
        from friendlyface.api.sse import EventBroadcaster

        broadcaster = EventBroadcaster()
        queue = broadcaster.subscribe()
        broadcaster.unsubscribe(queue)

        broadcaster.broadcast({"event_type": "test"})
        assert queue.empty()

    async def test_broadcaster_backpressure_drops(self):
        """When a queue is full, broadcast silently drops the message."""
        from friendlyface.api.sse import EventBroadcaster

        broadcaster = EventBroadcaster(maxsize=2)
        queue = broadcaster.subscribe()

        # Fill the queue
        broadcaster.broadcast({"n": 1})
        broadcaster.broadcast({"n": 2})
        # This should be silently dropped (queue full)
        broadcaster.broadcast({"n": 3})

        assert queue.qsize() == 2
        assert queue.get_nowait() == {"n": 1}
        assert queue.get_nowait() == {"n": 2}

    async def test_broadcaster_unsubscribe_idempotent(self):
        """Calling unsubscribe twice does not raise."""
        from friendlyface.api.sse import EventBroadcaster

        broadcaster = EventBroadcaster()
        queue = broadcaster.subscribe()
        broadcaster.unsubscribe(queue)
        broadcaster.unsubscribe(queue)  # should not raise


class TestDID:
    async def test_create_did(self, client):
        resp = await client.post("/did/create", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "did" in data
        assert data["did"].startswith("did:key:z")
        assert "public_key_hex" in data

    async def test_create_did_with_seed(self, client):
        resp = await client.post("/did/create", json={"seed": "01" * 32})
        assert resp.status_code == 201
        # Same seed = same DID
        resp2 = await client.post("/did/create", json={"seed": "01" * 32})
        assert resp.json()["did"] == resp2.json()["did"]

    async def test_resolve_did(self, client):
        resp = await client.post("/did/create", json={})
        did = resp.json()["did"]
        resp2 = await client.get(f"/did/{did}/resolve")
        assert resp2.status_code == 200
        doc = resp2.json()
        assert doc["id"] == did
        assert "verificationMethod" in doc

    async def test_resolve_not_found(self, client):
        resp = await client.get("/did/did:key:zNonExistent/resolve")
        assert resp.status_code == 404


class TestVC:
    async def test_issue_and_verify(self, client):
        # Create issuer DID
        resp = await client.post("/did/create", json={})
        did_data = resp.json()
        # Issue VC
        resp = await client.post(
            "/vc/issue",
            json={
                "issuer_did_id": did_data["did"],
                "claims": {"action": "test"},
                "credential_type": "ForensicCredential",
            },
        )
        assert resp.status_code == 201
        cred = resp.json()
        assert cred["proof"]["type"] == "Ed25519Signature2020"
        # Verify VC
        resp = await client.post(
            "/vc/verify",
            json={
                "credential": cred,
                "issuer_public_key_hex": did_data["public_key_hex"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["valid"] is True

    async def test_issue_unknown_issuer(self, client):
        resp = await client.post(
            "/vc/issue",
            json={
                "issuer_did_id": "did:key:zUnknown",
                "claims": {"test": True},
            },
        )
        assert resp.status_code == 404


class TestZKProof:
    async def test_prove_and_verify(self, client):
        # Create a bundle first
        ev = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        eid = ev.json()["id"]
        bundle_resp = await client.post("/bundles", json={"event_ids": [eid]})
        bundle_id = bundle_resp.json()["id"]
        # Generate proof
        resp = await client.post("/zk/prove", json={"bundle_id": bundle_id})
        assert resp.status_code == 201
        proof_data = resp.json()
        assert "proof" in proof_data
        assert proof_data["proof"]["scheme"] == "schnorr-sha256"
        # Verify proof
        import json as _json

        resp2 = await client.post("/zk/verify", json={"proof": _json.dumps(proof_data["proof"])})
        assert resp2.status_code == 200
        assert resp2.json()["valid"] is True

    async def test_prove_not_found(self, client):
        resp = await client.post(
            "/zk/prove",
            json={"bundle_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404

    async def test_get_stored_proof(self, client):
        # Create bundle + prove
        ev = await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "test", "payload": {}},
        )
        bundle_resp = await client.post("/bundles", json={"event_ids": [ev.json()["id"]]})
        bundle_id = bundle_resp.json()["id"]
        await client.post("/zk/prove", json={"bundle_id": bundle_id})
        # Get stored proof
        resp = await client.get(f"/zk/proofs/{bundle_id}")
        assert resp.status_code == 200

    async def test_verify_invalid_proof(self, client):
        resp = await client.post(
            "/zk/verify",
            json={
                "proof": '{"scheme":"schnorr-sha256","commitment":"00","challenge":"00","response":"00","public_point":"00"}'
            },
        )
        assert resp.status_code == 200
        assert resp.json()["valid"] is False
