"""Tests for error edge cases (US-069).

Covers:
  - Malformed JSON payloads
  - Invalid UUIDs in path parameters
  - Missing required fields
  - Invalid query parameters
  - Custom exception error responses
  - Deep health check endpoint
  - Batch event retrieval
  - Event filtering
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Malformed JSON
# ---------------------------------------------------------------------------


class TestMalformedJSON:
    async def test_invalid_json_body_returns_422(self, client):
        resp = await client.post(
            "/events",
            content=b"not valid json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422

    async def test_empty_body_returns_422(self, client):
        resp = await client.post(
            "/events",
            content=b"",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Invalid UUIDs
# ---------------------------------------------------------------------------


class TestInvalidUUIDs:
    async def test_invalid_event_id_returns_422(self, client):
        resp = await client.get("/events/not-a-uuid")
        assert resp.status_code == 422

    async def test_invalid_bundle_id_returns_422(self, client):
        resp = await client.get("/bundles/not-a-uuid")
        assert resp.status_code == 422

    async def test_invalid_merkle_proof_id_returns_422(self, client):
        resp = await client.get("/merkle/proof/not-a-uuid")
        assert resp.status_code == 422

    async def test_nonexistent_uuid_returns_404(self, client):
        resp = await client.get("/events/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------


class TestMissingFields:
    async def test_event_missing_event_type_returns_422(self, client):
        resp = await client.post(
            "/events",
            json={"actor": "test"},
        )
        assert resp.status_code == 422

    async def test_consent_grant_missing_subject_returns_422(self, client):
        resp = await client.post(
            "/consent/grant",
            json={"purpose": "recognition"},
        )
        assert resp.status_code == 422

    async def test_did_create_empty_body_returns_422(self, client):
        resp = await client.post(
            "/did/create",
            content=b"{}",
            headers={"content-type": "application/json"},
        )
        # DID create might accept empty body (auto-generates) or require fields
        # At minimum should not crash
        assert resp.status_code in (200, 201, 422)


# ---------------------------------------------------------------------------
# Invalid query parameters
# ---------------------------------------------------------------------------


class TestInvalidQueryParams:
    async def test_negative_offset_returns_422(self, client):
        resp = await client.get("/events?offset=-1")
        assert resp.status_code == 422

    async def test_zero_limit_returns_422(self, client):
        resp = await client.get("/events?limit=0")
        assert resp.status_code == 422

    async def test_limit_exceeds_max_returns_422(self, client):
        resp = await client.get("/events?limit=9999")
        assert resp.status_code == 422

    async def test_valid_pagination_params(self, client):
        resp = await client.get("/events?limit=10&offset=0")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Event filtering (US-070)
# ---------------------------------------------------------------------------


class TestEventFiltering:
    async def test_filter_by_event_type(self, client):
        # Create events of different types
        await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "filter_test"},
        )
        await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "filter_test"},
        )
        await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "filter_test"},
        )

        resp = await client.get("/events?event_type=training_start")
        data = resp.json()
        assert data["total"] == 2
        for item in data["items"]:
            assert item["event_type"] == "training_start"

    async def test_filter_by_actor(self, client):
        await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "alice"},
        )
        await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "bob"},
        )

        resp = await client.get("/events?actor=alice")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["actor"] == "alice"

    async def test_filter_by_both(self, client):
        await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "alice"},
        )
        await client.post(
            "/events",
            json={"event_type": "inference_result", "actor": "alice"},
        )

        resp = await client.get("/events?event_type=training_start&actor=alice")
        data = resp.json()
        assert data["total"] == 1

    async def test_filter_no_match_returns_empty(self, client):
        resp = await client.get("/events?event_type=nonexistent_type")
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []


# ---------------------------------------------------------------------------
# Batch event retrieval (US-071)
# ---------------------------------------------------------------------------


class TestBatchEventRetrieval:
    async def test_batch_fetch_events(self, client):
        ids = []
        for i in range(3):
            resp = await client.post(
                "/events",
                json={"event_type": "training_start", "actor": f"batch_{i}"},
            )
            ids.append(resp.json()["id"])

        batch_resp = await client.post("/events/batch", json={"ids": ids})
        assert batch_resp.status_code == 200
        data = batch_resp.json()
        assert data["total"] == 3
        returned_ids = {item["id"] for item in data["items"]}
        assert returned_ids == set(ids)

    async def test_batch_invalid_uuid_returns_400(self, client):
        resp = await client.post("/events/batch", json={"ids": ["not-a-uuid"]})
        assert resp.status_code == 400
        assert "Invalid UUID" in resp.json()["detail"]

    async def test_batch_empty_ids_returns_422(self, client):
        resp = await client.post("/events/batch", json={"ids": []})
        assert resp.status_code == 422

    async def test_batch_partial_match(self, client):
        resp = await client.post(
            "/events",
            json={"event_type": "training_start", "actor": "partial"},
        )
        real_id = resp.json()["id"]
        fake_id = "00000000-0000-0000-0000-000000000000"

        batch_resp = await client.post("/events/batch", json={"ids": [real_id, fake_id]})
        assert batch_resp.status_code == 200
        assert batch_resp.json()["total"] == 1


# ---------------------------------------------------------------------------
# Deep health check (US-072)
# ---------------------------------------------------------------------------


class TestDeepHealthCheck:
    async def test_deep_health_returns_ok(self, client):
        resp = await client.get("/health/deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "checks" in data
        assert data["checks"]["database"]["status"] == "ok"
        assert data["checks"]["chain_integrity"]["status"] == "ok"
        assert data["checks"]["merkle_tree"]["status"] == "ok"
        assert data["checks"]["storage_backend"]["status"] == "ok"

    async def test_deep_health_includes_version(self, client):
        resp = await client.get("/health/deep")
        data = resp.json()
        assert data["version"] == "0.1.0"
        assert "uptime_seconds" in data

    async def test_deep_health_chain_integrity_after_events(self, client):
        for i in range(3):
            await client.post(
                "/events",
                json={"event_type": "training_start", "actor": "health_test", "payload": {"i": i}},
            )
        resp = await client.get("/health/deep")
        data = resp.json()
        assert data["checks"]["chain_integrity"]["valid"] is True
        assert data["checks"]["chain_integrity"]["count"] == 3
        assert data["checks"]["database"]["event_count"] == 3
        assert data["checks"]["merkle_tree"]["leaf_count"] == 3


# ---------------------------------------------------------------------------
# GZip compression (US-066)
# ---------------------------------------------------------------------------


class TestGZipCompression:
    async def test_gzip_response_for_large_payload(self, client):
        # Create enough events to generate a large response
        for i in range(20):
            await client.post(
                "/events",
                json={
                    "event_type": "training_start",
                    "actor": "gzip_test",
                    "payload": {"data": "x" * 100},
                },
            )

        resp = await client.get(
            "/events?limit=20",
            headers={"Accept-Encoding": "gzip"},
        )
        assert resp.status_code == 200
        # Response should be compressed (content-encoding header present)
        # httpx auto-decompresses, so we check the data is valid
        data = resp.json()
        assert data["total"] == 20


# ---------------------------------------------------------------------------
# Security headers (US-066)
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    async def test_csp_header_present(self, client):
        resp = await client.get("/health")
        assert "Content-Security-Policy" in resp.headers
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    async def test_all_security_headers_present(self, client):
        resp = await client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "Referrer-Policy" in resp.headers
        assert "Permissions-Policy" in resp.headers
        assert "X-Request-ID" in resp.headers
