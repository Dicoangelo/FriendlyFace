"""Tests for US-005: Compliance Proxy Middleware.

Tests the proxy's ability to:
- Forward recognition requests to an upstream API (using /proxy/echo)
- Log forensic events (inference_request + inference_result)
- Check consent when subject_id is provided
- Handle upstream failures gracefully
- Include correct fields in forensic event payloads
"""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api import app as app_module
from friendlyface.api.app import _db, _service, app, limiter


@pytest_asyncio.fixture
async def client(tmp_path):
    """HTTP test client wired to a fresh database with proxy initialized."""
    _db.db_path = tmp_path / "proxy_test.db"
    await _db.connect()
    await _db.run_migrations()
    await _service.initialize()

    # Reset in-memory state
    _service.merkle = __import__("friendlyface.core.merkle", fromlist=["MerkleTree"]).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()

    # Disable rate limiter for tests
    limiter.enabled = False

    transport = ASGITransport(app=app)

    # Create an httpx client that routes to ASGI transport (for proxy upstream calls)
    proxy_http_client = AsyncClient(transport=transport, base_url="http://test")

    # Initialize proxy with the ASGI-backed httpx client so upstream calls
    # go through the same FastAPI app (hitting /proxy/echo)
    from friendlyface.proxy.middleware import ComplianceProxy

    app_module._proxy = ComplianceProxy(
        forensic_service=_service, db=_db, http_client=proxy_http_client
    )

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    # Cleanup
    await proxy_http_client.aclose()
    app_module._proxy = None
    await _db.close()


# ---------------------------------------------------------------------------
# Echo endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_echo_returns_image_metadata(client):
    """Echo endpoint should return image hash and size."""
    image_bytes = b"fake-image-data-for-testing"
    resp = await client.post(
        "/proxy/echo",
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "input_hash" in data
    assert data["image_size"] == len(image_bytes)
    assert data["matches"][0]["label"] == "echo_test"
    assert data["matches"][0]["confidence"] == 0.99


# ---------------------------------------------------------------------------
# Proxy recognize tests (using echo as upstream)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_recognize_with_echo_upstream(client):
    """Proxy should forward to echo endpoint and return forensic event IDs."""
    image_bytes = b"proxy-test-image-data"

    # Use the echo endpoint as the upstream
    resp = await client.post(
        "/proxy/recognize",
        params={"upstream_url": "http://test/proxy/echo"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Check upstream response came through
    assert data["upstream_status"] == 200
    assert data["upstream_response"]["matches"][0]["label"] == "echo_test"

    # Check forensic metadata
    forensic = data["forensic"]
    assert forensic["request_event_id"] is not None
    assert forensic["result_event_id"] is not None
    assert forensic["input_hash"] is not None
    assert forensic["latency_ms"] >= 0
    assert forensic["consent_checked"] is False
    assert forensic["consent_allowed"] is None


@pytest.mark.asyncio
async def test_proxy_logs_forensic_events(client):
    """Proxy should create inference_request and inference_result events."""
    # Count events before
    events_before = await client.get("/events")
    count_before = len(events_before.json()["items"])

    image_bytes = b"forensic-event-test-image"
    await client.post(
        "/proxy/recognize",
        params={"upstream_url": "http://test/proxy/echo"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )

    # Count events after — should have 2 new events (request + result)
    events_after = await client.get("/events")
    count_after = len(events_after.json()["items"])
    assert count_after == count_before + 2

    # Verify event types
    items = events_after.json()["items"]
    event_types = [e["event_type"] for e in items]
    assert "inference_request" in event_types
    assert "inference_result" in event_types


@pytest.mark.asyncio
async def test_proxy_forensic_event_payloads(client):
    """Forensic events should contain correct fields."""
    image_bytes = b"payload-test-image"
    resp = await client.post(
        "/proxy/recognize",
        params={"upstream_url": "http://test/proxy/echo"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    data = resp.json()

    # Get the request event
    req_event_id = data["forensic"]["request_event_id"]
    req_resp = await client.get(f"/events/{req_event_id}")
    assert req_resp.status_code == 200
    req_event = req_resp.json()

    assert req_event["payload"]["upstream_url"] == "http://test/proxy/echo"
    assert req_event["payload"]["input_hash"] is not None
    assert req_event["payload"]["consent_checked"] is False
    assert req_event["actor"] == "compliance_proxy"

    # Get the result event
    result_event_id = data["forensic"]["result_event_id"]
    result_resp = await client.get(f"/events/{result_event_id}")
    assert result_resp.status_code == 200
    result_event = result_resp.json()

    assert result_event["payload"]["upstream_status"] == 200
    assert result_event["payload"]["latency_ms"] >= 0
    assert result_event["payload"]["input_hash"] is not None
    assert "response_summary" in result_event["payload"]


@pytest.mark.asyncio
async def test_proxy_without_subject_id(client):
    """Proxy should work without subject_id (no consent check)."""
    image_bytes = b"no-subject-test"
    resp = await client.post(
        "/proxy/recognize",
        params={"upstream_url": "http://test/proxy/echo"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    forensic = resp.json()["forensic"]
    assert forensic["consent_checked"] is False
    assert forensic["consent_allowed"] is None


@pytest.mark.asyncio
async def test_proxy_with_consent_checking(client):
    """Proxy should check consent when subject_id is provided."""
    subject_id = "test-subject-001"

    # Grant consent first
    await client.post(
        "/consent/grant",
        json={
            "subject_id": subject_id,
            "purpose": "recognition",
            "actor": "test",
        },
    )

    # Proxy with subject_id
    image_bytes = b"consent-test-image"
    resp = await client.post(
        "/proxy/recognize",
        params={
            "upstream_url": "http://test/proxy/echo",
            "subject_id": subject_id,
        },
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    forensic = resp.json()["forensic"]
    assert forensic["consent_checked"] is True
    assert forensic["consent_allowed"] is True


@pytest.mark.asyncio
async def test_proxy_with_upstream_failure(client):
    """Proxy should handle upstream failures gracefully.

    Since the test client uses ASGI transport, we test upstream failure by
    creating a proxy with a separate httpx client that points to a real
    (non-existent) network host.
    """
    from friendlyface.proxy.middleware import ComplianceProxy

    # Create a proxy with a real httpx client (no ASGI transport)
    # so the request to a bogus host actually fails
    failure_proxy = ComplianceProxy(
        forensic_service=_service,
        db=_db,
        http_client=httpx.AsyncClient(timeout=2.0),
    )

    image_bytes = b"failure-test-image"
    result = await failure_proxy.recognize(
        image_bytes=image_bytes,
        upstream_url="http://192.0.2.1:9999/recognize",  # RFC 5737 TEST-NET, guaranteed unreachable
    )

    assert result["upstream_status"] == 502
    assert result["upstream_response"] == {}
    assert result["forensic"]["request_event_id"] is not None
    assert "error" in result["forensic"]

    await failure_proxy.close()


@pytest.mark.asyncio
async def test_proxy_no_upstream_url_returns_400(client):
    """Proxy should return 400 if no upstream URL is configured or provided."""
    image_bytes = b"no-url-test"
    resp = await client.post(
        "/proxy/recognize",
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 400
    assert "upstream" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_proxy_with_metadata_json(client):
    """Proxy should accept metadata as JSON query parameter."""
    image_bytes = b"metadata-test-image"
    metadata = {"camera_id": "cam-001", "location": "entrance"}
    resp = await client.post(
        "/proxy/recognize",
        params={
            "upstream_url": "http://test/proxy/echo",
            "metadata_json": json.dumps(metadata),
        },
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert resp.status_code == 200

    # Check that metadata was included in the forensic event
    data = resp.json()
    req_event_id = data["forensic"]["request_event_id"]
    req_resp = await client.get(f"/events/{req_event_id}")
    req_event = req_resp.json()
    assert req_event["payload"]["metadata"]["camera_id"] == "cam-001"
    assert req_event["payload"]["metadata"]["location"] == "entrance"


@pytest.mark.asyncio
async def test_proxy_response_summary_extracts_match_count(client):
    """Response summary in forensic event should extract match_count from upstream."""
    image_bytes = b"summary-test-image"
    resp = await client.post(
        "/proxy/recognize",
        params={"upstream_url": "http://test/proxy/echo"},
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    data = resp.json()

    result_event_id = data["forensic"]["result_event_id"]
    result_resp = await client.get(f"/events/{result_event_id}")
    result_event = result_resp.json()

    summary = result_event["payload"]["response_summary"]
    assert summary["match_count"] == 1  # echo returns 1 match
