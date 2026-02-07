"""Tests for security headers middleware (US-052)."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_security_headers_on_health_endpoint(client: AsyncClient):
    """Verify security headers are present on GET /health."""
    response = await client.get("/health")
    assert response.status_code == 200

    # Check all required security headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


@pytest.mark.asyncio
async def test_security_headers_on_events_endpoint(client: AsyncClient):
    """Verify security headers are present on POST /events."""
    from friendlyface.core.models import EventType

    payload = {
        "event_type": EventType.INFERENCE_RESULT.value,
        "actor": "test_actor",
        "payload": {"test": "data"},
    }
    response = await client.post("/events", json=payload)
    assert response.status_code == 201

    # Check all required security headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


@pytest.mark.asyncio
async def test_security_headers_on_versioned_endpoint(client: AsyncClient):
    """Verify security headers are present on /api/v1/ versioned routes."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200

    # Check all required security headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


@pytest.mark.asyncio
async def test_security_headers_on_dashboard_endpoint(client: AsyncClient):
    """Verify security headers are present on GET /dashboard."""
    response = await client.get("/dashboard")
    assert response.status_code == 200

    # Check all required security headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


@pytest.mark.asyncio
async def test_security_headers_on_error_response(client: AsyncClient):
    """Verify security headers are present even on error responses."""
    response = await client.get("/events/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404

    # Check all required security headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"


@pytest.mark.asyncio
async def test_security_headers_preserved_with_other_middleware(client: AsyncClient):
    """Verify security headers work alongside other middleware (CORS, request logging)."""
    response = await client.get("/health")
    assert response.status_code == 200

    # Security headers should be present
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"

    # Request logging middleware should also add its headers
    assert "X-Request-ID" in response.headers
