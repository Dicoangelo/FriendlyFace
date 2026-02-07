"""Tests for structured JSON logging (US-049).

Verifies that:
- FF_LOG_FORMAT=json produces valid JSON log lines with required fields.
- FF_LOG_FORMAT=text (or unset) produces human-readable output.
- FF_LOG_LEVEL controls the effective log level.
- Startup log includes version, storage_backend, rate_limit_config, did_seed_status.
- Request middleware attaches request_id, path, method, status_code, duration_ms.
- Error logs include a ``traceback`` structured field.
"""

from __future__ import annotations

import json
import logging

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from friendlyface.api.app import _db, _service, app
from friendlyface.logging_config import (
    StructuredJsonFormatter,
    log_startup_info,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def log_client(tmp_path, monkeypatch):
    """HTTP test client with JSON logging enabled."""
    monkeypatch.setenv("FF_LOG_FORMAT", "json")
    monkeypatch.setenv("FF_LOG_LEVEL", "DEBUG")

    _db.db_path = tmp_path / "log_test.db"
    await _db.connect()
    await _service.initialize()

    _service.merkle = __import__("friendlyface.core.merkle", fromlist=["MerkleTree"]).MerkleTree()
    _service._event_index = {}
    _service.provenance = __import__(
        "friendlyface.core.provenance", fromlist=["ProvenanceDAG"]
    ).ProvenanceDAG()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    await _db.close()


# ---------------------------------------------------------------------------
# Unit tests — StructuredJsonFormatter
# ---------------------------------------------------------------------------


class TestStructuredJsonFormatter:
    """Verify the JSON formatter produces valid JSON with expected keys."""

    def test_basic_log_record_is_valid_json(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="friendlyface",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        line = formatter.format(record)
        parsed = json.loads(line)
        assert parsed["message"] == "hello world"
        assert parsed["levelname"] == "INFO"
        assert "asctime" in parsed

    def test_structured_extras_appear_in_json(self):
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="friendlyface",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="request finished",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc12345"
        record.path = "/health"
        record.method = "GET"
        record.status_code = 200
        record.duration_ms = 12.3
        line = formatter.format(record)
        parsed = json.loads(line)
        assert parsed["request_id"] == "abc12345"
        assert parsed["path"] == "/health"
        assert parsed["method"] == "GET"
        assert parsed["status_code"] == 200
        assert parsed["duration_ms"] == 12.3

    def test_traceback_appears_as_structured_field(self):
        formatter = StructuredJsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="friendlyface",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="something failed",
            args=(),
            exc_info=exc_info,
        )
        line = formatter.format(record)
        parsed = json.loads(line)
        assert "traceback" in parsed
        assert isinstance(parsed["traceback"], list)
        tb_text = "".join(parsed["traceback"])
        assert "ValueError" in tb_text
        assert "boom" in tb_text


# ---------------------------------------------------------------------------
# Unit tests — setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Verify that setup_logging respects env vars."""

    def test_json_mode_sets_json_formatter(self, monkeypatch):
        monkeypatch.setenv("FF_LOG_FORMAT", "json")
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, StructuredJsonFormatter)

    def test_text_mode_sets_standard_formatter(self, monkeypatch):
        monkeypatch.setenv("FF_LOG_FORMAT", "text")
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert not isinstance(root.handlers[0].formatter, StructuredJsonFormatter)

    def test_default_mode_is_text(self, monkeypatch):
        monkeypatch.delenv("FF_LOG_FORMAT", raising=False)
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert not isinstance(root.handlers[0].formatter, StructuredJsonFormatter)

    def test_log_level_from_env(self, monkeypatch):
        monkeypatch.setenv("FF_LOG_LEVEL", "WARNING")
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_default_log_level_is_info(self, monkeypatch):
        monkeypatch.delenv("FF_LOG_LEVEL", raising=False)
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_invalid_log_level_falls_back_to_info(self, monkeypatch):
        monkeypatch.setenv("FF_LOG_LEVEL", "NOTAVALIDLEVEL")
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO


# ---------------------------------------------------------------------------
# Unit tests — log_startup_info
# ---------------------------------------------------------------------------


class TestLogStartupInfo:
    """Verify the startup log emits expected structured fields."""

    def test_startup_log_contains_required_fields(self, monkeypatch, caplog):
        monkeypatch.delenv("FF_DID_SEED", raising=False)
        monkeypatch.delenv("FF_STORAGE", raising=False)
        with caplog.at_level(logging.INFO, logger="friendlyface"):
            log_startup_info()
        assert len(caplog.records) >= 1
        rec = caplog.records[-1]
        assert "FriendlyFace started" in rec.message
        assert rec.version == "0.1.0"
        assert rec.storage_backend == "sqlite"
        assert rec.did_seed_status == "random"
        assert hasattr(rec, "rate_limit_config")

    def test_startup_log_with_did_seed(self, monkeypatch, caplog):
        monkeypatch.setenv("FF_DID_SEED", "aa" * 32)
        with caplog.at_level(logging.INFO, logger="friendlyface"):
            log_startup_info()
        rec = caplog.records[-1]
        assert rec.did_seed_status == "configured"

    def test_startup_log_with_supabase_backend(self, monkeypatch, caplog):
        monkeypatch.setenv("FF_STORAGE", "supabase")
        with caplog.at_level(logging.INFO, logger="friendlyface"):
            log_startup_info()
        rec = caplog.records[-1]
        assert rec.storage_backend == "supabase"


# ---------------------------------------------------------------------------
# Integration test — request middleware JSON output
# ---------------------------------------------------------------------------


class TestRequestLoggingMiddlewareJSON:
    """Verify that HTTP requests produce structured JSON log output."""

    async def test_health_request_logs_json_fields(self, log_client, caplog):
        with caplog.at_level(logging.INFO, logger="friendlyface"):
            resp = await log_client.get("/health")
        assert resp.status_code == 200

        # Find the middleware log record
        middleware_records = [
            r for r in caplog.records if hasattr(r, "request_id") and hasattr(r, "path")
        ]
        assert len(middleware_records) >= 1
        rec = middleware_records[-1]
        assert rec.path == "/health"
        assert rec.method == "GET"
        assert rec.status_code == 200
        assert isinstance(rec.duration_ms, float)
        assert len(rec.request_id) == 8

    async def test_response_has_request_id_header(self, log_client):
        resp = await log_client.get("/health")
        assert resp.status_code == 200
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) == 8

    async def test_post_request_logs_201(self, log_client, caplog):
        with caplog.at_level(logging.INFO, logger="friendlyface"):
            resp = await log_client.post(
                "/events",
                json={"event_type": "training_start", "actor": "test_logging"},
            )
        assert resp.status_code == 201

        middleware_records = [
            r for r in caplog.records if hasattr(r, "status_code") and r.status_code == 201
        ]
        assert len(middleware_records) >= 1
        rec = middleware_records[-1]
        assert rec.method == "POST"
        assert rec.path == "/events"


# ---------------------------------------------------------------------------
# Integration test — JSON formatter on captured output
# ---------------------------------------------------------------------------


class TestJSONOutputFormat:
    """Verify actual JSON output lines parse correctly."""

    def test_json_format_produces_parseable_json(self, monkeypatch):
        monkeypatch.setenv("FF_LOG_FORMAT", "json")
        setup_logging()
        root = logging.getLogger()
        handler = root.handlers[0]

        record = logging.LogRecord(
            name="friendlyface",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.request_id = "abcd1234"
        record.path = "/events"
        record.method = "POST"
        record.status_code = 201
        record.duration_ms = 5.2

        output = handler.format(record)
        parsed = json.loads(output)

        assert parsed["message"] == "test message"
        assert parsed["request_id"] == "abcd1234"
        assert parsed["path"] == "/events"
        assert parsed["method"] == "POST"
        assert parsed["status_code"] == 201
        assert parsed["duration_ms"] == 5.2
        assert "asctime" in parsed
        assert parsed["levelname"] == "INFO"
        assert parsed["name"] == "friendlyface"
