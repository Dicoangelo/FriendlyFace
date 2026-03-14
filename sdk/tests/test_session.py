"""Tests for ForensicSession context manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from friendlyface_sdk.client import FriendlyFaceClient
from friendlyface_sdk.models import Bundle, ForensicEvent



def _make_event(event_id: str) -> ForensicEvent:
    return ForensicEvent(
        event_id=event_id,
        event_type="test",
        event_hash="hash",
        timestamp="2026-01-01T00:00:00Z",
    )


class TestForensicSession:
    def test_forensic_session_bundles_events(self):
        with patch("friendlyface_sdk.client.requests.Session"):
            client = FriendlyFaceClient("http://localhost:8000")
            client.log_event = MagicMock(
                side_effect=[_make_event("evt-1"), _make_event("evt-2")]
            )
            client.create_bundle = MagicMock(
                return_value=Bundle(
                    bundle_id="bnd-1",
                    events=["evt-1", "evt-2"],
                    merkle_root="root",
                )
            )

            with client.forensic_session() as session:
                session.log_event("inference_request", {"actor": "sdk"})
                session.log_event("inference_result", {"actor": "sdk"})

            assert client.log_event.call_count == 2
            client.create_bundle.assert_called_once_with(["evt-1", "evt-2"])
            assert session.bundle is not None
            assert session.bundle.bundle_id == "bnd-1"

    def test_forensic_session_no_bundle_on_empty(self):
        with patch("friendlyface_sdk.client.requests.Session"):
            client = FriendlyFaceClient("http://localhost:8000")
            client.create_bundle = MagicMock()

            with client.forensic_session() as session:
                pass  # no events logged

            client.create_bundle.assert_not_called()
            assert session.bundle is None

    def test_forensic_session_event_ids_tracked(self):
        with patch("friendlyface_sdk.client.requests.Session"):
            client = FriendlyFaceClient("http://localhost:8000")
            client.log_event = MagicMock(
                side_effect=[_make_event("a"), _make_event("b"), _make_event("c")]
            )
            client.create_bundle = MagicMock(return_value=Bundle(bundle_id="bnd-x"))

            with client.forensic_session() as session:
                session.log_event("t1", {"actor": "sdk"})
                session.log_event("t2", {"actor": "sdk"})
                session.log_event("t3", {"actor": "sdk"})
                assert session.event_ids == ["a", "b", "c"]
