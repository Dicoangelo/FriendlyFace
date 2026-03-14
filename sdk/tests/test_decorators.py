"""Tests for @forensic_trace decorator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from friendlyface_sdk.client import FriendlyFaceClient
from friendlyface_sdk.decorators import forensic_trace
from friendlyface_sdk.models import ForensicEvent


def _make_event(event_id: str = "evt-1") -> ForensicEvent:
    return ForensicEvent(
        event_id=event_id,
        event_type="forensic_trace_input",
        event_hash="hash",
        timestamp="2026-01-01T00:00:00Z",
    )


class TestForensicTrace:
    def test_forensic_trace_decorator(self):
        with patch("friendlyface_sdk.client.requests.Session"):
            client = FriendlyFaceClient("http://localhost:8000")
            client.log_event = MagicMock(side_effect=[_make_event("in-1"), _make_event("out-1")])

            @forensic_trace(client)
            def add(a: int, b: int) -> int:
                return a + b

            result = add(2, 3)

            assert result == 5
            assert client.log_event.call_count == 2

            # First call: input event
            first_call = client.log_event.call_args_list[0]
            assert first_call.kwargs["event_type"] == "forensic_trace_input"
            assert "add" in first_call.kwargs["content"]["function"]

            # Second call: output event
            second_call = client.log_event.call_args_list[1]
            assert second_call.kwargs["event_type"] == "forensic_trace_output"
            assert "5" in second_call.kwargs["content"]["result"]

    def test_decorator_preserves_function_name(self):
        with patch("friendlyface_sdk.client.requests.Session"):
            client = FriendlyFaceClient("http://localhost:8000")
            client.log_event = MagicMock(return_value=_make_event())

            @forensic_trace(client)
            def my_function():
                pass

            assert my_function.__name__ == "my_function"
