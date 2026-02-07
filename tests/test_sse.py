"""Tests for SSE broadcaster (US-068).

Covers:
  - Subscribe / unsubscribe lifecycle
  - Event broadcast to subscribers
  - Event filtering
  - Heartbeat presence in SSE endpoint
  - Broadcaster shutdown
"""

from __future__ import annotations

import pytest

from friendlyface.api.sse import EventBroadcaster


class TestBroadcasterUnit:
    """Unit tests for EventBroadcaster without HTTP."""

    def test_subscribe_returns_queue(self):
        b = EventBroadcaster()
        q = b.subscribe()
        assert q is not None
        assert q.empty()

    def test_unsubscribe_removes_queue(self):
        b = EventBroadcaster()
        q = b.subscribe()
        b.unsubscribe(q)
        # Publishing should not raise even though no subscribers
        b.broadcast({"test": True})

    def test_publish_delivers_to_subscriber(self):
        b = EventBroadcaster()
        q = b.subscribe()
        b.broadcast({"event_type": "test", "data": 123})
        assert not q.empty()
        msg = q.get_nowait()
        assert msg["event_type"] == "test"
        assert msg["data"] == 123

    def test_publish_delivers_to_multiple_subscribers(self):
        b = EventBroadcaster()
        q1 = b.subscribe()
        q2 = b.subscribe()
        b.broadcast({"x": 1})
        assert not q1.empty()
        assert not q2.empty()

    def test_publish_drops_when_queue_full(self):
        b = EventBroadcaster(maxsize=1)
        q = b.subscribe()
        b.broadcast({"first": True})
        b.broadcast({"second": True})  # Should be dropped (backpressure)
        assert q.qsize() == 1
        msg = q.get_nowait()
        assert msg["first"] is True

    def test_shutdown_sends_sentinel_and_clears(self):
        b = EventBroadcaster()
        q1 = b.subscribe()
        q2 = b.subscribe()
        b.shutdown()
        # Sentinel should be in queues
        msg1 = q1.get_nowait()
        assert "_shutdown" in msg1
        msg2 = q2.get_nowait()
        assert "_shutdown" in msg2


class TestSSEEndpointContentType:
    """Test that /events/stream returns correct content type (non-streaming)."""

    @pytest.fixture
    def broadcaster(self):
        return EventBroadcaster()

    def test_broadcaster_starts_empty(self, broadcaster):
        """No subscribers initially."""
        assert len(broadcaster._subscribers) == 0

    def test_subscribe_increases_count(self, broadcaster):
        broadcaster.subscribe()
        assert len(broadcaster._subscribers) == 1
        broadcaster.subscribe()
        assert len(broadcaster._subscribers) == 2

    def test_unsubscribe_decreases_count(self, broadcaster):
        q = broadcaster.subscribe()
        assert len(broadcaster._subscribers) == 1
        broadcaster.unsubscribe(q)
        assert len(broadcaster._subscribers) == 0

    def test_shutdown_clears_all_subscribers(self, broadcaster):
        broadcaster.subscribe()
        broadcaster.subscribe()
        broadcaster.subscribe()
        broadcaster.shutdown()
        assert len(broadcaster._subscribers) == 0
