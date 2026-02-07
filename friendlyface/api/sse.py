"""Server-Sent Events (SSE) broadcaster for real-time forensic event streaming."""

from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBroadcaster:
    """Fan-out broadcaster that pushes event data to all connected SSE clients.

    Each client gets its own :class:`asyncio.Queue` via :meth:`subscribe`.
    :meth:`broadcast` puts data onto every subscriber queue using non-blocking
    ``put_nowait`` so a slow consumer never blocks other clients or the
    producing coroutine.  If a queue is full the message is silently dropped
    for that subscriber (backpressure).

    Thread-safety is ensured via a :class:`threading.Lock` around the
    subscriber set mutations.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._maxsize = maxsize
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Create and register a new subscriber queue."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            self._subscribers.discard(queue)

    def broadcast(self, event_data: dict[str, Any]) -> None:
        """Push *event_data* to every subscriber (non-blocking, drop on full)."""
        with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            try:
                queue.put_nowait(event_data)
            except asyncio.QueueFull:
                # Backpressure: drop the event for this slow subscriber
                pass
