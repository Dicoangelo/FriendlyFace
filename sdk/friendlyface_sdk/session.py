"""Forensic session context manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from friendlyface_sdk.models import Bundle, ForensicEvent

if TYPE_CHECKING:
    from friendlyface_sdk.client import FriendlyFaceClient


class ForensicSession:
    """Context manager that tracks events and auto-bundles on exit.

    Usage::

        with client.forensic_session() as session:
            session.log_event("inference_request", {"actor": "sdk", "model": "v1"})
            session.log_event("inference_result", {"actor": "sdk", "result": "ok"})
        # bundle is created automatically on exit
    """

    def __init__(self, client: "FriendlyFaceClient") -> None:
        self._client = client
        self._event_ids: List[str] = []
        self.bundle: Bundle | None = None

    def __enter__(self) -> "ForensicSession":
        self._event_ids = []
        self.bundle = None
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._event_ids:
            self.bundle = self._client.create_bundle(self._event_ids)

    def log_event(
        self,
        event_type: str,
        content: dict,
        metadata: dict | None = None,
    ) -> ForensicEvent:
        """Log a forensic event and track its ID for bundling."""
        event = self._client.log_event(event_type, content, metadata)
        self._event_ids.append(event.event_id)
        return event

    @property
    def event_ids(self) -> List[str]:
        """Event IDs collected during this session."""
        return list(self._event_ids)
