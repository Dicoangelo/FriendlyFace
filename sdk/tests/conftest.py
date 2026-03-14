"""Shared fixtures for SDK tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from friendlyface_sdk.client import FriendlyFaceClient


@pytest.fixture()
def mock_session():
    """Patch requests.Session and return the mock instance."""
    with patch("friendlyface_sdk.client.requests.Session") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture()
def client(mock_session):
    """Return a FriendlyFaceClient wired to a mocked requests.Session."""
    return FriendlyFaceClient("http://localhost:8000", api_key="test-key")
