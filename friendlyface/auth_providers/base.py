"""Base authentication provider protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class AuthResult:
    """Result of an authentication attempt."""

    authenticated: bool
    identity: str = ""
    provider: str = ""
    roles: list[str] = field(default_factory=list)
    claims: dict = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class AuthProvider(Protocol):
    """Protocol that all auth providers must implement."""

    name: str

    async def authenticate(self, token: str) -> AuthResult:
        """Authenticate a token/key and return an AuthResult."""
        ...
