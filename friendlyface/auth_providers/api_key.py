"""API key authentication provider (refactored from auth.py)."""

from __future__ import annotations

from friendlyface.auth_providers.base import AuthResult


class ApiKeyProvider:
    """Authenticate via static API keys from FF_API_KEYS."""

    name = "api_key"

    def __init__(self, valid_keys: set[str], key_roles: dict[str, list[str]] | None = None) -> None:
        self._valid_keys = valid_keys
        self._key_roles = key_roles or {}

    async def authenticate(self, token: str) -> AuthResult:
        if token in self._valid_keys:
            roles = self._key_roles.get(token, ["admin"])
            return AuthResult(
                authenticated=True,
                identity=f"api_key:{token[:8]}...",
                provider=self.name,
                roles=roles,
            )
        return AuthResult(
            authenticated=False,
            provider=self.name,
            error="Invalid API key",
        )
