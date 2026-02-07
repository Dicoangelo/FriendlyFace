"""Factory for creating auth providers based on configuration."""

from __future__ import annotations

import logging

from friendlyface.auth_providers.api_key import ApiKeyProvider
from friendlyface.auth_providers.base import AuthProvider, AuthResult

logger = logging.getLogger("friendlyface.auth_providers.factory")


def create_provider(
    provider_name: str,
    *,
    api_keys: set[str] | None = None,
    key_roles: dict[str, list[str]] | None = None,
    supabase_jwt_secret: str | None = None,
    oidc_issuer: str | None = None,
    oidc_audience: str | None = None,
) -> AuthProvider:
    """Create an auth provider by name."""
    if provider_name == "api_key":
        return ApiKeyProvider(api_keys or set(), key_roles)

    if provider_name == "supabase":
        if not supabase_jwt_secret:
            msg = "supabase_jwt_secret required for supabase auth provider"
            raise ValueError(msg)
        from friendlyface.auth_providers.jwt_provider import SupabaseJWTProvider

        return SupabaseJWTProvider(supabase_jwt_secret)

    if provider_name == "oidc":
        if not oidc_issuer or not oidc_audience:
            msg = "oidc_issuer and oidc_audience required for OIDC auth provider"
            raise ValueError(msg)
        from friendlyface.auth_providers.jwt_provider import OIDCProvider

        return OIDCProvider(oidc_issuer, oidc_audience)

    msg = f"Unknown auth provider: {provider_name}"
    raise ValueError(msg)


class MultiProvider:
    """Try multiple auth providers in order."""

    name = "multi"

    def __init__(self, providers: list[AuthProvider]) -> None:
        self._providers = providers

    async def authenticate(self, token: str) -> AuthResult:
        for provider in self._providers:
            result = await provider.authenticate(token)
            if result.authenticated:
                return result
        return AuthResult(
            authenticated=False,
            provider="multi",
            error="No provider could authenticate the token",
        )
