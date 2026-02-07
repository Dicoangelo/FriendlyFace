"""API key and multi-provider authentication for FriendlyFace.

Authentication is controlled by two environment variables:
- ``FF_API_KEYS`` — comma-separated list of valid API keys. When empty,
  auth is **disabled** (dev mode).
- ``FF_AUTH_PROVIDER`` — selects the backend: ``api_key`` (default),
  ``supabase``, or ``oidc``.

Clients supply credentials via:
- ``Authorization: Bearer <token>`` header (preferred)
- ``X-API-Key`` header (legacy, api_key provider only)
- ``api_key`` query parameter (legacy, api_key provider only)
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import HTTPException, Request, status

from friendlyface.auth_providers.base import AuthResult
from friendlyface.auth_providers.factory import create_provider
from friendlyface.config import settings

# Paths that are always public, even when auth is enabled.
PUBLIC_PATHS: frozenset[str] = frozenset({"/health", "/metrics"})

_audit_logger = logging.getLogger("friendlyface.audit")


def _extract_token(request: Request) -> str | None:
    """Extract auth token from request headers or query params.

    Priority: Authorization Bearer > X-API-Key header > api_key query param.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    api_key = request.headers.get("X-API-Key")
    if api_key is not None:
        return api_key

    return request.query_params.get("api_key")


def _parse_key_roles(raw: str) -> dict[str, list[str]]:
    """Parse FF_API_KEY_ROLES JSON string into a dict."""
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


async def require_api_key(request: Request) -> None:
    """FastAPI dependency that enforces authentication.

    Delegates to the configured auth provider via ``FF_AUTH_PROVIDER``.

    Behaviour:
    * Requests to paths listed in ``PUBLIC_PATHS`` are always allowed.
    * If ``FF_API_KEYS`` is not set or empty **and** the provider is
      ``api_key``, authentication is **disabled** (dev mode).
    * Otherwise the caller must present a valid token.

    The :class:`AuthResult` is attached to ``request.state.auth`` so
    downstream handlers can inspect identity and roles.

    Raises:
        HTTPException 403: auth is enabled but no token was provided.
        HTTPException 401: a token was provided but it is not valid.
    """
    # Always allow public paths.
    if request.url.path in PUBLIC_PATHS:
        return

    # Read config — use os.environ so monkeypatch works in tests.
    api_keys_raw = os.environ.get("FF_API_KEYS", settings.api_keys)
    valid_keys = frozenset(k.strip() for k in api_keys_raw.split(",") if k.strip())
    provider_name = os.environ.get("FF_AUTH_PROVIDER", settings.auth_provider)

    # Dev mode: api_key provider with no keys configured -> auth disabled.
    if provider_name == "api_key" and not valid_keys:
        request.state.auth = AuthResult(
            authenticated=True, identity="dev", provider="dev", roles=["admin"]
        )
        return

    key_roles = _parse_key_roles(os.environ.get("FF_API_KEY_ROLES", settings.api_key_roles))

    provider = create_provider(
        provider_name,
        api_keys=valid_keys if valid_keys else None,
        key_roles=key_roles if key_roles else None,
        supabase_jwt_secret=os.environ.get("FF_SUPABASE_JWT_SECRET", settings.supabase_jwt_secret),
        oidc_issuer=os.environ.get("FF_OIDC_ISSUER", settings.oidc_issuer),
        oidc_audience=os.environ.get("FF_OIDC_AUDIENCE", settings.oidc_audience),
    )

    token = _extract_token(request)

    if token is None:
        _audit_logger.warning(
            "Auth failure (no token): %s %s from %s",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
            extra={
                "event_category": "audit",
                "action": "auth_failure",
                "reason": "no_token",
                "path": request.url.path,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key required. Provide via X-API-Key header or api_key query parameter.",
        )

    result = await provider.authenticate(token)
    if not result.authenticated:
        _audit_logger.warning(
            "Auth failure (invalid token): %s %s from %s",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
            extra={
                "event_category": "audit",
                "action": "auth_failure",
                "reason": "invalid_token",
                "path": request.url.path,
                "provider": result.provider,
                "error": result.error,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    # Attach result so downstream can inspect identity/roles.
    request.state.auth = result


def require_role(*roles: str):
    """Dependency factory: require the authenticated user to have one of the given roles.

    Usage::

        @app.post("/admin/thing", dependencies=[Depends(require_role("admin"))])
        async def admin_thing(): ...
    """

    async def _check(request: Request) -> None:
        auth: AuthResult | None = getattr(request.state, "auth", None)
        if auth is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated.")
        # Dev mode always passes role checks.
        if auth.provider == "dev":
            return
        if not any(r in auth.roles for r in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(roles)}",
            )

    return _check
