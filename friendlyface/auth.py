"""API key authentication for FriendlyFace.

Authentication is controlled by the ``FF_API_KEYS`` environment variable:
- **Not set / empty** -> dev mode, all requests pass through without auth.
- **Set** (comma-separated list of valid keys) -> auth is enforced on every
  endpoint except explicitly excluded paths (e.g. ``/health``).

Clients supply their key via the ``X-API-Key`` header **or** the ``api_key``
query parameter.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, Request, status

# Paths that are always public, even when auth is enabled.
PUBLIC_PATHS: frozenset[str] = frozenset({"/health"})


def _load_api_keys() -> set[str]:
    """Return the set of valid API keys from the environment.

    If ``FF_API_KEYS`` is unset or empty, returns an empty set which
    signals that authentication is disabled (dev mode).
    """
    raw = os.environ.get("FF_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(request: Request) -> None:
    """FastAPI dependency that enforces API-key authentication.

    Usage::

        app = FastAPI(dependencies=[Depends(require_api_key)])

    Behaviour:
    * If ``FF_API_KEYS`` is not set or empty, authentication is **disabled**
      and every request is allowed (dev mode).
    * Requests to paths listed in ``PUBLIC_PATHS`` are always allowed.
    * Otherwise the caller must present a valid key via the ``X-API-Key``
      header or the ``api_key`` query parameter.

    Raises:
        HTTPException 403: auth is enabled but no key was provided.
        HTTPException 401: a key was provided but it is not valid.
    """
    # Always allow public paths.
    if request.url.path in PUBLIC_PATHS:
        return

    valid_keys = _load_api_keys()

    # Dev mode: no keys configured -> auth is disabled.
    if not valid_keys:
        return

    # Try header first, then query parameter.
    api_key: str | None = request.headers.get("X-API-Key")
    if api_key is None:
        api_key = request.query_params.get("api_key")

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key required. Provide via X-API-Key header or api_key query parameter.",
        )

    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
