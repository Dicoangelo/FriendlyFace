"""JWT-based authentication providers (Supabase, generic OIDC)."""

from __future__ import annotations

import logging

from friendlyface.auth_providers.base import AuthResult

logger = logging.getLogger("friendlyface.auth_providers.jwt")


class SupabaseJWTProvider:
    """Authenticate via Supabase JWT tokens."""

    name = "supabase"

    def __init__(self, jwt_secret: str) -> None:
        self._jwt_secret = jwt_secret

    async def authenticate(self, token: str) -> AuthResult:
        try:
            import jwt

            payload = jwt.decode(
                token, self._jwt_secret, algorithms=["HS256"], audience="authenticated"
            )
            sub = payload.get("sub", "")
            role = payload.get("role", "authenticated")
            return AuthResult(
                authenticated=True,
                identity=f"supabase:{sub}",
                provider=self.name,
                roles=[role],
                claims=payload,
            )
        except ImportError:
            return AuthResult(
                authenticated=False,
                provider=self.name,
                error="pyjwt not installed (pip install pyjwt[crypto])",
            )
        except Exception as e:
            return AuthResult(
                authenticated=False,
                provider=self.name,
                error=f"JWT validation failed: {e}",
            )


class OIDCProvider:
    """Authenticate via generic OIDC JWT tokens with JWKS signature verification.

    Uses ``jwt.PyJWKClient`` to fetch and cache JWKS from the issuer's
    ``/.well-known/openid-configuration`` endpoint. RS256 signatures are
    verified against the published keys.
    """

    name = "oidc"

    def __init__(self, issuer: str, audience: str) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._jwks_client: object | None = None

    def _get_jwks_client(self):
        """Lazily create and cache the JWKS client (1-hour TTL)."""
        if self._jwks_client is None:
            import jwt

            jwks_url = f"{self._issuer}/.well-known/jwks.json"
            self._jwks_client = jwt.PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=3600)
        return self._jwks_client

    async def authenticate(self, token: str) -> AuthResult:
        try:
            import jwt

            # Fetch signing key from JWKS endpoint
            jwks_client = self._get_jwks_client()
            try:
                signing_key = jwks_client.get_signing_key_from_jwt(token)
            except Exception as e:
                logger.warning("JWKS fetch/lookup failed: %s", e)
                return AuthResult(
                    authenticated=False,
                    provider=self.name,
                    error=f"JWKS verification failed: {e}",
                )

            # Full decode with signature verification
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["sub", "iss", "exp"]},
            )

            sub = payload.get("sub", "")
            roles = payload.get("roles", [])
            return AuthResult(
                authenticated=True,
                identity=f"oidc:{sub}",
                provider=self.name,
                roles=roles,
                claims=payload,
            )
        except ImportError:
            return AuthResult(
                authenticated=False,
                provider=self.name,
                error="pyjwt not installed",
            )
        except Exception as e:
            return AuthResult(
                authenticated=False,
                provider=self.name,
                error=f"OIDC validation failed: {e}",
            )
