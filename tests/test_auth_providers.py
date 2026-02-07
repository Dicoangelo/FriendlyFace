"""Tests for pluggable authentication providers (US-037)."""

from __future__ import annotations

import pytest

from friendlyface.auth_providers.base import AuthProvider, AuthResult
from friendlyface.auth_providers.api_key import ApiKeyProvider
from friendlyface.auth_providers.factory import MultiProvider, create_provider
from friendlyface.auth_providers.jwt_provider import OIDCProvider, SupabaseJWTProvider


# ---------------------------------------------------------------------------
# AuthResult
# ---------------------------------------------------------------------------


class TestAuthResult:
    def test_defaults(self):
        r = AuthResult(authenticated=False)
        assert r.authenticated is False
        assert r.identity == ""
        assert r.provider == ""
        assert r.roles == []
        assert r.claims == {}
        assert r.error is None

    def test_full(self):
        r = AuthResult(
            authenticated=True,
            identity="user:1",
            provider="test",
            roles=["admin"],
            claims={"sub": "1"},
            error=None,
        )
        assert r.authenticated is True
        assert r.identity == "user:1"
        assert r.roles == ["admin"]
        assert r.claims == {"sub": "1"}


# ---------------------------------------------------------------------------
# ApiKeyProvider
# ---------------------------------------------------------------------------


class TestApiKeyProvider:
    @pytest.fixture
    def provider(self):
        return ApiKeyProvider(
            {"secret-key-1", "secret-key-2"},
            {"secret-key-1": ["admin"], "secret-key-2": ["viewer"]},
        )

    async def test_valid_key(self, provider):
        result = await provider.authenticate("secret-key-1")
        assert result.authenticated is True
        assert result.provider == "api_key"
        assert "admin" in result.roles
        assert result.identity.startswith("api_key:")

    async def test_valid_key_viewer_role(self, provider):
        result = await provider.authenticate("secret-key-2")
        assert result.authenticated is True
        assert "viewer" in result.roles

    async def test_invalid_key(self, provider):
        result = await provider.authenticate("wrong-key")
        assert result.authenticated is False
        assert result.error == "Invalid API key"

    async def test_empty_keys_all_rejected(self):
        provider = ApiKeyProvider(set())
        result = await provider.authenticate("any-key")
        assert result.authenticated is False

    async def test_default_admin_role_when_no_mapping(self):
        provider = ApiKeyProvider({"mykey"})
        result = await provider.authenticate("mykey")
        assert result.authenticated is True
        assert result.roles == ["admin"]

    def test_name_attribute(self, provider):
        assert provider.name == "api_key"

    def test_conforms_to_protocol(self, provider):
        assert isinstance(provider, AuthProvider)


# ---------------------------------------------------------------------------
# SupabaseJWTProvider
# ---------------------------------------------------------------------------


class TestSupabaseJWTProvider:
    @pytest.fixture
    def provider(self):
        return SupabaseJWTProvider("test-secret-key-for-jwt-signing")

    async def test_valid_token(self, provider):
        import jwt

        token = jwt.encode(
            {"sub": "user-123", "role": "authenticated", "aud": "authenticated"},
            "test-secret-key-for-jwt-signing",
            algorithm="HS256",
        )
        result = await provider.authenticate(token)
        assert result.authenticated is True
        assert result.identity == "supabase:user-123"
        assert "authenticated" in result.roles
        assert result.claims["sub"] == "user-123"

    async def test_invalid_token(self, provider):
        result = await provider.authenticate("garbage-token")
        assert result.authenticated is False
        assert "JWT validation failed" in result.error

    async def test_wrong_secret(self):
        import jwt

        token = jwt.encode(
            {"sub": "user", "role": "auth", "aud": "authenticated"},
            "correct-secret",
            algorithm="HS256",
        )
        provider = SupabaseJWTProvider("wrong-secret")
        result = await provider.authenticate(token)
        assert result.authenticated is False

    async def test_expired_token(self, provider):
        import jwt
        import time

        token = jwt.encode(
            {"sub": "user", "role": "auth", "aud": "authenticated", "exp": int(time.time()) - 3600},
            "test-secret-key-for-jwt-signing",
            algorithm="HS256",
        )
        result = await provider.authenticate(token)
        assert result.authenticated is False

    def test_name(self, provider):
        assert provider.name == "supabase"


# ---------------------------------------------------------------------------
# OIDCProvider (US-077: JWKS-based signature verification)
# ---------------------------------------------------------------------------


def _make_rsa_key_and_token(claims: dict):
    """Helper: generate an RSA key pair and sign a JWT token."""
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    token = jwt.encode(claims, private_key, algorithm="RS256")
    return private_key, token


class _FakeSigningKey:
    """Mimics jwt.PyJWK with a .key attribute."""

    def __init__(self, key):
        self.key = key


class _FakeJWKSClient:
    """Replaces jwt.PyJWKClient in tests to avoid real HTTP."""

    def __init__(self, public_key):
        self._public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._public_key)


class _FailingJWKSClient:
    """JWKS client that always fails (simulates network error)."""

    def get_signing_key_from_jwt(self, token):
        raise ConnectionError("JWKS endpoint unreachable")


class TestOIDCProvider:
    @pytest.fixture
    def provider(self):
        return OIDCProvider("https://issuer.example.com", "my-audience")

    async def test_valid_rs256_token(self, provider):
        import time

        private_key, token = _make_rsa_key_and_token(
            {
                "sub": "oidc-user",
                "iss": "https://issuer.example.com",
                "aud": "my-audience",
                "exp": int(time.time()) + 3600,
                "roles": ["analyst"],
            }
        )
        public_key = private_key.public_key()
        provider._jwks_client = _FakeJWKSClient(public_key)

        result = await provider.authenticate(token)
        assert result.authenticated is True
        assert result.identity == "oidc:oidc-user"
        assert "analyst" in result.roles

    async def test_invalid_signature_rejected(self, provider):
        """Token signed with wrong key is rejected."""
        import time

        _, token = _make_rsa_key_and_token(
            {
                "sub": "user",
                "iss": "https://issuer.example.com",
                "aud": "my-audience",
                "exp": int(time.time()) + 3600,
            }
        )
        # Use a DIFFERENT key for verification → signature mismatch
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        other_key = rsa.generate_private_key(65537, 2048, default_backend()).public_key()
        provider._jwks_client = _FakeJWKSClient(other_key)

        result = await provider.authenticate(token)
        assert result.authenticated is False
        assert "OIDC validation failed" in result.error

    async def test_wrong_issuer_rejected(self, provider):
        import time

        private_key, token = _make_rsa_key_and_token(
            {
                "sub": "user",
                "iss": "https://other.example.com",
                "aud": "my-audience",
                "exp": int(time.time()) + 3600,
            }
        )
        provider._jwks_client = _FakeJWKSClient(private_key.public_key())

        result = await provider.authenticate(token)
        assert result.authenticated is False

    async def test_wrong_audience_rejected(self, provider):
        import time

        private_key, token = _make_rsa_key_and_token(
            {
                "sub": "user",
                "iss": "https://issuer.example.com",
                "aud": "wrong-audience",
                "exp": int(time.time()) + 3600,
            }
        )
        provider._jwks_client = _FakeJWKSClient(private_key.public_key())

        result = await provider.authenticate(token)
        assert result.authenticated is False

    async def test_expired_token_rejected(self, provider):
        import time

        private_key, token = _make_rsa_key_and_token(
            {
                "sub": "user",
                "iss": "https://issuer.example.com",
                "aud": "my-audience",
                "exp": int(time.time()) - 3600,
            }
        )
        provider._jwks_client = _FakeJWKSClient(private_key.public_key())

        result = await provider.authenticate(token)
        assert result.authenticated is False

    async def test_jwks_fetch_failure_returns_unauthenticated(self, provider):
        provider._jwks_client = _FailingJWKSClient()
        result = await provider.authenticate("some.jwt.token")
        assert result.authenticated is False
        assert "JWKS verification failed" in result.error

    async def test_invalid_token(self, provider):
        provider._jwks_client = _FailingJWKSClient()
        result = await provider.authenticate("not-a-jwt")
        assert result.authenticated is False

    async def test_roles_default_empty(self, provider):
        import time

        private_key, token = _make_rsa_key_and_token(
            {
                "sub": "user",
                "iss": "https://issuer.example.com",
                "aud": "my-audience",
                "exp": int(time.time()) + 3600,
            }
        )
        provider._jwks_client = _FakeJWKSClient(private_key.public_key())

        result = await provider.authenticate(token)
        assert result.authenticated is True
        assert result.roles == []

    def test_name(self, provider):
        assert provider.name == "oidc"


# ---------------------------------------------------------------------------
# Factory — create_provider()
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_api_key_provider(self):
        p = create_provider("api_key", api_keys={"k1"})
        assert isinstance(p, ApiKeyProvider)

    def test_create_supabase_provider(self):
        p = create_provider("supabase", supabase_jwt_secret="secret123")
        assert isinstance(p, SupabaseJWTProvider)

    def test_create_supabase_missing_secret(self):
        with pytest.raises(ValueError, match="supabase_jwt_secret"):
            create_provider("supabase")

    def test_create_oidc_provider(self):
        p = create_provider("oidc", oidc_issuer="https://ex.com", oidc_audience="aud")
        assert isinstance(p, OIDCProvider)

    def test_create_oidc_missing_issuer(self):
        with pytest.raises(ValueError, match="oidc_issuer"):
            create_provider("oidc", oidc_audience="aud")

    def test_create_oidc_missing_audience(self):
        with pytest.raises(ValueError, match="oidc_issuer"):
            create_provider("oidc", oidc_issuer="https://ex.com")

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown auth provider"):
            create_provider("ldap")


# ---------------------------------------------------------------------------
# MultiProvider
# ---------------------------------------------------------------------------


class TestMultiProvider:
    async def test_first_provider_wins(self):
        p1 = ApiKeyProvider({"key-a"})
        p2 = ApiKeyProvider({"key-b"})
        multi = MultiProvider([p1, p2])
        result = await multi.authenticate("key-a")
        assert result.authenticated is True
        assert result.provider == "api_key"

    async def test_second_provider_fallback(self):
        p1 = ApiKeyProvider(set())  # always rejects
        p2 = ApiKeyProvider({"key-b"})
        multi = MultiProvider([p1, p2])
        result = await multi.authenticate("key-b")
        assert result.authenticated is True

    async def test_all_reject(self):
        p1 = ApiKeyProvider(set())
        p2 = ApiKeyProvider(set())
        multi = MultiProvider([p1, p2])
        result = await multi.authenticate("any")
        assert result.authenticated is False
        assert result.provider == "multi"
        assert "No provider" in result.error

    def test_name_attribute(self):
        multi = MultiProvider([])
        assert multi.name == "multi"
