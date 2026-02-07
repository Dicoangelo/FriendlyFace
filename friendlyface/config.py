"""Centralized configuration for FriendlyFace.

Uses Pydantic BaseSettings with environment variable loading and validation.
All FF_* environment variables are validated at import time.
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": "FF_", "case_sensitive": False}

    # Storage
    storage: str = Field(default="sqlite", description="Storage backend: sqlite or supabase")
    db_path: str = Field(default="friendlyface.db", description="SQLite database path")

    # Auth
    api_keys: str = Field(default="", description="Comma-separated API keys (empty = dev mode)")
    auth_provider: str = Field(
        default="api_key", description="Auth provider: api_key, supabase, oidc"
    )
    supabase_jwt_secret: str | None = Field(default=None, description="Supabase JWT secret")
    oidc_issuer: str | None = Field(default=None, description="OIDC issuer URL")
    oidc_audience: str | None = Field(default=None, description="OIDC audience")
    api_key_roles: str = Field(
        default="", description='JSON map of api_key -> roles (e.g., \'{"key1": ["admin"]}\')'
    )

    # Crypto
    did_seed: str | None = Field(
        default=None, description="64-char hex seed for deterministic DID key"
    )

    # Logging
    log_format: str = Field(default="text", description="Log format: text or json")
    log_level: str = Field(default="INFO", description="Python log level")

    # Server
    host: str = Field(default="0.0.0.0", description="Server bind host")  # noqa: S104
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")

    # Frontend
    serve_frontend: bool = Field(default=True, description="Serve React dashboard static files")

    # CORS
    cors_origins: str = Field(default="*", description="Comma-separated CORS origins")

    # Federated Learning
    fl_mode: str = Field(
        default="simulation",
        description="FL operation mode: simulation or production",
    )

    # Rate limiting
    rate_limit: str = Field(
        default="100/minute",
        description="Default rate limit (e.g., 100/minute). Set to 'none' to disable.",
    )

    # Migrations
    migrations_enabled: bool = Field(default=False, description="Run SQL migrations on startup")

    # Merkle
    merkle_checkpoint_interval: int = Field(
        default=100, ge=1, description="Save Merkle checkpoint every N events"
    )

    # Backup
    backup_dir: str = Field(default="backups", description="Backup directory path")
    backup_interval_minutes: int = Field(
        default=60, ge=1, description="Auto-backup interval in minutes"
    )

    # Supabase
    supabase_url: str | None = Field(default=None, description="Supabase project URL")
    supabase_key: str | None = Field(default=None, description="Supabase service role key")

    model_config = {"env_prefix": "FF_", "case_sensitive": False, "extra": "ignore"}

    @field_validator("storage")
    @classmethod
    def validate_storage(cls, v: str) -> str:
        v = v.lower()
        if v not in ("sqlite", "supabase"):
            msg = f"FF_STORAGE must be 'sqlite' or 'supabase', got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("fl_mode")
    @classmethod
    def validate_fl_mode(cls, v: str) -> str:
        v = v.lower()
        if v not in ("simulation", "production"):
            msg = f"FF_FL_MODE must be 'simulation' or 'production', got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        v = v.lower()
        if v not in ("text", "json"):
            msg = f"FF_LOG_FORMAT must be 'text' or 'json', got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        import logging

        v = v.upper()
        if not hasattr(logging, v):
            msg = f"FF_LOG_LEVEL must be a valid Python log level, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("did_seed")
    @classmethod
    def validate_did_seed(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 64:
            msg = f"FF_DID_SEED must be exactly 64 hex characters, got {len(v)}"
            raise ValueError(msg)
        try:
            bytes.fromhex(v)
        except ValueError:
            msg = "FF_DID_SEED must be valid hexadecimal"
            raise ValueError(msg)  # noqa: B904
        return v

    @property
    def api_key_set(self) -> set[str]:
        """Return parsed set of API keys."""
        if not self.api_keys.strip():
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        """Return parsed list of CORS origins."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Singleton — validated at import time.
# Supabase vars use SUPABASE_ prefix, not FF_, so set extra="ignore".
settings = Settings()
