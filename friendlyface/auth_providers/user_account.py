"""User account authentication provider with bcrypt + JWT."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from datetime import datetime, timezone

import jwt

from friendlyface.auth_providers.base import AuthResult

logger = logging.getLogger("friendlyface.auth_providers.user_account")

# JWT config
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days


def _get_jwt_secret() -> str:
    secret = os.environ.get("FF_JWT_SECRET", "")
    if not secret:
        logger.warning("FF_JWT_SECRET not set — using insecure default (dev only)")
        return "ff-dev-secret-do-not-use-in-production"
    return secret


def _hash_password(password: str) -> str:
    """Hash password using PBKDF2-SHA256 (stdlib, no C dependency)."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return f"pbkdf2:sha256:260000${salt}${dk.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify password against PBKDF2-SHA256 hash."""
    try:
        parts = password_hash.split("$")
        if len(parts) != 3:
            return False
        prefix_and_iterations = parts[0]  # pbkdf2:sha256:260000
        salt = parts[1]
        stored_hash = parts[2]
        iterations = int(prefix_and_iterations.split(":")[-1])
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations)
        return hmac.compare_digest(dk.hex(), stored_hash)
    except (ValueError, IndexError):
        return False


def issue_jwt(user_id: str, email: str) -> str:
    """Issue a JWT token for a user."""
    now = time.time()
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(now),
        "exp": int(now + _JWT_EXPIRY_SECONDS),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm=_JWT_ALGORITHM)


def decode_jwt(token: str) -> dict | None:
    """Decode and validate a JWT token. Returns claims or None."""
    try:
        return jwt.decode(token, _get_jwt_secret(), algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


async def register_user(db, email: str, password: str, name: str | None = None) -> dict:
    """Register a new user and create a starter subscription.

    Returns dict with user_id, email, token.
    Raises ValueError if email already taken.
    """
    password_hash = _hash_password(password)
    user_id = secrets.token_hex(16).upper()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    async with db._get_connection() as conn:
        # Check if email exists
        cursor = await conn.execute("SELECT id FROM ff_users WHERE email = ?", (email,))
        existing = await cursor.fetchone()
        if existing:
            raise ValueError("Email already registered")

        # Insert user
        await conn.execute(
            "INSERT INTO ff_users (id, email, password_hash, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, email, password_hash, name, now, now),
        )

        # Create starter subscription
        sub_id = secrets.token_hex(16).upper()
        await conn.execute(
            "INSERT INTO ff_subscriptions (id, user_id, plan, status, created_at, updated_at) VALUES (?, ?, 'starter', 'active', ?, ?)",
            (sub_id, user_id, now, now),
        )
        await conn.commit()

    token = issue_jwt(user_id, email)
    return {"user_id": user_id, "email": email, "token": token}


async def login_user(db, email: str, password: str) -> dict:
    """Authenticate user by email/password and return JWT.

    Returns dict with user_id, email, token.
    Raises ValueError if credentials invalid.
    """
    async with db._get_connection() as conn:
        cursor = await conn.execute(
            "SELECT id, email, password_hash FROM ff_users WHERE email = ?", (email,)
        )
        row = await cursor.fetchone()

    if not row:
        raise ValueError("Invalid email or password")

    user_id, user_email, password_hash = row[0], row[1], row[2]
    if not _verify_password(password, password_hash):
        raise ValueError("Invalid email or password")

    token = issue_jwt(user_id, user_email)
    return {"user_id": user_id, "email": user_email, "token": token}


class UserAccountProvider:
    """Authenticate via JWT tokens issued by the user account system."""

    name = "user_account"

    async def authenticate(self, token: str) -> AuthResult:
        claims = decode_jwt(token)
        if claims is None:
            return AuthResult(
                authenticated=False,
                provider=self.name,
                error="Invalid or expired token",
            )
        return AuthResult(
            authenticated=True,
            identity=claims["sub"],
            provider=self.name,
            roles=["user"],
            claims=claims,
        )
