"""Self-service API key management routes."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from friendlyface.billing.stripe_client import get_user_subscription
from friendlyface.billing.gate import PLAN_RATE_LIMITS

router = APIRouter(prefix="/keys", tags=["api-keys"])


class CreateKeyRequest(BaseModel):
    name: str = "default"


class CreateKeyResponse(BaseModel):
    id: str
    key: str
    name: str
    rate_limit: int
    created_at: str


class KeyInfo(BaseModel):
    id: str
    name: str | None
    rate_limit: int
    created_at: str
    revoked_at: str | None


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@router.post("", response_model=CreateKeyResponse, status_code=201)
async def create_key(req: CreateKeyRequest, request: Request):
    """Generate a new API key for the authenticated user."""
    auth = getattr(request.state, "auth", None)
    if not auth or not auth.authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user_id = auth.identity
    db = request.app.state.db

    # Determine rate limit from subscription plan
    sub = await get_user_subscription(db, user_id)
    plan = sub["plan"] if sub else "starter"
    rate_limit = PLAN_RATE_LIMITS.get(plan, 100)

    key_id = secrets.token_hex(8)
    raw_key = f"ff_{secrets.token_urlsafe(32)}"
    key_hash = _hash_key(raw_key)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    async with db._get_connection() as conn:
        await conn.execute(
            "INSERT INTO ff_api_keys (id, user_id, key_hash, name, rate_limit, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (key_id, user_id, key_hash, req.name, rate_limit, now),
        )
        await conn.commit()

    return CreateKeyResponse(
        id=key_id,
        key=raw_key,
        name=req.name,
        rate_limit=rate_limit,
        created_at=now,
    )


@router.get("", response_model=list[KeyInfo])
async def list_keys(request: Request):
    """List all API keys for the authenticated user (never shows the raw key)."""
    auth = getattr(request.state, "auth", None)
    if not auth or not auth.authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    db = request.app.state.db
    async with db._get_connection() as conn:
        cursor = await conn.execute(
            "SELECT id, name, rate_limit, created_at, revoked_at FROM ff_api_keys WHERE user_id = ? ORDER BY created_at DESC",
            (auth.identity,),
        )
        rows = await cursor.fetchall()

    return [
        KeyInfo(
            id=row[0],
            name=row[1],
            rate_limit=row[2],
            created_at=row[3],
            revoked_at=row[4],
        )
        for row in rows
    ]


@router.delete("/{key_id}", status_code=204)
async def revoke_key(key_id: str, request: Request):
    """Revoke an API key (sets revoked_at timestamp)."""
    auth = getattr(request.state, "auth", None)
    if not auth or not auth.authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    db = request.app.state.db
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    async with db._get_connection() as conn:
        cursor = await conn.execute(
            "UPDATE ff_api_keys SET revoked_at = ? WHERE id = ? AND user_id = ? AND revoked_at IS NULL",
            (now, key_id, auth.identity),
        )
        await conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Key not found or already revoked")
