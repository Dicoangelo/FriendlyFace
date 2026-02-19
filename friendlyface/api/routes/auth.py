"""Auth routes for user registration and login."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, EmailStr

from friendlyface.auth_providers.user_account import login_user, register_user

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    user_id: str
    email: str
    token: str


class UserInfo(BaseModel):
    user_id: str
    email: str
    name: str | None
    plan: str
    created_at: str


@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(req: RegisterRequest, request: Request):
    """Register a new user account with a starter subscription."""
    db = request.app.state.db
    try:
        result = await register_user(db, req.email, req.password, req.name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    return AuthResponse(**result)


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest, request: Request):
    """Login with email and password, returns JWT."""
    db = request.app.state.db
    try:
        result = await login_user(db, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    return AuthResponse(**result)


@router.get("/me", response_model=UserInfo)
async def get_me(request: Request):
    """Get current authenticated user info."""
    auth = getattr(request.state, "auth", None)
    if not auth or not auth.authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    db = request.app.state.db
    async with db._get_connection() as conn:
        cursor = await conn.execute(
            "SELECT u.id, u.email, u.name, u.created_at, COALESCE(s.plan, 'starter') as plan "
            "FROM ff_users u LEFT JOIN ff_subscriptions s ON u.id = s.user_id WHERE u.id = ?",
            (auth.identity,),
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return UserInfo(
        user_id=row[0],
        email=row[1],
        name=row[2],
        plan=row[4],
        created_at=row[3],
    )
