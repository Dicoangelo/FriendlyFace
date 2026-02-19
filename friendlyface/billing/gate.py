"""Feature gating based on subscription plan."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from friendlyface.billing.stripe_client import get_user_subscription

# Plan hierarchy for comparison
PLAN_RANK = {"starter": 0, "professional": 1, "enterprise": 2}

# Rate limits per plan
PLAN_RATE_LIMITS = {
    "starter": 100,
    "professional": 500,
    "enterprise": 2000,
}


def require_plan(*plans: str):
    """FastAPI dependency factory: require user to have one of the given plans.

    Usage::

        @app.post("/fl/start", dependencies=[Depends(require_plan("professional", "enterprise"))])
        async def start_fl(): ...
    """

    async def _check(request: Request) -> None:
        auth = getattr(request.state, "auth", None)
        if not auth or not auth.authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        # Dev mode and admin roles bypass plan checks
        if auth.provider == "dev" or "admin" in auth.roles:
            return

        db = request.app.state.db
        sub = await get_user_subscription(db, auth.identity)

        user_plan = sub["plan"] if sub else "starter"
        if user_plan not in plans:
            min_required = min(plans, key=lambda p: PLAN_RANK.get(p, 99))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires the {min_required} plan or higher. Your plan: {user_plan}.",
            )

    return _check
