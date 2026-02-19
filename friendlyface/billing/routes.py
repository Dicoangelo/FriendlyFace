"""Billing routes for Stripe integration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import stripe
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from friendlyface.billing.stripe_client import (
    create_checkout_session,
    create_customer,
    create_portal_session,
    get_user_subscription,
    get_webhook_secret,
)

logger = logging.getLogger("friendlyface.billing")

router = APIRouter(prefix="/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    plan: str = "professional"
    success_url: str = "/pricing?success=true"
    cancel_url: str = "/pricing?canceled=true"


class PortalRequest(BaseModel):
    return_url: str = "/"


@router.post("/checkout")
async def create_checkout(req: CheckoutRequest, request: Request):
    """Create a Stripe Checkout session for plan upgrade."""
    auth = getattr(request.state, "auth", None)
    if not auth or not auth.authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user_id = auth.identity
    db = request.app.state.db

    sub = await get_user_subscription(db, user_id)
    if not sub:
        raise HTTPException(status_code=404, detail="No subscription found")

    customer_id = sub.get("stripe_customer_id")
    if not customer_id:
        # Look up email and create customer
        async with db._get_connection() as conn:
            cursor = await conn.execute("SELECT email, name FROM ff_users WHERE id = ?", (user_id,))
            row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        customer_id = await create_customer(row[0], row[1])
        async with db._get_connection() as conn:
            await conn.execute(
                "UPDATE ff_subscriptions SET stripe_customer_id = ?, updated_at = ? WHERE user_id = ?",
                (customer_id, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), user_id),
            )
            await conn.commit()

    url = await create_checkout_session(
        customer_id, req.plan, req.success_url, req.cancel_url
    )
    return {"checkout_url": url}


@router.post("/portal")
async def billing_portal(req: PortalRequest, request: Request):
    """Create a Stripe Billing Portal session."""
    auth = getattr(request.state, "auth", None)
    if not auth or not auth.authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    user_id = auth.identity
    db = request.app.state.db

    sub = await get_user_subscription(db, user_id)
    if not sub or not sub.get("stripe_customer_id"):
        raise HTTPException(status_code=404, detail="No Stripe customer found")

    url = await create_portal_session(sub["stripe_customer_id"], req.return_url)
    return {"portal_url": url}


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    body = await request.body()
    sig = request.headers.get("stripe-signature", "")
    webhook_secret = get_webhook_secret()

    if not webhook_secret:
        logger.error("FF_STRIPE_WEBHOOK_SECRET not configured")
        raise HTTPException(status_code=500, detail="Webhook not configured")

    try:
        event = stripe.Webhook.construct_event(body, sig, webhook_secret)
    except stripe.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error("Webhook error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid payload")

    db = request.app.state.db
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")
        if customer_id and subscription_id:
            async with db._get_connection() as conn:
                await conn.execute(
                    "UPDATE ff_subscriptions SET stripe_subscription_id = ?, status = 'active', updated_at = ? WHERE stripe_customer_id = ?",
                    (subscription_id, now, customer_id),
                )
                await conn.commit()
            logger.info("Checkout completed for customer %s", customer_id)

    elif event["type"] == "customer.subscription.updated":
        sub = event["data"]["object"]
        customer_id = sub.get("customer")
        sub_status = sub.get("status", "active")
        period_end = sub.get("current_period_end")
        period_end_str = (
            datetime.fromtimestamp(period_end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if period_end
            else None
        )

        # Map Stripe plan to our tiers
        plan = "starter"
        items = sub.get("items", {}).get("data", [])
        if items:
            amount = items[0].get("price", {}).get("unit_amount", 0)
            if amount >= 49900:
                plan = "enterprise"
            elif amount >= 19900:
                plan = "professional"

        status_map = {"active": "active", "past_due": "past_due", "canceled": "canceled", "trialing": "trialing"}
        mapped_status = status_map.get(sub_status, "active")

        async with db._get_connection() as conn:
            await conn.execute(
                "UPDATE ff_subscriptions SET plan = ?, status = ?, current_period_end = ?, updated_at = ? WHERE stripe_customer_id = ?",
                (plan, mapped_status, period_end_str, now, customer_id),
            )
            await conn.commit()
        logger.info("Subscription updated for customer %s: %s/%s", customer_id, plan, mapped_status)

    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub.get("customer")
        async with db._get_connection() as conn:
            await conn.execute(
                "UPDATE ff_subscriptions SET plan = 'starter', status = 'canceled', updated_at = ? WHERE stripe_customer_id = ?",
                (now, customer_id),
            )
            await conn.commit()
        logger.info("Subscription canceled for customer %s", customer_id)

    return {"received": True}


@router.get("/subscription")
async def get_subscription(request: Request):
    """Get current user's subscription details."""
    auth = getattr(request.state, "auth", None)
    if not auth or not auth.authenticated:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    db = request.app.state.db
    sub = await get_user_subscription(db, auth.identity)
    if not sub:
        return {"plan": "starter", "status": "active"}
    return {
        "plan": sub["plan"],
        "status": sub["status"],
        "current_period_end": sub["current_period_end"],
    }
