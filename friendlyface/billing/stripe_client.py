"""Stripe client wrapper for FriendlyFace billing."""

from __future__ import annotations

import logging
import os

import stripe

logger = logging.getLogger("friendlyface.billing.stripe")

# Plan price IDs — set via environment or use test defaults
PLAN_PRICES = {
    "starter": os.environ.get("FF_STRIPE_PRICE_STARTER", "price_starter"),
    "professional": os.environ.get("FF_STRIPE_PRICE_PROFESSIONAL", "price_professional"),
    "enterprise": os.environ.get("FF_STRIPE_PRICE_ENTERPRISE", "price_enterprise"),
}

PLAN_AMOUNTS = {
    "starter": 4900,       # $49/mo
    "professional": 19900,  # $199/mo
    "enterprise": 49900,    # $499/mo
}


def _get_stripe():
    """Initialize stripe with the secret key."""
    key = os.environ.get("FF_STRIPE_SECRET_KEY", "")
    if not key:
        logger.warning("FF_STRIPE_SECRET_KEY not set — Stripe calls will fail")
    stripe.api_key = key
    return stripe


def get_webhook_secret() -> str:
    return os.environ.get("FF_STRIPE_WEBHOOK_SECRET", "")


async def create_customer(email: str, name: str | None = None) -> str:
    """Create a Stripe customer and return the customer ID."""
    s = _get_stripe()
    customer = s.Customer.create(email=email, name=name or "")
    return customer.id


async def create_checkout_session(
    customer_id: str, plan: str, success_url: str, cancel_url: str
) -> str:
    """Create a Stripe Checkout session and return the URL."""
    s = _get_stripe()
    session = s.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": PLAN_PRICES.get(plan, PLAN_PRICES["starter"]), "quantity": 1}],
        mode="subscription",
        success_url=success_url,
        cancel_url=cancel_url,
    )
    return session.url


async def create_portal_session(customer_id: str, return_url: str) -> str:
    """Create a Stripe Billing Portal session and return the URL."""
    s = _get_stripe()
    session = s.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url,
    )
    return session.url


async def get_user_subscription(db, user_id: str) -> dict | None:
    """Query the ff_subscriptions table for a user's subscription."""
    async with db._get_connection() as conn:
        cursor = await conn.execute(
            "SELECT id, user_id, stripe_customer_id, stripe_subscription_id, plan, status, current_period_end, created_at FROM ff_subscriptions WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "user_id": row[1],
        "stripe_customer_id": row[2],
        "stripe_subscription_id": row[3],
        "plan": row[4],
        "status": row[5],
        "current_period_end": row[6],
        "created_at": row[7],
    }
