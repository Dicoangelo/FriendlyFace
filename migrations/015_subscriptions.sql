-- Migration 015: Subscriptions for Stripe billing
CREATE TABLE IF NOT EXISTS ff_subscriptions (
    id TEXT PRIMARY KEY DEFAULT (hex(randomblob(16))),
    user_id TEXT NOT NULL UNIQUE REFERENCES ff_users(id),
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    plan TEXT NOT NULL DEFAULT 'starter' CHECK(plan IN ('starter', 'professional', 'enterprise')),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'past_due', 'canceled', 'trialing')),
    current_period_end TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ff_subscriptions_user_id ON ff_subscriptions (user_id);
CREATE INDEX IF NOT EXISTS idx_ff_subscriptions_stripe_customer ON ff_subscriptions (stripe_customer_id);
