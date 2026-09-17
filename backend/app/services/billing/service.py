"""Stripe orchestration (Phase 5): Checkout, Billing Portal, and webhook
event application.

Thin orchestration over the Stripe SDK and the Subscription model --
mirrors app.services.auth.service's style (a small, composing entry-point
module, no business logic duplicated elsewhere). Nothing here touches the
RAG/QA pipeline, retrieval, embeddings, indexing, the agent, or
authentication itself; usage-limit enforcement lives in
app.services.billing.entitlements, not here.

Security-relevant design choices, all deliberate:
  * Checkout always uses settings.stripe_price_id_pro (server-configured) --
    never a price ID accepted from the client.
  * The Stripe customer id is created once per user and persisted
    immediately (see _get_or_create_customer_id), so repeated checkout
    attempts reuse it instead of creating duplicate Stripe customers.
  * Webhook events are correlated back to a local Subscription row by
    stripe_customer_id -- a value this service itself generated and stored,
    never trusted from webhook metadata. An event for an unknown customer
    id is silently ignored (see _find_subscription_by_customer_id).
  * Every webhook handler re-applies the full subscription state from
    Stripe (never increments/toggles anything), so processing the same
    event twice is a no-op the second time -- idempotent by construction,
    no separate processed-event-id ledger needed.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import stripe
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger
from app.db.models.subscription import Subscription
from app.db.models.user import User


class StripeNotConfiguredError(RuntimeError):
    """Raised when a billing route is invoked without stripe_secret_key set."""


class StripeWebhookVerificationError(ValueError):
    """Raised when a webhook payload's signature cannot be verified."""


class StripeAPIError(RuntimeError):
    """Raised when a real Stripe API call fails (authentication, rate
    limit, connection error, invalid request, etc.) -- as opposed to
    StripeNotConfiguredError (nothing was even attempted) or
    StripeWebhookVerificationError (a webhook signature didn't verify).
    Carries only a static, client-safe message; the real stripe.StripeError
    is preserved as __cause__ and logged server-side by _call_stripe, never
    surfaced to a caller."""


# Never str(e) a stripe.StripeError back to a client: Stripe's own error
# messages can echo configuration details (e.g. "No such price: 'price_x'"
# reveals settings.stripe_price_id_pro) or a masked API key fragment (e.g.
# "Invalid API Key provided: sk_test_****real"). This message is the only
# thing that ever reaches an HTTP response for a failed Stripe API call.
_STRIPE_API_ERROR_MESSAGE = "Stripe is temporarily unavailable. Please try again shortly."


def _configure_stripe() -> None:
    if not settings.stripe_secret_key:
        raise StripeNotConfiguredError("STRIPE_SECRET_KEY is not configured on the server.")
    stripe.api_key = settings.stripe_secret_key


async def _call_stripe(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a blocking Stripe SDK call in a thread (asyncio.to_thread, the
    same pattern every real Stripe API call in this module already uses),
    catching stripe.StripeError and re-raising as StripeAPIError.

    Deliberately narrow: only wraps the four real Stripe API calls this
    module makes (Customer.create, checkout.Session.create,
    billing_portal.Session.create, Subscription.retrieve). Must never wrap
    stripe.Webhook.construct_event -- stripe.SignatureVerificationError is
    itself a stripe.StripeError subclass, and verify_webhook_signature
    already has its own, separate, correct handling for it (-> 400); a
    broad catch here must not touch that call site.
    """
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except stripe.StripeError as e:
        logger.warning(f"Stripe API call failed: {e}")
        raise StripeAPIError(_STRIPE_API_ERROR_MESSAGE) from e


async def _get_or_create_customer_id(db: AsyncSession, user: User) -> str:
    """Return the user's Stripe customer id, creating both the Stripe
    Customer and the local Subscription row on first use.

    Idempotent per user: a second call for the same user finds the
    already-persisted row and never creates a second Stripe customer.
    """
    result = await db.execute(select(Subscription).where(Subscription.user_id == user.id))
    subscription = result.scalar_one_or_none()
    if subscription is not None:
        return subscription.stripe_customer_id

    _configure_stripe()
    customer = await _call_stripe(
        stripe.Customer.create,
        email=user.email,
        metadata={"user_id": str(user.id)},
    )

    subscription = Subscription(
        user_id=user.id,
        stripe_customer_id=customer["id"],
        status="incomplete",
    )
    db.add(subscription)
    await db.commit()
    await db.refresh(subscription)
    return subscription.stripe_customer_id


async def create_checkout_session(db: AsyncSession, user: User) -> str:
    """Create a Stripe Checkout Session for the Pro plan and return its
    redirect URL. Always uses the server-configured Pro price id."""
    if not settings.stripe_price_id_pro:
        raise StripeNotConfiguredError("STRIPE_PRICE_ID_PRO is not configured on the server.")

    customer_id = await _get_or_create_customer_id(db, user)
    _configure_stripe()
    session = await _call_stripe(
        stripe.checkout.Session.create,
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": settings.stripe_price_id_pro, "quantity": 1}],
        success_url=f"{settings.frontend_base_url}/billing?checkout=success",
        cancel_url=f"{settings.frontend_base_url}/billing?checkout=cancelled",
    )
    return session["url"]


async def create_portal_session(db: AsyncSession, user: User) -> str:
    """Create a Stripe Billing Portal session for self-service plan
    management and return its URL. Requires an existing customer (i.e. the
    user has started checkout at least once)."""
    result = await db.execute(select(Subscription).where(Subscription.user_id == user.id))
    subscription = result.scalar_one_or_none()
    if subscription is None:
        raise StripeNotConfiguredError("No billing account exists yet -- start a checkout first.")

    _configure_stripe()
    portal_session = await _call_stripe(
        stripe.billing_portal.Session.create,
        customer=subscription.stripe_customer_id,
        return_url=f"{settings.frontend_base_url}/billing",
    )
    return portal_session["url"]


def verify_webhook_signature(payload: bytes, sig_header: str) -> Any:
    """Verify a raw webhook request body against its Stripe-Signature
    header. Raises StripeWebhookVerificationError for any failure (missing
    header, bad signature, malformed payload) -- callers must reject the
    request (400) rather than proceed, since an unverified payload could be
    forged by anyone."""
    if not settings.stripe_webhook_secret:
        raise StripeNotConfiguredError("STRIPE_WEBHOOK_SECRET is not configured on the server.")
    try:
        return stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
    except (ValueError, stripe.SignatureVerificationError) as e:
        raise StripeWebhookVerificationError(str(e)) from e


def _as_dict(obj: Any) -> dict:
    """Normalize a Stripe SDK object into a plain dict.

    stripe-python 15.x's StripeObject deliberately does NOT subclass dict
    and blocks dict-style methods like .get()/.items() via __getattr__ (it
    only supports [] access and `in`) -- see stripe._stripe_object.
    Everything below this point uses .get() freely, so every Stripe object
    (an Event's data.object, or a Subscription returned by
    stripe.Subscription.retrieve) is normalized here first via .to_dict(),
    which recursively converts nested StripeObjects too. A plain dict (as
    used directly by this module's own unit tests) passes through as-is.
    """
    if isinstance(obj, dict):
        return obj
    return obj.to_dict()


def _apply_stripe_subscription(subscription: Subscription, stripe_subscription: Any) -> None:
    """Overwrite every locally-tracked field from a full Stripe Subscription
    object. Always re-applies the complete state (never increments/toggles
    anything), which is what makes repeated webhook delivery of the same
    event idempotent."""
    subscription.stripe_subscription_id = stripe_subscription["id"]
    subscription.status = stripe_subscription["status"]
    subscription.cancel_at_period_end = bool(stripe_subscription.get("cancel_at_period_end", False))

    items = (stripe_subscription.get("items") or {}).get("data") or []
    if items:
        subscription.stripe_price_id = items[0]["price"]["id"]

    period_start = stripe_subscription.get("current_period_start")
    if period_start is not None:
        subscription.current_period_start = datetime.fromtimestamp(period_start, tz=timezone.utc)

    period_end = stripe_subscription.get("current_period_end")
    if period_end is not None:
        subscription.current_period_end = datetime.fromtimestamp(period_end, tz=timezone.utc)


async def _find_subscription_by_customer_id(db: AsyncSession, customer_id: Optional[str]) -> Optional[Subscription]:
    if not customer_id:
        return None
    result = await db.execute(select(Subscription).where(Subscription.stripe_customer_id == customer_id))
    return result.scalar_one_or_none()


async def _handle_checkout_session_completed(db: AsyncSession, session_obj: Any) -> None:
    """checkout.session.completed carries only a subscription id, not the
    full Subscription object -- fetch it once so status/price/period are
    all populated immediately rather than waiting for a later
    customer.subscription.updated event."""
    customer_id = session_obj.get("customer")
    subscription_id = session_obj.get("subscription")
    subscription = await _find_subscription_by_customer_id(db, customer_id)
    if subscription is None or not subscription_id:
        logger.warning(f"Stripe checkout.session.completed for unknown customer '{customer_id}' -- ignored.")
        return

    _configure_stripe()
    stripe_subscription = _as_dict(await _call_stripe(stripe.Subscription.retrieve, subscription_id))
    _apply_stripe_subscription(subscription, stripe_subscription)
    await db.commit()


async def _handle_subscription_upsert(db: AsyncSession, stripe_subscription: Any) -> None:
    """Shared by customer.subscription.updated and .deleted -- both events
    carry the full Subscription object, so both just re-apply it."""
    customer_id = stripe_subscription.get("customer")
    subscription = await _find_subscription_by_customer_id(db, customer_id)
    if subscription is None:
        logger.warning(f"Stripe subscription event for unknown customer '{customer_id}' -- ignored.")
        return

    _apply_stripe_subscription(subscription, stripe_subscription)
    await db.commit()


async def handle_webhook_event(db: AsyncSession, event: Any) -> None:
    """Dispatch a verified Stripe event to its handler. Unrecognized event
    types are acknowledged and ignored (Stripe only retries on non-2xx
    responses, so silently ignoring a type we don't handle is correct, not
    an oversight)."""
    event_type = event["type"]
    data_object = _as_dict(event["data"]["object"])

    if event_type == "checkout.session.completed":
        await _handle_checkout_session_completed(db, data_object)
    elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        await _handle_subscription_upsert(db, data_object)
