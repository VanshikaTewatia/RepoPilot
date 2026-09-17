"""Stripe billing API routes (Phase 5): Checkout, Billing Portal, current
subscription status, and the Stripe webhook receiver.

Every route except the webhook requires CurrentUserDep, exactly like every
other authenticated route in this codebase -- no new auth mechanism. The
webhook route is the one deliberate exception (Stripe cannot send this
app's session cookie); its only protection is verifying the raw body
against its Stripe-Signature header (see
app.services.billing.service.verify_webhook_signature), which is why that
verification is non-negotiable.

Does not touch retrieval, indexing, embeddings, the agent, or
app.services.auth -- entitlement/usage enforcement for OTHER routes
(repositories, tasks) lives in app.services.billing.entitlements and is
wired into app.api.v1.repositories / app.api.v1.agent directly, not here.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.api.deps import CurrentUserDep, SessionDep
from app.core.logging import logger
from app.services.billing import entitlements, service as billing_service
from app.services.billing.service import (
    StripeAPIError,
    StripeNotConfiguredError,
    StripeWebhookVerificationError,
)

router = APIRouter(prefix="/billing", tags=["Billing"])


class CheckoutSessionResponse(BaseModel):
    checkout_url: str


class PortalSessionResponse(BaseModel):
    portal_url: str


class SubscriptionResponse(BaseModel):
    plan: str  # "free" | "pro"
    status: Optional[str] = None
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False


@router.post("/checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session_route(db: SessionDep, current_user: CurrentUserDep) -> Dict[str, str]:
    """Start a Stripe Checkout session for the Pro plan. The price charged
    always comes from server config (settings.stripe_price_id_pro) -- the
    request body has no price/customer/user fields to accept, so there is
    nothing for a client to tamper with here."""
    try:
        checkout_url = await billing_service.create_checkout_session(db, current_user)
    except StripeNotConfiguredError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except StripeAPIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    return {"checkout_url": checkout_url}


@router.post("/portal-session", response_model=PortalSessionResponse)
async def create_portal_session_route(db: SessionDep, current_user: CurrentUserDep) -> Dict[str, str]:
    """Start a Stripe Billing Portal session for self-service plan
    management (cancel, update payment method, etc.)."""
    try:
        portal_url = await billing_service.create_portal_session(db, current_user)
    except StripeNotConfiguredError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except StripeAPIError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    return {"portal_url": portal_url}


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription_route(db: SessionDep, current_user: CurrentUserDep) -> Dict[str, Any]:
    """Return the authenticated user's plan and subscription status."""
    subscription = await entitlements.get_subscription(db, current_user)
    return {
        "plan": entitlements.get_plan(subscription),
        "status": subscription.status if subscription else None,
        "current_period_end": (
            subscription.current_period_end.isoformat() if subscription and subscription.current_period_end else None
        ),
        "cancel_at_period_end": subscription.cancel_at_period_end if subscription else False,
    }


@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(request: Request, db: SessionDep) -> Dict[str, bool]:
    """Receive and apply a Stripe webhook event.

    Deliberately unauthenticated (no CurrentUserDep) -- Stripe cannot send
    this app's session cookie. Reads the RAW request body (never a parsed
    Pydantic model) because Stripe's signature is computed over the exact
    raw bytes; parsing/re-serializing first would break verification.
    Handles checkout.session.completed, customer.subscription.updated, and
    customer.subscription.deleted -- every other event type is acknowledged
    (200) and ignored.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = billing_service.verify_webhook_signature(payload, sig_header)
    except StripeWebhookVerificationError as e:
        logger.warning(f"Rejected Stripe webhook with invalid signature: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid webhook signature.")
    except StripeNotConfiguredError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    try:
        await billing_service.handle_webhook_event(db, event)
    except StripeAPIError as e:
        # A validly-signed event whose processing hit a real Stripe API
        # failure (e.g. Subscription.retrieve). Must NOT be swallowed into
        # a 200 -- Stripe only retries a webhook on a non-2xx response, and
        # returning 200 here would silently lose the event, leaving the
        # local subscription's status/price/period permanently stale.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
    return {"received": True}
