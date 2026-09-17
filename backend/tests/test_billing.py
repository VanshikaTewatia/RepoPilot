"""Tests for Phase 5 Stripe billing: subscription persistence, Checkout/
Portal session creation (authorization + configuration), webhook signature
verification, webhook idempotency, subscription activation/update/deletion,
and free/Pro entitlement enforcement.

Follows this codebase's established convention: direct route/service
function calls with a fake, in-process database session for everything
that doesn't need the real HTTP layer (test_authorization.py,
test_repositories_github.py, etc.), except the webhook endpoint, which is
exercised through the real TestClient + app.dependency_overrides so the
raw-body/signature-header behavior is verified at the actual HTTP layer,
not guessed at by calling the route function directly (same rationale as
test_auth.py's own use of the ``client`` fixture for cookie behavior).

No real Stripe network calls anywhere: stripe.Customer.create,
stripe.checkout.Session.create, stripe.billing_portal.Session.create, and
stripe.Subscription.retrieve are all monkeypatched.
"""

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import stripe
from fastapi import HTTPException

from app.core.config import settings
from app.db.models.repository import Repository
from app.db.models.subscription import Subscription
from app.db.models.task import Task
from app.db.models.user import User
from app.db.session import get_db
from app.main import app
from app.services.billing import entitlements, service as billing_service
from app.services.billing.service import (
    StripeAPIError,
    StripeNotConfiguredError,
    StripeWebhookVerificationError,
)

USER_A = User(id=1, email="alice@example.com", hashed_password="x")
USER_B = User(id=2, email="bob@example.com", hashed_password="x")


# ---------------------------------------------------------------------------
# Stateful fake DB session covering exactly the query shapes Phase 5 code
# issues: Subscription-by-user_id, Subscription-by-stripe_customer_id, a
# repository COUNT, and a task-joined-to-repository COUNT.
# ---------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, scalar=None):
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar

    def scalar_one(self):
        return self._scalar


class _FakeDB:
    def __init__(self):
        self.subscriptions: List[Subscription] = []
        self.repositories: List[Repository] = []
        self.tasks: List[Task] = []
        self._pending = None
        self._next_sub_id = 1

    def add(self, obj) -> None:
        self._pending = obj

    async def commit(self) -> None:
        if self._pending is not None and isinstance(self._pending, Subscription):
            if self._pending.id is None:
                self._pending.id = self._next_sub_id
                self._next_sub_id += 1
            if self._pending not in self.subscriptions:
                self.subscriptions.append(self._pending)
        self._pending = None

    async def refresh(self, obj) -> None:
        pass

    def _task_matches(self, task: Task, params: Dict) -> bool:
        repo = next((r for r in self.repositories if r.id == task.repository_id), None)
        if repo is None or repo.user_id != params.get("user_id_1"):
            return False
        threshold = params.get("created_at_1")
        if threshold is not None and task.created_at < threshold:
            return False
        return True

    async def execute(self, stmt):
        entity = stmt.column_descriptions[0].get("entity")
        params = stmt.compile().params

        if entity is Subscription:
            if "user_id_1" in params:
                match = next((s for s in self.subscriptions if s.user_id == params["user_id_1"]), None)
            elif "stripe_customer_id_1" in params:
                match = next(
                    (s for s in self.subscriptions if s.stripe_customer_id == params["stripe_customer_id_1"]),
                    None,
                )
            else:
                match = None
            return _FakeResult(scalar=match)

        # A count() query -- distinguish repository-only vs. task-joined-
        # to-repository by the FROM clause's table names.
        froms = stmt.get_final_froms()
        joined = str(froms[0]) if froms else ""
        if "tasks" in joined and "repositories" in joined:
            count = sum(1 for t in self.tasks if self._task_matches(t, params))
        else:
            count = sum(1 for r in self.repositories if r.user_id == params.get("user_id_1"))
        return _FakeResult(scalar=count)


def _sig_header(payload: bytes, secret: str) -> str:
    ts = int(time.time())
    signed_payload = f"{ts}.{payload.decode()}"
    sig = hmac.new(secret.encode(), signed_payload.encode(), hashlib.sha256).hexdigest()
    return f"t={ts},v1={sig}"


# ===========================================================================
# Subscription persistence / entitlement calculation
# ===========================================================================
def test_no_subscription_row_means_free_plan():
    assert entitlements.get_plan(None) == "free"


def test_active_status_means_pro_plan():
    sub = Subscription(id=1, user_id=1, stripe_customer_id="cus_1", status="active")
    assert entitlements.get_plan(sub) == "pro"


def test_trialing_status_means_pro_plan():
    sub = Subscription(id=1, user_id=1, stripe_customer_id="cus_1", status="trialing")
    assert entitlements.get_plan(sub) == "pro"


@pytest.mark.parametrize("bad_status", ["incomplete", "past_due", "canceled", "unpaid", "incomplete_expired"])
def test_non_entitled_statuses_mean_free_plan(bad_status):
    sub = Subscription(id=1, user_id=1, stripe_customer_id="cus_1", status=bad_status)
    assert entitlements.get_plan(sub) == "free"


# ===========================================================================
# Free-tier repository limit
# ===========================================================================
@pytest.mark.asyncio
async def test_free_tier_repository_limit_allows_under_cap():
    db = _FakeDB()
    await entitlements.check_repository_limit(db, USER_A)  # no repos yet -- no raise


@pytest.mark.asyncio
async def test_free_tier_repository_limit_blocks_at_cap():
    db = _FakeDB()
    db.repositories.append(Repository(id=1, name="r1", local_path="/tmp/a", user_id=USER_A.id))

    with pytest.raises(HTTPException) as exc_info:
        await entitlements.check_repository_limit(db, USER_A)

    assert exc_info.value.status_code == 402


@pytest.mark.asyncio
async def test_free_tier_repository_limit_is_per_user():
    db = _FakeDB()
    db.repositories.append(Repository(id=1, name="r1", local_path="/tmp/a", user_id=USER_A.id))

    await entitlements.check_repository_limit(db, USER_B)  # Bob has none of his own -- no raise


@pytest.mark.asyncio
async def test_pro_user_bypasses_repository_limit():
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="active"))
    for i in range(5):
        db.repositories.append(Repository(id=i, name=f"r{i}", local_path=f"/tmp/{i}", user_id=USER_A.id))

    await entitlements.check_repository_limit(db, USER_A)  # well over the free cap -- still no raise


# ===========================================================================
# Free-tier monthly task limit
# ===========================================================================
def _seed_repo_and_tasks(db: _FakeDB, owner_id: int, count: int, created_at: Optional[datetime] = None) -> None:
    db.repositories.append(Repository(id=1, name="r1", local_path="/tmp/a", user_id=owner_id))
    for i in range(count):
        db.tasks.append(
            Task(
                id=i,
                repository_id=1,
                title="t",
                description="d",
                status="investigating",
                created_at=created_at or datetime.now(timezone.utc),
            )
        )


@pytest.mark.asyncio
async def test_free_tier_task_limit_allows_under_cap():
    db = _FakeDB()
    _seed_repo_and_tasks(db, USER_A.id, count=settings.free_tier_monthly_task_limit - 1)
    await entitlements.check_task_limit(db, USER_A)  # no raise


@pytest.mark.asyncio
async def test_free_tier_task_limit_blocks_at_cap():
    db = _FakeDB()
    _seed_repo_and_tasks(db, USER_A.id, count=settings.free_tier_monthly_task_limit)

    with pytest.raises(HTTPException) as exc_info:
        await entitlements.check_task_limit(db, USER_A)

    assert exc_info.value.status_code == 402


@pytest.mark.asyncio
async def test_free_tier_task_limit_ignores_tasks_from_a_prior_calendar_month():
    db = _FakeDB()
    long_ago = datetime.now(timezone.utc) - timedelta(days=90)
    # At the cap by raw count, but all of it predates this month -- must not block.
    _seed_repo_and_tasks(db, USER_A.id, count=settings.free_tier_monthly_task_limit, created_at=long_ago)

    await entitlements.check_task_limit(db, USER_A)  # no raise


@pytest.mark.asyncio
async def test_free_tier_task_limit_uses_subscriptions_own_billing_period_once_one_exists():
    db = _FakeDB()
    period_start = datetime.now(timezone.utc) - timedelta(days=3)
    db.subscriptions.append(
        Subscription(
            id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="canceled",
            current_period_start=period_start,
        )
    )
    db.repositories.append(Repository(id=1, name="r1", local_path="/tmp/a", user_id=USER_A.id))
    # One task before the period started (must not count) and enough within it to hit the cap.
    db.tasks.append(
        Task(
            id=0, repository_id=1, title="old", description="d", status="failed",
            created_at=period_start - timedelta(days=1),
        )
    )
    for i in range(1, settings.free_tier_monthly_task_limit + 1):
        db.tasks.append(
            Task(id=i, repository_id=1, title="t", description="d", status="investigating", created_at=period_start)
        )

    with pytest.raises(HTTPException) as exc_info:
        await entitlements.check_task_limit(db, USER_A)
    assert exc_info.value.status_code == 402


@pytest.mark.asyncio
async def test_pro_user_bypasses_task_limit():
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="active"))
    _seed_repo_and_tasks(db, USER_A.id, count=settings.free_tier_monthly_task_limit + 10)

    await entitlements.check_task_limit(db, USER_A)  # no raise


# ===========================================================================
# Checkout session: authorization + configuration
# ===========================================================================
@pytest.mark.asyncio
async def test_create_checkout_session_requires_configured_price(monkeypatch):
    monkeypatch.setattr(settings, "stripe_price_id_pro", "")
    db = _FakeDB()

    with pytest.raises(StripeNotConfiguredError):
        await billing_service.create_checkout_session(db, USER_A)


@pytest.mark.asyncio
async def test_create_checkout_session_requires_configured_secret_key(monkeypatch):
    monkeypatch.setattr(settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(settings, "stripe_secret_key", "")
    db = _FakeDB()

    with pytest.raises(StripeNotConfiguredError):
        await billing_service.create_checkout_session(db, USER_A)


@pytest.mark.asyncio
async def test_create_checkout_session_uses_server_configured_price_never_a_client_value(monkeypatch):
    monkeypatch.setattr(settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()

    create_calls = []

    def _fake_session_create(**kwargs):
        create_calls.append(kwargs)
        return {"id": "cs_1", "url": "https://checkout.stripe.com/cs_1"}

    monkeypatch.setattr(stripe.checkout.Session, "create", _fake_session_create)
    monkeypatch.setattr(stripe.Customer, "create", lambda **kwargs: {"id": "cus_1"})

    url = await billing_service.create_checkout_session(db, USER_A)

    assert url == "https://checkout.stripe.com/cs_1"
    assert len(create_calls) == 1
    assert create_calls[0]["line_items"] == [{"price": "price_pro_123", "quantity": 1}]
    assert create_calls[0]["customer"] == "cus_1"


@pytest.mark.asyncio
async def test_create_checkout_session_reuses_existing_stripe_customer(monkeypatch):
    monkeypatch.setattr(settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()

    customer_create_calls = []
    monkeypatch.setattr(
        stripe.Customer, "create",
        lambda **kwargs: customer_create_calls.append(kwargs) or {"id": "cus_1"},
    )
    monkeypatch.setattr(
        stripe.checkout.Session, "create",
        lambda **kwargs: {"id": "cs_1", "url": "https://checkout.stripe.com/cs_1"},
    )

    await billing_service.create_checkout_session(db, USER_A)
    await billing_service.create_checkout_session(db, USER_A)

    assert len(customer_create_calls) == 1  # second checkout reused the stored customer id
    assert len(db.subscriptions) == 1


# ===========================================================================
# Billing Portal session
# ===========================================================================
@pytest.mark.asyncio
async def test_create_portal_session_requires_existing_customer():
    db = _FakeDB()  # no Subscription row -- user never checked out

    with pytest.raises(StripeNotConfiguredError):
        await billing_service.create_portal_session(db, USER_A)


@pytest.mark.asyncio
async def test_create_portal_session_uses_stored_customer_id(monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="active"))

    calls = []
    monkeypatch.setattr(
        stripe.billing_portal.Session, "create",
        lambda **kwargs: calls.append(kwargs) or {"id": "bps_1", "url": "https://billing.stripe.com/p/1"},
    )

    url = await billing_service.create_portal_session(db, USER_A)

    assert url == "https://billing.stripe.com/p/1"
    assert calls[0]["customer"] == "cus_1"


# ===========================================================================
# Real Stripe API failures -> StripeAPIError, never a raw stripe.StripeError
# (Phase 6D). Each uses a real, SDK-shaped exception -- not a bare Exception
# or a plain MagicMock error -- matching exactly what stripe.Customer.create/
# etc. raise for a rejected key (confirmed against the installed stripe SDK
# during the Phase 6C investigation: stripe.AuthenticationError(message=...,
# http_status=...)).
# ===========================================================================
def _real_auth_error(message: str = "Invalid API Key provided: sk_test_****REDACTED") -> stripe.AuthenticationError:
    return stripe.AuthenticationError(message=message, http_status=401)


@pytest.mark.asyncio
async def test_get_or_create_customer_id_wraps_stripe_error_in_stripe_api_error(monkeypatch):
    """A. stripe.Customer.create failing must surface as StripeAPIError,
    with the real exception preserved as the cause."""
    monkeypatch.setattr(settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()  # no existing Subscription row -- forces Customer.create

    monkeypatch.setattr(stripe.Customer, "create", MagicMock(side_effect=_real_auth_error()))

    with pytest.raises(StripeAPIError) as exc_info:
        await billing_service.create_checkout_session(db, USER_A)

    assert isinstance(exc_info.value.__cause__, stripe.AuthenticationError)


@pytest.mark.asyncio
async def test_create_checkout_session_wraps_stripe_error_in_stripe_api_error(monkeypatch):
    """B. stripe.checkout.Session.create failing must surface as
    StripeAPIError. A Subscription row is pre-seeded so
    _get_or_create_customer_id returns early and Customer.create is never
    called -- isolates the failure to Session.create specifically."""
    monkeypatch.setattr(settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="incomplete"))

    monkeypatch.setattr(stripe.checkout.Session, "create", MagicMock(side_effect=_real_auth_error()))

    with pytest.raises(StripeAPIError) as exc_info:
        await billing_service.create_checkout_session(db, USER_A)

    assert isinstance(exc_info.value.__cause__, stripe.AuthenticationError)


@pytest.mark.asyncio
async def test_create_portal_session_wraps_stripe_error_in_stripe_api_error(monkeypatch):
    """C. stripe.billing_portal.Session.create failing must surface as
    StripeAPIError."""
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="active"))

    monkeypatch.setattr(stripe.billing_portal.Session, "create", MagicMock(side_effect=_real_auth_error()))

    with pytest.raises(StripeAPIError) as exc_info:
        await billing_service.create_portal_session(db, USER_A)

    assert isinstance(exc_info.value.__cause__, stripe.AuthenticationError)


@pytest.mark.asyncio
async def test_checkout_session_completed_wraps_stripe_error_from_subscription_retrieve(monkeypatch):
    """D. stripe.Subscription.retrieve failing (inside webhook processing)
    must surface as StripeAPIError, not propagate as a raw StripeError."""
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="incomplete"))

    monkeypatch.setattr(stripe.Subscription, "retrieve", MagicMock(side_effect=_real_auth_error()))

    event = {
        "type": "checkout.session.completed",
        "data": {"object": {"customer": "cus_1", "subscription": "sub_1"}},
    }
    with pytest.raises(StripeAPIError) as exc_info:
        await billing_service.handle_webhook_event(db, event)

    assert isinstance(exc_info.value.__cause__, stripe.AuthenticationError)


# ===========================================================================
# Webhook signature verification (direct, service-level)
# ===========================================================================
def test_verify_webhook_signature_requires_configured_secret(monkeypatch):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "")
    with pytest.raises(StripeNotConfiguredError):
        billing_service.verify_webhook_signature(b"{}", "t=1,v1=bad")


def test_verify_webhook_signature_rejects_bad_signature(monkeypatch):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    with pytest.raises(StripeWebhookVerificationError):
        billing_service.verify_webhook_signature(b'{"type": "x"}', "t=1,v1=not_a_real_signature")


def test_verify_webhook_signature_accepts_a_correctly_signed_payload(monkeypatch):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    payload = json.dumps({"id": "evt_1", "type": "checkout.session.completed", "data": {"object": {}}}).encode()
    header = _sig_header(payload, "whsec_test_secret")

    event = billing_service.verify_webhook_signature(payload, header)
    assert event["type"] == "checkout.session.completed"


# ===========================================================================
# Webhook event application: activation / update / deletion, and idempotency
# ===========================================================================
@pytest.mark.asyncio
async def test_checkout_session_completed_activates_the_subscription(monkeypatch):
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="incomplete"))

    fake_stripe_sub = {
        "id": "sub_1",
        "status": "active",
        "cancel_at_period_end": False,
        "items": {"data": [{"price": {"id": "price_pro_123"}}]},
        "current_period_start": 1700000000,
        "current_period_end": 1702592000,
        "customer": "cus_1",
    }
    monkeypatch.setattr(stripe.Subscription, "retrieve", lambda subscription_id: fake_stripe_sub)

    event = {
        "type": "checkout.session.completed",
        "data": {"object": {"customer": "cus_1", "subscription": "sub_1"}},
    }
    await billing_service.handle_webhook_event(db, event)

    sub = db.subscriptions[0]
    assert sub.status == "active"
    assert sub.stripe_subscription_id == "sub_1"
    assert sub.stripe_price_id == "price_pro_123"
    assert sub.current_period_start == datetime.fromtimestamp(1700000000, tz=timezone.utc)
    assert entitlements.get_plan(sub) == "pro"


@pytest.mark.asyncio
async def test_checkout_session_completed_for_unknown_customer_is_ignored():
    db = _FakeDB()  # no matching Subscription row for this customer id
    event = {
        "type": "checkout.session.completed",
        "data": {"object": {"customer": "cus_unknown", "subscription": "sub_1"}},
    }
    await billing_service.handle_webhook_event(db, event)  # must not raise
    assert db.subscriptions == []


@pytest.mark.asyncio
async def test_subscription_updated_event_updates_price_and_period(monkeypatch):
    db = _FakeDB()
    db.subscriptions.append(
        Subscription(
            id=1, user_id=USER_A.id, stripe_customer_id="cus_1",
            stripe_subscription_id="sub_1", status="active", stripe_price_id="price_old",
        )
    )
    event = {
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_1",
                "customer": "cus_1",
                "status": "active",
                "cancel_at_period_end": True,
                "items": {"data": [{"price": {"id": "price_new"}}]},
                "current_period_start": 1710000000,
                "current_period_end": 1712592000,
            }
        },
    }
    await billing_service.handle_webhook_event(db, event)

    sub = db.subscriptions[0]
    assert sub.stripe_price_id == "price_new"
    assert sub.cancel_at_period_end is True


@pytest.mark.asyncio
async def test_subscription_deleted_event_revokes_entitlement():
    db = _FakeDB()
    sub = Subscription(
        id=1, user_id=USER_A.id, stripe_customer_id="cus_1",
        stripe_subscription_id="sub_1", status="active",
    )
    db.subscriptions.append(sub)
    assert entitlements.get_plan(sub) == "pro"

    event = {
        "type": "customer.subscription.deleted",
        "data": {
            "object": {
                "id": "sub_1",
                "customer": "cus_1",
                "status": "canceled",
                "cancel_at_period_end": False,
                "items": {"data": []},
            }
        },
    }
    await billing_service.handle_webhook_event(db, event)

    assert sub.status == "canceled"
    assert entitlements.get_plan(sub) == "free"


@pytest.mark.asyncio
async def test_webhook_processing_is_idempotent_for_a_retried_event():
    db = _FakeDB()
    db.subscriptions.append(
        Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="incomplete")
    )
    event = {
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_1",
                "customer": "cus_1",
                "status": "active",
                "cancel_at_period_end": False,
                "items": {"data": [{"price": {"id": "price_pro_123"}}]},
                "current_period_start": 1700000000,
                "current_period_end": 1702592000,
            }
        },
    }

    await billing_service.handle_webhook_event(db, event)
    await billing_service.handle_webhook_event(db, event)  # Stripe retry -- same event again

    assert len(db.subscriptions) == 1  # no duplicate row
    sub = db.subscriptions[0]
    assert sub.status == "active"
    assert sub.stripe_price_id == "price_pro_123"


# ===========================================================================
# Webhook endpoint: real HTTP layer (raw body + Stripe-Signature header)
# ===========================================================================
@pytest.fixture
def fake_webhook_db():
    session = _FakeDB()

    async def _get_db():
        yield session

    app.dependency_overrides[get_db] = _get_db
    yield session
    app.dependency_overrides.clear()


def test_webhook_endpoint_rejects_missing_signature(client, monkeypatch, fake_webhook_db):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    payload = json.dumps({"type": "checkout.session.completed", "data": {"object": {}}}).encode()

    resp = client.post("/api/v1/billing/webhook", content=payload, headers={"content-type": "application/json"})
    assert resp.status_code == 400


def test_webhook_endpoint_rejects_invalid_signature(client, monkeypatch, fake_webhook_db):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    payload = json.dumps({"type": "checkout.session.completed", "data": {"object": {}}}).encode()

    resp = client.post(
        "/api/v1/billing/webhook",
        content=payload,
        headers={"content-type": "application/json", "stripe-signature": "t=1,v1=deadbeef"},
    )
    assert resp.status_code == 400


def test_webhook_endpoint_accepts_a_correctly_signed_unknown_event_type(client, monkeypatch, fake_webhook_db):
    """An event type this app doesn't handle is still acknowledged 200, not
    rejected -- only the signature is checked, never the event type."""
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    payload = json.dumps(
        {"id": "evt_1", "type": "invoice.paid", "data": {"object": {}}}
    ).encode()
    header = _sig_header(payload, "whsec_test_secret")

    resp = client.post(
        "/api/v1/billing/webhook",
        content=payload,
        headers={"content-type": "application/json", "stripe-signature": header},
    )
    assert resp.status_code == 200
    assert resp.json() == {"received": True}


def test_webhook_endpoint_applies_a_correctly_signed_subscription_deleted_event(
    client, monkeypatch, fake_webhook_db
):
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    fake_webhook_db.subscriptions.append(
        Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_http_1", status="active")
    )
    payload = json.dumps(
        {
            "id": "evt_1",
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": "sub_http_1",
                    "customer": "cus_http_1",
                    "status": "canceled",
                    "cancel_at_period_end": False,
                    "items": {"data": []},
                }
            },
        }
    ).encode()
    header = _sig_header(payload, "whsec_test_secret")

    resp = client.post(
        "/api/v1/billing/webhook",
        content=payload,
        headers={"content-type": "application/json", "stripe-signature": header},
    )

    assert resp.status_code == 200
    assert fake_webhook_db.subscriptions[0].status == "canceled"


def test_webhook_route_maps_stripe_api_error_to_502(client, monkeypatch, fake_webhook_db):
    """G. A validly-signed webhook whose processing hits a real Stripe API
    failure (Subscription.retrieve) must return 502, never an accidental
    generic 500, and never 200 -- 200 would stop Stripe from retrying and
    silently lose the event."""
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    fake_webhook_db.subscriptions.append(
        Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_http_1", status="incomplete")
    )
    monkeypatch.setattr(stripe.Subscription, "retrieve", MagicMock(side_effect=_real_auth_error()))

    payload = json.dumps(
        {
            "id": "evt_1",
            "type": "checkout.session.completed",
            "data": {"object": {"customer": "cus_http_1", "subscription": "sub_http_1"}},
        }
    ).encode()
    header = _sig_header(payload, "whsec_test_secret")

    resp = client.post(
        "/api/v1/billing/webhook",
        content=payload,
        headers={"content-type": "application/json", "stripe-signature": header},
    )

    assert resp.status_code == 502


def test_stripe_api_error_never_leaks_raw_stripe_message_to_client(client, monkeypatch, fake_webhook_db):
    """I. The raw stripe.StripeError text -- which can echo back
    configuration details (e.g. an invalid price id) or a masked API key
    fragment -- must never reach the client-facing HTTPException.detail."""
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_secret")
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    fake_webhook_db.subscriptions.append(
        Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_http_1", status="incomplete")
    )
    sensitive_fragment = "price_super_secret_config_value"
    monkeypatch.setattr(
        stripe.Subscription, "retrieve",
        MagicMock(side_effect=stripe.InvalidRequestError(
            message=f"No such price: '{sensitive_fragment}'", param="price",
        )),
    )

    payload = json.dumps(
        {
            "id": "evt_1",
            "type": "checkout.session.completed",
            "data": {"object": {"customer": "cus_http_1", "subscription": "sub_http_1"}},
        }
    ).encode()
    header = _sig_header(payload, "whsec_test_secret")

    resp = client.post(
        "/api/v1/billing/webhook",
        content=payload,
        headers={"content-type": "application/json", "stripe-signature": header},
    )

    assert resp.status_code == 502
    assert sensitive_fragment not in resp.text
    assert "No such price" not in resp.text


# ===========================================================================
# Route-level: GET /billing/subscription
# ===========================================================================
@pytest.mark.asyncio
async def test_get_subscription_route_reports_free_plan_with_no_row():
    from app.api.v1.billing import get_subscription_route

    db = _FakeDB()
    result = await get_subscription_route(db, USER_A)

    assert result["plan"] == "free"
    assert result["status"] is None


@pytest.mark.asyncio
async def test_get_subscription_route_reports_pro_plan_with_active_row():
    from app.api.v1.billing import get_subscription_route

    db = _FakeDB()
    db.subscriptions.append(
        Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="active", cancel_at_period_end=True)
    )
    result = await get_subscription_route(db, USER_A)

    assert result["plan"] == "pro"
    assert result["status"] == "active"
    assert result["cancel_at_period_end"] is True


# ===========================================================================
# Route-level: checkout/portal session error mapping
# ===========================================================================
@pytest.mark.asyncio
async def test_checkout_session_route_returns_500_when_not_configured(monkeypatch):
    from app.api.v1.billing import create_checkout_session_route

    monkeypatch.setattr(settings, "stripe_price_id_pro", "")
    db = _FakeDB()

    with pytest.raises(HTTPException) as exc_info:
        await create_checkout_session_route(db, USER_A)
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_portal_session_route_returns_400_when_no_customer_exists():
    from app.api.v1.billing import create_portal_session_route

    db = _FakeDB()
    with pytest.raises(HTTPException) as exc_info:
        await create_portal_session_route(db, USER_A)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_checkout_session_route_maps_stripe_api_error_to_502(monkeypatch):
    """E. A real Stripe API failure during checkout must map to 502, not
    500 -- StripeNotConfiguredError (server misconfiguration, above) and
    StripeAPIError (a real but failed Stripe call) are deliberately kept
    distinguishable at the route layer."""
    from app.api.v1.billing import create_checkout_session_route

    monkeypatch.setattr(settings, "stripe_price_id_pro", "price_pro_123")
    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()
    monkeypatch.setattr(stripe.Customer, "create", MagicMock(side_effect=_real_auth_error()))

    with pytest.raises(HTTPException) as exc_info:
        await create_checkout_session_route(db, USER_A)

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == billing_service._STRIPE_API_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_portal_session_route_maps_stripe_api_error_to_502(monkeypatch):
    """F. Same mapping for the portal route."""
    from app.api.v1.billing import create_portal_session_route

    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_x")
    db = _FakeDB()
    db.subscriptions.append(Subscription(id=1, user_id=USER_A.id, stripe_customer_id="cus_1", status="active"))
    monkeypatch.setattr(stripe.billing_portal.Session, "create", MagicMock(side_effect=_real_auth_error()))

    with pytest.raises(HTTPException) as exc_info:
        await create_portal_session_route(db, USER_A)

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == billing_service._STRIPE_API_ERROR_MESSAGE
