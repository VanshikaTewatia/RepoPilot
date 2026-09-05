"""Unit tests for app.services.agent.db.open_session -- the short-lived,
per-node database session helper Phase 6B's semantic retrieval uses.

No real Postgres connection anywhere: AsyncSessionLocal is patched with a
fake sessionmaker/session pair.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.services.agent.db import open_session


class _FakeSession:
    """A minimal async-context-manager stand-in for AsyncSession."""

    def __init__(self, fail_on_enter: bool = False):
        self.fail_on_enter = fail_on_enter
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        if self.fail_on_enter:
            raise RuntimeError("connection refused")
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.exited = True
        return False


class _RaisingSessionmaker:
    """A fake AsyncSessionLocal whose very construction raises."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("engine not configured")


@pytest.mark.asyncio
async def test_open_session_yields_a_working_session():
    fake_session = _FakeSession()
    with patch("app.services.agent.db.AsyncSessionLocal", return_value=fake_session):
        async with open_session() as session:
            assert session is fake_session
            assert session.entered is True
            assert session.exited is False

    assert fake_session.exited is True


@pytest.mark.asyncio
async def test_open_session_yields_none_when_sessionmaker_call_raises():
    with patch("app.services.agent.db.AsyncSessionLocal", _RaisingSessionmaker()):
        async with open_session() as session:
            assert session is None


@pytest.mark.asyncio
async def test_open_session_yields_none_when_connection_fails_on_enter():
    fake_session = _FakeSession(fail_on_enter=True)
    with patch("app.services.agent.db.AsyncSessionLocal", return_value=fake_session):
        async with open_session() as session:
            assert session is None


@pytest.mark.asyncio
async def test_open_session_never_raises_out_of_the_context_manager():
    """The whole point of open_session is that a caller never needs its own
    try/except around it -- any failure degrades to None."""
    with patch("app.services.agent.db.AsyncSessionLocal", _RaisingSessionmaker()):
        try:
            async with open_session() as session:
                assert session is None
        except Exception as e:  # pragma: no cover -- this branch must never run
            pytest.fail(f"open_session raised unexpectedly: {e}")


@pytest.mark.asyncio
async def test_open_session_produces_independent_sessions_across_calls():
    """Two sequential uses must never reuse or cache a session -- each
    call opens (and closes) its own, exactly as a short-lived per-node
    session should."""
    sessions = [_FakeSession(), _FakeSession()]
    call_count = {"n": 0}

    def _factory(*args, **kwargs):
        session = sessions[call_count["n"]]
        call_count["n"] += 1
        return session

    with patch("app.services.agent.db.AsyncSessionLocal", MagicMock(side_effect=_factory)):
        async with open_session() as first:
            assert first is sessions[0]
        async with open_session() as second:
            assert second is sessions[1]

    assert first is not second
    assert sessions[0].exited is True
    assert sessions[1].exited is True
