"""API-layer tests for Phase 6F: non-blocking, progressively observable task
execution (POST /tasks -> background LangGraph execution -> GET /tasks/{id}
polling).

Before this phase, `create_and_run_task` had NO dedicated test coverage at
all -- every existing agent-graph test calls `agent_app.ainvoke()` or
individual node functions directly (see test_agent_graph.py,
test_agent_outcomes.py, etc.), never the actual API endpoint. This suite is
new coverage for that endpoint, following the same direct-function-call
convention already established by test_task_review.py (not FastAPI's
TestClient) -- deliberately so, since Starlette's TestClient runs
BackgroundTasks synchronously as part of a single blocking call, which would
make it impossible to observe "task still in progress" state between the
response and background completion; calling create_and_run_task/
_run_agent_task/get_task directly, with a real `BackgroundTasks()` instance
under the test's own control, gives precise, deterministic control over
when the background execution actually runs.

No real Postgres, Gemini, or Docker calls anywhere: the agent graph's own
`agent_app.astream` is patched with a controlled fake async generator, and
both the request-scoped and background database sessions are lightweight
fakes mirroring test_task_review.py's own `_make_db` pattern.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.api.v1 import agent as agent_module
from app.api.v1.agent import (
    CreateTaskRequest,
    _run_agent_task,
    create_and_run_task,
    get_task,
)
from app.db.models.repository import Repository
from app.db.models.task import Task


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------
def _make_db(
    task: Optional[Task] = None,
    repo: Optional[Repository] = None,
    assign_id_on_refresh: Optional[int] = None,
) -> MagicMock:
    """Mock async session dispatching scalar results by selected entity --
    mirrors test_task_review.py's own `_make_db` helper.

    `assign_id_on_refresh`, when given, makes `db.refresh(obj)` simulate a
    real DB assigning an autoincrement primary key -- needed only for
    create_and_run_task's own freshly-constructed Task (never looked up via
    `db.execute`, so the `task=` argument above doesn't apply to it)."""
    db = MagicMock()

    async def execute(stmt):
        res = MagicMock()
        entity = stmt.column_descriptions[0]["entity"]
        res.scalar_one_or_none.return_value = task if entity is Task else repo
        return res

    async def refresh(obj):
        if assign_id_on_refresh is not None and getattr(obj, "id", None) is None:
            obj.id = assign_id_on_refresh

    db.execute = AsyncMock(side_effect=execute)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock(side_effect=refresh)
    return db


def _make_session_cm(session: MagicMock) -> MagicMock:
    """Wrap a fake session in an async context manager, mimicking
    `async with AsyncSessionLocal() as db:`."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _fake_astream_factory(steps: List[Dict[str, Any]], delay: float = 0.0):
    """Builds a replacement for `agent_app.astream` that yields exactly the
    given {node_name: partial_state} steps -- the same shape LangGraph's
    real `stream_mode="updates"` yields (verified empirically against the
    installed langgraph version before this implementation)."""

    async def _fake_astream(initial_state, **kwargs):
        for step in steps:
            if delay:
                await asyncio.sleep(delay)
            yield step

    return _fake_astream


def _make_workspace_manager(workspace: Path) -> MagicMock:
    manager_cls = MagicMock()
    instance = MagicMock()
    instance.create_workspace.return_value = workspace
    instance.cleanup_workspace.return_value = True
    manager_cls.return_value = instance
    return manager_cls


def _base_payload(repository_id: int = 1) -> CreateTaskRequest:
    return CreateTaskRequest(
        repository_id=repository_id,
        title="Fix the bug",
        description="Something is broken",
    )


# ===========================================================================
# 1. NON-BLOCKING: POST /tasks returns before the agent graph runs at all.
# ===========================================================================
@pytest.mark.asyncio
async def test_create_and_run_task_returns_before_graph_executes(tmp_path):
    repo = Repository(id=1, name="repo", local_path=str(tmp_path / "source"))
    db = _make_db(repo=repo, assign_id_on_refresh=1)
    background_tasks = BackgroundTasks()

    fake_astream = MagicMock(side_effect=_fake_astream_factory([]))
    manager_cls = _make_workspace_manager(tmp_path / "workspace")

    with patch.object(agent_module, "WorkspaceManager", manager_cls), patch.object(
        agent_module.agent_app, "astream", fake_astream
    ):
        result = await create_and_run_task(_base_payload(), db, background_tasks)

    assert result.id == 1
    assert result.status == "investigating"
    # The handler returned without the background task ever having been
    # invoked -- proving it did not wait for (or even start) the graph.
    fake_astream.assert_not_called()
    # Exactly one background task was scheduled for Starlette to run after
    # the response is sent.
    assert len(background_tasks.tasks) == 1


# ===========================================================================
# 2. PROGRESS: multiple node updates must persist multiple intermediate
# statuses before the terminal status is reached.
# ===========================================================================
@pytest.mark.asyncio
async def test_run_agent_task_persists_intermediate_statuses_in_order(tmp_path):
    task = Task(id=7, repository_id=1, title="t", description="d", status="investigating")
    background_db = _make_db(task=task)
    session_local = MagicMock(return_value=_make_session_cm(background_db))

    committed_statuses: List[str] = []
    background_db.commit.side_effect = lambda: committed_statuses.append(task.status)

    steps = [
        {"investigate": {"status": "investigating"}},
        {"retrieve": {"status": "retrieved", "retrieved_context": []}},
        {"plan": {"status": "planning", "repair_plan": "p", "proposed_patches": []}},
        {"finalize": {"outcome": "NO_CHANGE_NEEDED", "outcome_detail": "already correct"}},
    ]
    manager_cls = _make_workspace_manager(tmp_path)

    with patch.object(agent_module, "AsyncSessionLocal", session_local), patch.object(
        agent_module, "WorkspaceManager", manager_cls
    ), patch.object(agent_module.agent_app, "astream", MagicMock(side_effect=_fake_astream_factory(steps))):
        await _run_agent_task(
            task_id=7,
            workspace=tmp_path,
            task_description="d",
            test_target=None,
            max_attempts=3,
            repository_id=1,
        )

    # "investigating" is skipped (already the task's status, per
    # _persist_task_status's no-redundant-write guard) -- "retrieved" and
    # "planning" are both observed as real intermediate states BEFORE the
    # terminal "no_change_needed" write.
    assert committed_statuses == ["retrieved", "planning", "no_change_needed"]
    assert task.status == "no_change_needed"


# ===========================================================================
# 3. SESSION SAFETY: the background execution must use its own session, and
# must never touch the request-scoped session.
# ===========================================================================
@pytest.mark.asyncio
async def test_run_agent_task_uses_its_own_session_not_the_request_session(tmp_path):
    task = Task(id=3, repository_id=1, title="t", description="d", status="investigating")
    request_db = _make_db(task=task)
    background_db = _make_db(task=task)
    session_local = MagicMock(return_value=_make_session_cm(background_db))

    steps = [{"finalize": {"outcome": "FAILED", "outcome_detail": "no fix"}}]
    manager_cls = _make_workspace_manager(tmp_path)

    with patch.object(agent_module, "AsyncSessionLocal", session_local), patch.object(
        agent_module, "WorkspaceManager", manager_cls
    ), patch.object(agent_module.agent_app, "astream", MagicMock(side_effect=_fake_astream_factory(steps))):
        await _run_agent_task(
            task_id=3,
            workspace=tmp_path,
            task_description="d",
            test_target=None,
            max_attempts=3,
            repository_id=1,
        )

    # The patched AsyncSessionLocal (the background execution's own
    # sessionmaker) was actually used to read and commit the task.
    session_local.assert_called_once()
    background_db.execute.assert_called()
    background_db.commit.assert_called()
    # _run_agent_task's signature takes no `db` parameter at all -- it is
    # structurally incapable of reusing the request's session -- and this
    # confirms the request-scoped session was never touched as a result.
    request_db.execute.assert_not_called()
    request_db.commit.assert_not_called()
    # Reached this status because the graph genuinely finalized as FAILED
    # (not because of some other, unrelated exception) -- the cleanup path
    # ran cleanly via the mocked WorkspaceManager.
    assert task.status == "failed"
    manager_cls.return_value.cleanup_workspace.assert_called_once_with(tmp_path)


# ===========================================================================
# 4. BACKGROUND EXCEPTION: a crash during graph execution must never escape
# _run_agent_task, and must still reach a terminal "failed" status with
# cleanup performed.
# ===========================================================================
@pytest.mark.asyncio
async def test_run_agent_task_exception_reaches_failed_and_cleans_up(tmp_path):
    task = Task(id=9, repository_id=1, title="t", description="d", status="investigating")
    background_db = _make_db(task=task)
    session_local = MagicMock(return_value=_make_session_cm(background_db))
    manager_cls = _make_workspace_manager(tmp_path)

    async def _raising_astream(initial_state, **kwargs):
        raise RuntimeError("boom")
        yield {}  # pragma: no cover -- makes this a generator function

    with patch.object(agent_module, "AsyncSessionLocal", session_local), patch.object(
        agent_module, "WorkspaceManager", manager_cls
    ), patch.object(agent_module.agent_app, "astream", _raising_astream):
        # Must not raise -- an exception here must never become an
        # unhandled process-level failure.
        await _run_agent_task(
            task_id=9,
            workspace=tmp_path,
            task_description="d",
            test_target=None,
            max_attempts=3,
            repository_id=1,
        )

    assert task.status == "failed"
    assert "boom" in task.test_output
    manager_cls.return_value.cleanup_workspace.assert_called_once_with(tmp_path)


# ===========================================================================
# 5. RESPONSE/ROUTING: task creation returns a task_id immediately, usable
# by the frontend's existing router.push(`/repositories/${id}/tasks/${created.id}`).
# ===========================================================================
@pytest.mark.asyncio
async def test_create_and_run_task_response_has_id_immediately(tmp_path):
    repo = Repository(id=2, name="repo", local_path=str(tmp_path / "source"))
    db = _make_db(repo=repo, assign_id_on_refresh=42)
    background_tasks = BackgroundTasks()
    manager_cls = _make_workspace_manager(tmp_path / "workspace")

    with patch.object(agent_module, "WorkspaceManager", manager_cls), patch.object(
        agent_module.agent_app, "astream", MagicMock(side_effect=_fake_astream_factory([]))
    ):
        result = await create_and_run_task(_base_payload(repository_id=2), db, background_tasks)

    # Every field TaskResponse (and the frontend's Task type) expects is
    # present and correctly typed immediately -- no change to the response
    # schema was needed for this phase. (`attempts`/`patch_content`/etc.
    # default application is unrelated, pre-existing SQLAlchemy behavior,
    # not exercised by this fake session -- this test is only about `id`
    # and `status` being correct immediately.)
    assert isinstance(result.id, int) and result.id == 42
    assert result.repository_id == 2
    assert result.status == "investigating"
    assert result.patch_content is None
    assert result.test_output is None
    assert result.pr_url is None


# ===========================================================================
# 6. ARCHITECTURAL CHECK: GET /tasks/{id} remains responsive while a
# background task is executing -- verified directly with real asyncio
# concurrency, not assumed from LangGraph's internal node dispatch.
# ===========================================================================
@pytest.mark.asyncio
async def test_get_task_remains_responsive_during_background_execution(tmp_path):
    task = Task(id=11, repository_id=1, title="t", description="d", status="investigating")
    background_db = _make_db(task=task)
    session_local = MagicMock(return_value=_make_session_cm(background_db))

    steps = [
        {"investigate": {"status": "investigating"}},
        {"retrieve": {"status": "retrieved", "retrieved_context": []}},
        {"plan": {"status": "planning", "repair_plan": "p", "proposed_patches": []}},
        {"finalize": {"outcome": "NO_CHANGE_NEEDED", "outcome_detail": "already correct"}},
    ]
    manager_cls = _make_workspace_manager(tmp_path)

    with patch.object(agent_module, "AsyncSessionLocal", session_local), patch.object(
        agent_module, "WorkspaceManager", manager_cls
    ), patch.object(agent_module.agent_app, "astream", MagicMock(side_effect=_fake_astream_factory(steps, delay=0.2))):
        bg = asyncio.create_task(
            _run_agent_task(
                task_id=11,
                workspace=tmp_path,
                task_description="d",
                test_target=None,
                max_attempts=3,
                repository_id=1,
            )
        )

        # Give the background task time to start its first sleep, then
        # issue a real, concurrent GET while it is still mid-run.
        await asyncio.sleep(0.05)
        get_db = _make_db(task=task)
        loop = asyncio.get_event_loop()
        start = loop.time()
        fetched = await asyncio.wait_for(get_task(11, get_db), timeout=0.5)
        elapsed = loop.time() - start

        # The GET returned quickly -- well under the ~0.8s total background
        # duration (4 steps x 0.2s delay) -- proving the event loop served
        # it without waiting for the background run to finish.
        assert elapsed < 0.3
        # And it observed a genuine intermediate (non-terminal) state, not
        # a task that had secretly already finished.
        assert fetched.status in ("investigating", "retrieved")

        await bg  # let the background task finish cleanly before teardown

    assert task.status == "no_change_needed"
