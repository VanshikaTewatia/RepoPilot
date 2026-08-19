"""Unit tests for LangGraph state machine execution and 3-attempt retry loop."""

import tempfile
from pathlib import Path
import pytest

from app.services.agent.graph import agent_app


@pytest.mark.asyncio
async def test_agent_graph_success_flow():
    """Test full LangGraph state machine run with test passing on first attempt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Create a passing test file in workspace
        (workspace / "test_sample.py").write_text(
            "def test_ok():\n    assert 1 == 1\n",
            encoding="utf-8",
        )

        initial_state = {
            "task_id": 1,
            "repository_id": 1,
            "workspace_dir": str(workspace),
            "task_description": "Verify test suite works",
            "status": "pending",
            "attempt_count": 0,
            "max_attempts": 3,
            "investigation_findings": "",
            "retrieved_context": [],
            "repair_plan": "",
            "proposed_patches": [],
            "test_results": None,
            "error_analysis": None,
            "is_verified": False,
            "messages": [],
        }

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["attempt_count"] == 1
        assert final_state["is_verified"] is True
        assert final_state["status"] == "verified"


@pytest.mark.asyncio
async def test_agent_graph_failure_loop_stops_at_max_attempts():
    """Test that LangGraph stops at max_attempts when tests persistently fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        # Create a failing test file
        (workspace / "test_fail.py").write_text(
            "def test_bad():\n    assert 1 == 2\n",
            encoding="utf-8",
        )

        initial_state = {
            "task_id": 2,
            "repository_id": 1,
            "workspace_dir": str(workspace),
            "task_description": "Failing test task",
            "status": "pending",
            "attempt_count": 0,
            "max_attempts": 3,
            "investigation_findings": "",
            "retrieved_context": [],
            "repair_plan": "",
            "proposed_patches": [],
            "test_results": None,
            "error_analysis": None,
            "is_verified": False,
            "messages": [],
        }

        final_state = await agent_app.ainvoke(initial_state)

        assert final_state["is_verified"] is False
        assert final_state["attempt_count"] == 3
