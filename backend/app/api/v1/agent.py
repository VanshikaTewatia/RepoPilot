"""LangGraph agent task execution, diff inspection, and approval routes."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import SessionDep
from app.core.logging import logger
from app.db.models.repository import Repository
from app.db.models.task import Task
from app.services.agent.graph import agent_app
from app.services.git_service import GitService
from app.services.workspace_manager import WorkspaceManager

router = APIRouter(prefix="/tasks", tags=["Agent Tasks"])


class CreateTaskRequest(BaseModel):
    repository_id: int
    title: str = Field(..., example="Fix null pointer exception in auth handler")
    description: str = Field(..., example="When user is not found, get_user returns None causing crash")
    test_target: Optional[str] = Field(default=None, example="tests/test_auth.py::test_user_not_found")
    max_attempts: int = Field(default=3, ge=1, le=5)


class FixBugRequest(BaseModel):
    repository_id: int
    issue_description: str = Field(..., example="Fix division by zero when denominator is 0 in calculator")
    test_target: Optional[str] = Field(default=None, example="tests/test_calculator.py::test_zero_div")
    max_attempts: int = Field(default=3, ge=1, le=5)


class TaskResponse(BaseModel):
    id: int
    repository_id: int
    title: str
    description: str
    status: str
    attempts: int
    patch_content: Optional[str]
    test_output: Optional[str]
    pr_url: Optional[str]


class TaskDiffResponse(BaseModel):
    task_id: int
    status: str
    diff: str
    changed_files: List[str]


class TaskReviewResponse(BaseModel):
    task_id: int
    status: str
    message: str


@router.post("", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_and_run_task(
    payload: CreateTaskRequest,
    db: SessionDep,
) -> Task:
    """Create a new debugging/feature task and execute the LangGraph loop."""
    repo_res = await db.execute(select(Repository).where(Repository.id == payload.repository_id))
    repo = repo_res.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    task = Task(
        repository_id=payload.repository_id,
        title=payload.title,
        description=payload.description,
        status="investigating",
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    # Create an isolated copy of the repository so the live repo is never
    # modified during investigate/retrieve/plan/edit/test/retry.
    manager = WorkspaceManager()
    try:
        workspace = manager.create_workspace(task.id, repo.local_path)
    except Exception as e:
        task.status = "failed"
        task.test_output = f"Workspace isolation failed: {e}"
        await db.commit()
        await db.refresh(task)
        return task

    initial_state = {
        "task_id": task.id,
        "repository_id": repo.id,
        "workspace_dir": str(workspace),
        "task_description": task.description,
        "test_target": payload.test_target,
        "status": "pending",
        "attempt_count": 0,
        "max_attempts": payload.max_attempts,
        "investigation_findings": "",
        "retrieved_context": [],
        "repair_plan": "",
        "proposed_patches": [],
        "test_results": None,
        "error_analysis": None,
        "is_verified": False,
        "messages": [],
    }

    try:
        final_state = await agent_app.ainvoke(initial_state)
        verified = bool(final_state.get("is_verified"))
        task.status = "human_approval_required" if verified else "failed"
        task.attempts = final_state.get("attempt_count", 0)
        test_res = final_state.get("test_results") or {}
        task.test_output = test_res.get("output")
        if verified:
            # Stage all workspace changes (incl. new files) so the captured
            # diff is complete and directly applicable via `git apply`.
            GitService.stage_all_changes(workspace)
            # Capture and persist review information from the isolated
            # workspace; the live repository is never read for this.
            task.patch_content = GitService.get_workspace_diff(workspace)
            task.changed_files = GitService.get_changed_files(workspace)
            # Keep the workspace (and its path) while the fix awaits review.
            task.workspace_path = str(workspace)
        else:
            manager.cleanup_workspace(workspace)
        await db.commit()
        await db.refresh(task)
    except Exception as e:
        task.status = "failed"
        task.test_output = str(e)
        try:
            manager.cleanup_workspace(workspace)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up workspace '{workspace}': {cleanup_error}")
        await db.commit()
        await db.refresh(task)

    return task


@router.post("/fix", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def fix_bug_endpoint(
    payload: FixBugRequest,
    db: SessionDep,
) -> Task:
    """Convenience endpoint to initiate a bug-fix agent task."""
    create_req = CreateTaskRequest(
        repository_id=payload.repository_id,
        title=payload.issue_description[:60],
        description=payload.issue_description,
        test_target=payload.test_target,
        max_attempts=payload.max_attempts,
    )
    return await create_and_run_task(create_req, db)


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: SessionDep) -> Task:
    """Get status and details of a specific task."""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/{task_id}/diff", response_model=TaskDiffResponse)
async def get_task_diff(task_id: int, db: SessionDep) -> Dict[str, Any]:
    """Retrieve the persisted review diff for a task.

    Uses the patch/changed files captured at review time; never recomputes a
    potentially stale diff from the live repository while a review is pending.
    """
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    diff = task.patch_content
    changed_files = task.changed_files

    # Legacy fallback only: tasks reviewed before persistence existed.
    if diff is None or changed_files is None:
        repo_res = await db.execute(select(Repository).where(Repository.id == task.repository_id))
        repo = repo_res.scalar_one_or_none()
        workspace_dir = task.workspace_path or (repo.local_path if repo else "")
        if workspace_dir:
            if diff is None:
                diff = GitService.get_workspace_diff(workspace_dir)
            if changed_files is None:
                changed_files = GitService.get_changed_files(workspace_dir)

    return {
        "task_id": task.id,
        "status": task.status,
        "diff": diff or "",
        "changed_files": changed_files or [],
    }


@router.post("/{task_id}/approve", response_model=TaskReviewResponse)
async def approve_task_fix(task_id: int, db: SessionDep) -> Dict[str, Any]:
    """Apply the stored patch to the original repository after human approval.

    The stored diff is applied safely (`git apply --check` first): if it no
    longer fits the current repository state, a conflict is returned, the task
    stays pending review, and the isolated workspace is preserved.
    """
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "human_approval_required":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task with status '{task.status}' cannot be approved (must be 'human_approval_required')",
        )

    repo_res = await db.execute(select(Repository).where(Repository.id == task.repository_id))
    repo = repo_res.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=404, detail="Repository not found")

    if not task.patch_content or not task.patch_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task has no stored patch to apply",
        )
    if not repo.local_path or not Path(repo.local_path).is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Original repository path is not accessible: {repo.local_path}",
        )

    # Apply the persisted diff to the ORIGINAL repository. The isolated
    # workspace is never used as an apply target or file source.
    applied, error = GitService.apply_diff(repo.local_path, task.patch_content)
    if not applied:
        # Conflict: leave the task pending review and keep the workspace.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patch cannot be applied cleanly to the current repository state: {error}",
        )

    task.status = "approved"

    if task.workspace_path:
        try:
            WorkspaceManager().cleanup_workspace(task.workspace_path)
            task.workspace_path = None
        except Exception as e:
            logger.warning(f"Failed to clean up workspace '{task.workspace_path}': {e}")

    await db.commit()
    await db.refresh(task)

    return {
        "task_id": task.id,
        "status": "approved",
        "message": "Patch applied to the original repository.",
    }


@router.post("/{task_id}/reject", response_model=TaskReviewResponse)
async def reject_task_fix(task_id: int, db: SessionDep) -> Dict[str, Any]:
    """Reject a pending fix: discard the isolated workspace, keep the original
    repository untouched."""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "human_approval_required":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task with status '{task.status}' cannot be rejected (must be 'human_approval_required')",
        )

    if task.workspace_path:
        try:
            WorkspaceManager().cleanup_workspace(task.workspace_path)
            task.workspace_path = None
        except Exception as e:
            logger.warning(f"Failed to clean up workspace '{task.workspace_path}': {e}")

    task.status = "rejected"
    await db.commit()
    await db.refresh(task)

    return {
        "task_id": task.id,
        "status": "rejected",
        "message": "Fix rejected; isolated workspace discarded and original repository untouched.",
    }
