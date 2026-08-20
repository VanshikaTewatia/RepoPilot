"""LangGraph agent task execution, diff inspection, and approval routes."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models.repository import Repository
from app.db.models.task import Task
from app.services.agent.graph import agent_app
from app.services.git_service import GitService

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


class TaskApproveResponse(BaseModel):
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

    initial_state = {
        "task_id": task.id,
        "repository_id": repo.id,
        "workspace_dir": repo.local_path,
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
        task.status = "human_approval_required" if final_state.get("is_verified") else "failed"
        task.attempts = final_state.get("attempt_count", 0)
        test_res = final_state.get("test_results") or {}
        task.test_output = test_res.get("output")
        task.patch_content = GitService.get_workspace_diff(repo.local_path)
        await db.commit()
        await db.refresh(task)
    except Exception as e:
        task.status = "failed"
        task.test_output = str(e)
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
    """Retrieve the generated git diff for a task."""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    repo_res = await db.execute(select(Repository).where(Repository.id == task.repository_id))
    repo = repo_res.scalar_one_or_none()
    workspace_dir = repo.local_path if repo else ""

    diff = task.patch_content or GitService.get_workspace_diff(workspace_dir)
    changed_files = GitService.get_changed_files(workspace_dir) if workspace_dir else []

    return {
        "task_id": task.id,
        "status": task.status,
        "diff": diff,
        "changed_files": changed_files,
    }


@router.post("/{task_id}/approve", response_model=TaskApproveResponse)
async def approve_task_fix(task_id: int, db: SessionDep) -> Dict[str, Any]:
    """Human-in-the-loop approval of the agent's verified fix."""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in ("human_approval_required", "verified"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task with status '{task.status}' cannot be approved (must be 'human_approval_required')",
        )

    task.status = "approved"
    await db.commit()
    await db.refresh(task)

    return {
        "task_id": task.id,
        "status": "approved",
        "message": "Fix has been approved by the user.",
    }
