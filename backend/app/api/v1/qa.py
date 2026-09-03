"""Deep Codebase Q&A API endpoint.

Exposes ``app.services.qa.service.ask_codebase`` (classifier -> investigator
-> answerer) over HTTP. Additive only: does not modify and is not used by
the existing ``POST /api/rag/ask`` endpoint (app.api.v1.rag) or
``CodeRetriever.answer_question``'s plain single-shot path -- both continue
to work exactly as before.

The request never accepts a filesystem path. ``workspace_dir`` is always
resolved server-side from the ``Repository`` row looked up by
``repository_id`` (the same pattern used by ``app.api.v1.agent`` for task
creation/approval) -- a caller can only select an already-registered
repository by its database id, never point the investigation at an
arbitrary path.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.deps import SessionDep
from app.db.models.repository import Repository
from app.services.qa import service as qa_service
from app.services.qa.models import QAAnswer
from app.services.qa.service import QuestionValidationError
from app.services.rag.retriever import CodeRetriever

router = APIRouter(prefix="/qa", tags=["Deep Codebase Q&A"])


def get_code_retriever() -> CodeRetriever:
    """FastAPI dependency for the retriever passed to ``ask_codebase``.

    A plain function (mirroring ``app.api.deps.SessionDep``'s pattern) so
    tests can override it via ``app.dependency_overrides`` instead of
    exercising real pgvector/Gemini calls -- the same DI convention already
    used for the database session.
    """
    return CodeRetriever()


RetrieverDep = Annotated[CodeRetriever, Depends(get_code_retriever)]


class QAAskRequest(BaseModel):
    question: str = Field(..., example="How does OrderService calculate discounts and taxes?")
    repository_id: int = Field(..., example=1)


@router.post("/ask", response_model=QAAnswer)
async def ask_deep_qa(
    payload: QAAskRequest,
    db: SessionDep,
    retriever: RetrieverDep,
) -> QAAnswer:
    """Run the full Deep Codebase Q&A pipeline (classify -> investigate ->
    answer) for a question about an already-registered repository.

    ``repository_id`` is resolved to ``Repository.local_path`` server-side --
    the request body has no path/workspace field, so no filesystem location
    supplied by the browser is ever trusted (see module docstring).
    """
    repo_res = await db.execute(select(Repository).where(Repository.id == payload.repository_id))
    repo = repo_res.scalar_one_or_none()
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    try:
        return await qa_service.ask_codebase(
            payload.question,
            workspace_dir=repo.local_path,
            repository_id=payload.repository_id,
            db=db,
            retriever=retriever,
        )
    except QuestionValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)) from e
