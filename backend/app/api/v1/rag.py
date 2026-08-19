"""Code-aware RAG question answering endpoints."""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import SessionDep
from app.services.rag.retriever import CodeRetriever

router = APIRouter(prefix="/rag", tags=["Code-Aware RAG"])


class AskRequest(BaseModel):
    query: str = Field(..., example="Where is the database session defined?")
    repository_id: int = Field(..., example=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunkResponse(BaseModel):
    file_path: str
    symbol_name: str | None
    symbol_type: str
    start_line: int
    end_line: int
    source_code: str
    similarity_score: float
    citation: str


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunkResponse]
    citations: List[str]


@router.post("/ask", response_model=AskResponse)
async def ask_codebase(
    payload: AskRequest,
    db: SessionDep,
) -> Dict[str, Any]:
    """Perform semantic code retrieval and synthesize an answer citing exact line ranges."""
    retriever = CodeRetriever()
    response = await retriever.answer_question(
        query=payload.query,
        repository_id=payload.repository_id,
        top_k=payload.top_k,
        db=db,
    )
    return response
