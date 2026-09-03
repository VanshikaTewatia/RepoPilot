"""API v1 router aggregator."""

from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.repositories import router as repositories_router
from app.api.v1.rag import router as rag_router
from app.api.v1.agent import router as agent_router
from app.api.v1.qa import router as qa_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health_router)
api_v1_router.include_router(repositories_router)
api_v1_router.include_router(rag_router)
api_v1_router.include_router(agent_router)
api_v1_router.include_router(qa_router)

__all__ = ["api_v1_router"]
