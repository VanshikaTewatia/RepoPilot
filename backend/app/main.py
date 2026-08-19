"""Main FastAPI application entry point for RepoPilot."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api import api_v1_router
from app.api.v1.health import router as health_router
from app.api.v1.repositories import router as repositories_router
from app.api.v1.rag import router as rag_router
from app.api.v1.agent import router as agent_router
from app.core.config import settings
from app.core.logging import logger
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle handler."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version} [{settings.environment}]")
    await init_db()
    yield
    logger.info("Shutting down RepoPilot application.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI Software Engineering Agent backend for repository understanding, RAG, and self-correction.",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount base health check
    app.include_router(health_router)

    # Mount versioned API routes (/api/v1/...)
    app.include_router(api_v1_router)

    # Also mount direct /api/... aliases for direct MVP endpoints
    app.include_router(repositories_router, prefix="/api")
    app.include_router(rag_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/health")

    return app


app = create_app()
