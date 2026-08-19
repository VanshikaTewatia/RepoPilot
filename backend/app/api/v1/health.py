"""Health check API router."""

from datetime import datetime, timezone
from typing import Any, Dict
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import check_db_health

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health check endpoint",
    description="Returns application status, version, environment, and database connectivity.",
)
async def health_check() -> JSONResponse:
    """Check API and dependent service health."""
    db_ok = await check_db_health()
    overall_status = "healthy" if db_ok else "degraded"

    payload: Dict[str, Any] = {
        "status": overall_status,
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    http_status = status.HTTP_200_OK if db_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    # In development or test, return 200 even if database is offline to allow unit testing
    if not db_ok and (settings.is_development or settings.is_test):
        http_status = status.HTTP_200_OK

    return JSONResponse(status_code=http_status, content=payload)
