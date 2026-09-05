"""Short-lived database session acquisition for LangGraph agent nodes.

Phase 6B: ``retrieve_node`` needs an ``AsyncSession`` to call
``CodeRetriever.retrieve_chunks`` for semantic retrieval, but a LangGraph
node runs outside FastAPI's request-scoped ``Depends(get_db)`` dependency
injection. This module reuses the exact session-acquisition primitive
``app.db.session.get_db`` already wraps (``async with AsyncSessionLocal()
as session:``) without the generator/DI wrapper, so a node can open one
directly.

Deliberately short-lived by contract: a caller must open, use, and close
the session within a single node invocation (never across the retry loop,
Docker/test/verification steps, or multiple attempts) -- see
``app.services.agent.graph.retrieve_node``, the only caller. This keeps
connection lifetime bounded and avoids holding a pooled connection idle
for a potentially long-running task.

Never raises: semantic retrieval is an enhancement, not a requirement (see
``CodeRetriever.retrieve_chunks``, which already treats ``db=None`` as
"skip semantic retrieval" rather than an error) -- if a session cannot be
acquired for any reason, ``open_session`` yields ``None`` instead of
propagating the failure, so a caller can degrade to keyword-only
investigation without any special-case handling.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.db.session import AsyncSessionLocal


@asynccontextmanager
async def open_session() -> AsyncIterator[Optional[AsyncSession]]:
    """Yield a short-lived ``AsyncSession``, or ``None`` if one could not be
    acquired -- never raises."""
    try:
        async with AsyncSessionLocal() as session:
            yield session
    except Exception as e:  # noqa: BLE001 -- session acquisition must never crash a node; degrade to None instead
        logger.warning(f"Could not open a database session for the agent: {e}")
        yield None
