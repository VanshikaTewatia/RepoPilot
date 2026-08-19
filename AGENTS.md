# AGENTS.md

## Project overview

RepoPilot is an autonomous AI software engineering agent with a modular monolith architecture: **FastAPI backend** + **PostgreSQL/pgvector** + **LangGraph agent state machine** + **Next.js frontend**.

- `backend/` — FastAPI app, services, DB models, agent graph, tests
- `frontend/` — Next.js 14 + Tailwind dashboard (single `page.tsx`)
- `benchmark/` — Evaluation harness with 5 bug-fix tasks (`runner.py`, `tasks.py`)
- `demo_repo/` — Sample e-commerce repo used by benchmark and frontend demo

## Commands

All backend commands run from the `backend/` directory with the venv activated:

```bash
# Setup (one-time)
docker-compose up -d postgres          # PostgreSQL 16 + pgvector on port 5432
cd backend && python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
alembic upgrade head                    # Apply DB migrations

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Tests (async auto-mode via pytest.ini)
pytest                                  # all tests
pytest tests/test_parser.py             # single file
pytest -k test_health                   # single test by name
```

Frontend (from `frontend/`):

```bash
npm install
npm run dev                             # Next.js dev server on port 3000
npm run lint
```

Benchmark (from repo root):

```bash
python benchmark/runner.py              # runs all 5 tasks, writes results.json
```

## Architecture

- **Agent graph** (`backend/app/services/agent/graph.py`): LangGraph StateGraph — investigate → retrieve → plan → edit → test → verify. Loops back through `analyze_failure` up to 3 attempts before failing.
- **Agent state** (`backend/app/services/agent/state.py`): `AgentState` TypedDict drives the entire workflow.
- **Agent tools** (`backend/app/services/agent/tools.py`): `list_files`, `search_code`, `read_file`, `apply_patch`, `run_tests`. The sandbox runs tests in ephemeral Docker containers (`python:3.11-slim`, network disabled, 45s timeout).
- **Code indexing** (`backend/app/services/indexing/`): Tree-sitter AST parsing for Python → syntax-bounded chunks with SHA-256 change hashes.
- **RAG** (`backend/app/services/rag/retriever.py`): pgvector cosine similarity search with 768-dim Gemini embeddings.
- **Config** (`backend/app/core/config.py`): Pydantic Settings loaded from `backend/.env` (and `../.env` fallback). All config keys are lowercase with underscores.
- **DB** (`backend/app/db/`): SQLAlchemy async + asyncpg. `init_db()` auto-enables pgvector extension and creates tables on startup.
- **API routes** (`backend/app/api/v1/`): health, repositories, rag, agent. Mounted at both `/api/v1/` and `/api/`.

## Key gotchas

- **Backend `.env` is required.** Copy `backend/.env.example` to `backend/.env` and set `GEMINI_API_KEY` and `DATABASE_URL`. The app reads `DATABASE_URL` from env, not from `alembic.ini`.
- **Database password differs between `.env.example` and `alembic.ini`.** `.env.example` uses `postgres:postgres`; `docker-compose.yml` and `alembic.ini` use `postgres:postgrespassword`. Match these or migrations will fail.
- **pytest.ini** sets `asyncio_mode = auto` — async test functions run without needing `@pytest.mark.asyncio`.
- **Frontend hardcodes `API_BASE = "http://localhost:8000/api/v1"`** in `page.tsx`. Backend must be on port 8000.
- **Benchmark runs from repo root** (`python benchmark/runner.py`), not from `backend/`. It manipulates `sys.path` to import backend modules.
- **`demo_repo/` is the test fixture** for both the benchmark and the frontend demo. Do not modify its test expectations when changing backend services.
- **Docker sandbox** runs with `network_mode="none"`, non-root user, 2 CPU / 2GB RAM limits, and 45s hard timeout. Tests inside the sandbox cannot access the network or the host filesystem.
- **CORS** allows `localhost:3000` and `127.0.0.1:3000` by default (matches frontend dev server).
