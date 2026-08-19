# RepoPilot — Current Project State & Handoff Document

**Date:** 2026-08-18  
**Status:** Pre-implementation / Architecture & Specification Phase  
**Target Environment:** Windows 11 + WSL 2 (Ubuntu) / Docker Desktop  

---

## 1. What Has Been Implemented So Far

- **Product & System Specification**: Fully analyzed the functional and technical requirements for RepoPilot (AI Software Engineering Agent for repo understanding, bug investigation, sandbox test execution, and human-in-the-loop PR creation).
- **Architectural Design**: Completed the comprehensive system blueprint, covering:
  - Component boundaries and responsibilities.
  - Data flow diagrams for code indexing and the 3-attempt iterative self-correction loop.
  - Security containment model for untrusted code execution.
  - Testing & AI evaluation strategy (Mini SWE-Bench style benchmarks).
  - Scope boundaries (MVP vs. Post-MVP).
- **Workspace State**: Git repository initialized at workspace root (`C:\Users\Lakshay\RepoPilot`). No code was written yet per initial instruction to hold execution until architectural review.

---

## 2. Files Created or Modified

| File | Status | Description |
| :--- | :--- | :--- |
| `PROJECT_STATUS.md` | **Created** | This project status and handoff document. |
| `.git/` | **Initialized** | Base Git repository metadata directory. |

---

## 3. Current Architecture Decisions

1. **Architecture Pattern**: **Modular Monolith** over microservices to maximize developer velocity and minimize orchestration overhead for a single developer.
2. **Backend**: **FastAPI (Python 3.11+)** with asynchronous request handling and Server-Sent Events (SSE) for streaming agent reasoning and tool traces.
3. **Database & Vector Search**: **PostgreSQL 16 with `pgvector`** for unified storage of relational metadata, chat history, tasks, and code vector embeddings (hybrid search: dense embeddings + PostgreSQL `tsvector` full-text search).
4. **Code Understanding & Chunking**: **Tree-sitter (Python bindings)** for syntax-aware Abstract Syntax Tree (AST) parsing and chunking along function/class/method boundaries (avoiding arbitrary character splits).
5. **AI Orchestration**: **LangGraph** (StateGraph) with Gemini LLM for structured, deterministic multi-step reasoning:
   - State machine stages: *Explore -> Locate -> Plan -> Edit -> Test (Docker) -> Analyze/Self-Correct (Max 3 attempts) -> User Approval -> PR Creation*.
6. **Execution Sandbox**: **Docker Engine API** running ephemeral, locked-down containers (`network_mode="none"`, non-root user `1000:1000`, 2 CPU / 2GB RAM limits, 45s hard timeout).
7. **Frontend**: **Next.js (App Router, TypeScript, Tailwind CSS)** featuring a chat interface, real-time agent trace viewer, and side-by-side Git diff viewer.
8. **Anti-Patterns Excluded**: No Kafka/RabbitMQ, no standalone vector database clusters (e.g. Pinecone/Milvus), no complex multi-agent frameworks (CrewAI/AutoGen), and no heavy Language Server Protocol (LSP) daemon processes in MVP.

---

## 4. What Remains to Be Implemented

### Backend
- [ ] Database layer: PostgreSQL + `pgvector` connection pool, SQLAlchemy models (`Repo`, `CodeChunk`, `Task`, `Interaction`), and Alembic migrations.
- [ ] Git service: Workspace manager, shallow clone engine, patch applier, and unified diff generator.
- [ ] Indexer service: Tree-sitter AST parser, syntax chunker, and batch Gemini embedding pipeline.
- [ ] Hybrid retriever: Reciprocal Rank Fusion / cosine distance + lexical `tsvector` search.
- [ ] Sandbox service: Docker container manager with security containment policies and test suite runner (`pytest`).
- [ ] Agent core: LangGraph `StateGraph`, node definitions (Investigator, Coder, Tester, Verifier), and tool bindings.
- [ ] API endpoints: FastAPI routes for repo connection, SSE chat/task streams, diff inspection, and PR approval.

### Frontend
- [ ] Next.js 14+ project setup with Tailwind CSS & UI components.
- [ ] Repository connection & indexing progress dashboard.
- [ ] Real-time agent chat stream with tool execution trace toggles.
- [ ] Side-by-side Diff Viewer with "Approve & Create PR" and "Request Changes" actions.

### Testing & Evaluation
- [ ] Unit & integration tests for AST chunker, diff generator, and Docker runner.
- [ ] 10-case Mini SWE-Bench benchmark test suite for Pass@1 / Pass@3 evaluation.

---

## 5. Exact Next Steps (Post-Reboot)

Once your system has restarted and WSL/Docker is ready:

1. **Verify Prerequisites**:
   - Check WSL 2 status: `wsl --status`
   - Check Docker Desktop integration: `docker info`
   - Verify Python 3.11+ and Node.js 18+ availability.
2. **Step 1 — Project Scaffolding & Database Setup**:
   - Create `docker-compose.yml` for local PostgreSQL + `pgvector`.
   - Setup `backend/` directory with `pyproject.toml` / `requirements.txt` (FastAPI, SQLAlchemy, asyncpg, pgvector, tree-sitter, langgraph, google-genai).
   - Configure database models and run initial Alembic migration.
3. **Step 2 — Git Manager & Tree-sitter AST Chunker**:
   - Implement local repo cloning and AST-aware Python file chunker using Tree-sitter.
   - Verify chunking with unit tests.
4. **Step 3 — Embedding & Hybrid Retriever Pipeline**:
   - Implement Gemini embedding generation and pgvector similarity search.
5. **Step 4 — Docker Sandbox Execution Engine**:
   - Build isolated container runner for test suite execution.
6. **Step 5 — LangGraph State Machine & Agent Loop**:
   - Assemble the 3-attempt self-correction bug fixing graph.

---

## 6. Current Errors or Blockers

- **Pending Host Restart**: WSL 2 installation is awaiting computer restart to enable virtualization and Docker container orchestration.
- **No Blockers in Codebase**: Clean state ready for implementation upon reboot.

---

## 7. Important Environment & Configuration Requirements

Create a `.env` file in `backend/` upon setup with the following variables:

```env
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Database (PostgreSQL + pgvector)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/repopilot

# GitHub Integration
GITHUB_TOKEN=your_github_personal_access_token_here

# Workspace & Sandbox Storage
WORKSPACE_DIR=C:/Users/Lakshay/RepoPilot/storage/workspaces
DOCKER_SANDBOX_IMAGE=python:3.11-slim

# App Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Required Host Tools
- **WSL 2** with default Ubuntu distribution.
- **Docker Desktop** (with *"Use the WSL 2 based engine"* enabled in Settings -> General).
- **Python 3.11+** (or via WSL 2 virtualenv / pyenv).
- **Node.js 18+** & `npm` / `pnpm`.
