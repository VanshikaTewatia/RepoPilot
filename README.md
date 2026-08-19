# RepoPilot — Autonomous AI Software Engineer

> **RepoPilot** is an autonomous AI software engineering agent designed for deep repository comprehension, syntax-aware AST indexing, code-grounded RAG retrieval, and isolated sandbox test-driven self-correction.

---

## 1. Problem Statement

Modern software development workflows require engineering assistants to understand complex dependencies across dozens of files, accurately pinpoint the root cause of bugs, propose surgical modifications, and verify their correctness against existing test suites.

Generic LLM chat interfaces fail at this task because:
1. **Arbitrary Character Splitting**: Traditional naive RAG splits code by arbitrary token/character lengths (e.g. 500 characters), severing function signatures, splitting class definitions, and cutting docstrings in half.
2. **Hallucinated Edits without Verification**: Assistants propose code changes in isolation without executing tests, leading to subtle regressions, syntax errors, and broken dependencies.
3. **Unbounded Retries & Context Drift**: Without structured state machines, conversational agents enter uncontrolled loops, repeating identical erroneous patches and compounding errors.
4. **Security Risks**: Executing untrusted code or allowing LLM tools to perform arbitrary filesystem operations introduces critical path traversal and unauthorized shell execution vulnerabilities.

---

## 2. RepoPilot Solution & Architecture

RepoPilot resolves these challenges by combining:
- **Tree-sitter AST Parsing**: Generates semantic, syntax-bounded chunks (classes, methods, functions, import headers) with exact 1-indexed line numbers and cryptographic SHA-256 change hashes.
- **PostgreSQL 16 + `pgvector`**: High-performance unified vector similarity search (768-dim Gemini embeddings) with relational task tracking.
- **LangGraph State Machine**: Deterministic, multi-stage reasoning loop (`Investigate -> Retrieve -> Plan -> Edit -> Test -> Verify -> Self-Correct`).
- **3-Attempt Self-Correction Budget**: If tests fail, the failure output is fed directly into the error analysis node to revise the patch up to a maximum of 3 attempts.
- **Isolated Docker Test Sandbox**: Ephemeral containerized test runner (`python:3.11-slim`) with non-root execution, network disabled (`network_mode="none"`), CPU/memory quotas, and 45-second execution timeouts.

```mermaid
graph TD
    A[Start: User Request / Bug Description] --> B[Investigate: File Discovery & Keyword Search]
    B --> C[Retrieve: pgvector Dense Cosine Similarity Search]
    C --> D[Plan: Surgical Patch Strategy]
    D --> E[Edit: Apply Targeted Line-Range Patch]
    E --> F[Test: Ephemeral Docker Sandbox pytest]
    F --> G{Verify: Tests Passed?}
    G -- Yes --> H[Human-in-the-Loop Approval]
    G -- No, Attempts < 3 --> I[Analyze Failure: Parse Traceback & Refine Plan]
    I --> E
    G -- No, Attempts >= 3 --> J[Task Failed: Output Traceback & Diff]
    H --> K[Generate Unified Git Diff & Complete]
```

---

## 3. Core Subsystems

### A. Tree-sitter AST Syntax Indexer
Instead of character-count splitting, RepoPilot utilizes Tree-sitter Abstract Syntax Tree (AST) grammars to parse source files into atomic units:
- Functions (`function_definition`)
- Classes (`class_definition`)
- Methods (functions nested within classes)
- Import blocks (`import_statement`, `import_from_statement`)

Each symbol preserves its exact start and end line ranges (e.g., `src/calculator.py:12-18`) and computes a SHA-256 content hash to eliminate redundant embedding generation for unchanged files.

### B. Code-Aware Semantic RAG
- **Query Embedding**: Generated via Google Gemini (`text-embedding-004`).
- **Similarity Search**: Cosine distance ranking using `pgvector` (`Vector(768)`).
- **Grounded Citations**: Answers strictly cite specific line ranges and files without exposing internal chain-of-thought traces.

### C. LangGraph Multi-Step State Machine
The agent loop is managed via a strongly-typed `AgentState`:
```python
class AgentState(TypedDict, total=False):
    task_id: int
    repository_id: int
    workspace_dir: str
    task_description: str
    status: str
    attempt_count: int
    max_attempts: int
    test_target: Optional[str]
    investigation_findings: str
    retrieved_context: List[Dict[str, Any]]
    repair_plan: str
    proposed_patches: List[Dict[str, Any]]
    test_results: Optional[Dict[str, Any]]
    error_analysis: Optional[str]
    is_verified: bool
    messages: List[Dict[str, Any]]
```

### D. Secure Agent Tools
All tool interactions operate under strict security containment:
1. **`list_files`**: Safe directory listing excluding VCS and binary files.
2. **`search_code`**: Literal and regex pattern search across workspace code.
3. **`read_file`**: Bounded line-range reader with strict traversal prevention.
4. **`apply_patch`**: Surgical line-range replacement or unified patch applier.
5. **`run_tests`**: Sandbox execution runner with zero network access and hard execution timeouts.

---

## 4. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Application status, environment, version, and database connectivity. |
| `POST` | `/api/v1/repositories` | Register a local repository for tracking. |
| `GET` | `/api/v1/repositories` | List all registered repositories. |
| `POST` | `/api/v1/repositories/{id}/index` | Run Tree-sitter AST parsing and vector indexing. |
| `POST` | `/api/v1/rag/ask` | Query codebase using semantic RAG with line citations. |
| `POST` | `/api/v1/tasks` | Create and execute a LangGraph autonomous engineering task. |
| `POST` | `/api/v1/tasks/fix` | Convenience endpoint to trigger autonomous bug fixing. |
| `GET` | `/api/v1/tasks/{id}` | Retrieve task execution status, attempts, and test results. |
| `GET` | `/api/v1/tasks/{id}/diff` | Retrieve generated unified Git diff for the fix. |
| `POST` | `/api/v1/tasks/{id}/approve` | User approval of verified repair. |

---

## 5. Evaluation Benchmark & Actual Results

RepoPilot includes an automated evaluation harness (`benchmark/runner.py`) testing realistic bug repair scenarios across isolated workspaces.

### Actual Benchmark Metrics (Recorded from `benchmark/results.json`):
- **Total Tasks Evaluated**: `5`
- **Retrieval Success Rate**: `100.0%`
- **Pass@1 Rate**: `20.0%`
- **Pass@3 Rate**: `40.0%`
- **Average Attempts / Task**: `2.4`
- **Average Execution Duration**: `4.71 seconds`

### Benchmark Task Breakdown:
| Task ID | Description | Retrieval | Pass@1 | Pass@3 | Attempts | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `task-1-vip-discount` | Fix VIP discount across subtotal | ✅ | ❌ | ✅ | 2 | 3.48s |
| `task-2-cancel-idempotency` | Enforce cancel_order idempotency | ✅ | ❌ | ❌ | 3 | 4.42s |
| `task-3-payment-expiry` | Validate card expiry month range | ✅ | ✅ | ✅ | 1 | 2.24s |
| `task-4-stock-reservation` | Stock reservation validation | ✅ | ❌ | ❌ | 3 | 7.32s |
| `task-5-tax-computation` | Tax calculation on discounted total | ✅ | ❌ | ❌ | 3 | 6.11s |

---

## 6. Getting Started & Setup

### Prerequisites
- Python 3.11+
- Docker Desktop (with WSL 2 engine on Windows)
- PostgreSQL 16 + pgvector (via `docker-compose`)

### 1. Database Setup
Start PostgreSQL with the pgvector extension:
```bash
docker-compose up -d postgres
```

### 2. Backend Environment Setup
Create a `.env` file in `backend/` (or copy `backend/.env.example`):
```env
APP_NAME=RepoPilot
ENVIRONMENT=development
LOG_LEVEL=INFO
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql+asyncpg://postgres:postgrespassword@localhost:5432/repopilot
```

### 3. Install Dependencies & Run Migrations
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
alembic upgrade head
```

### 4. Start Backend Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run Backend Test Suite
```bash
pytest
```

### 6. Run Evaluation Benchmark
```bash
python benchmark/runner.py
```

### 7. Open Dashboard UI
Open `frontend/public/index.html` in your web browser, or launch the Next.js frontend:
```bash
cd frontend
npm install
npm run dev
```

---

## 7. Limitations & Future Roadmap

- **Multi-Language Support**: AST parsing currently implements Python grammars. Future extensions will incorporate TypeScript/JavaScript, Go, and Rust Tree-sitter parsers.
- **GitHub PR Integration**: Automated branching, remote pushing, and Pull Request generation with human-in-the-loop sign-off.
- **Live Test Suite Execution**: Deep integration with pre-built test containers for language runtimes beyond Python.
