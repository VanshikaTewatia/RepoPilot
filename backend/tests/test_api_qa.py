"""API-level tests for the Deep Codebase Q&A endpoint (POST /api/v1/qa/ask).

Exercises the endpoint through FastAPI's TestClient, not the service layer
directly (that is already covered by test_qa_service.py). No real Postgres
or Gemini calls are made: the database session and the ``CodeRetriever`` are
both overridden via ``app.dependency_overrides`` -- the same technique
``app.api.deps.SessionDep`` already establishes for ``get_db`` -- and the
default test-environment Gemini key (see conftest.py) makes
classify_question()/generate_answer() take their existing, deterministic
mock-mode fallback paths, exactly as test_qa_service.py relies on.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from app.api.v1 import qa as qa_module
from app.db.models.repository import Repository
from app.db.session import get_db
from app.main import app
from app.services.qa.answerer import NO_EVIDENCE_SUMMARY
from app.services.rag.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Shared fakes (mirrors the _FakeRetriever pattern already used in
# test_qa_service.py / test_qa_investigator.py)
# ---------------------------------------------------------------------------
class _FakeRetriever:
    def __init__(self, default_chunks: Optional[List[RetrievedChunk]] = None):
        self.calls: List[Dict] = []
        self._default_chunks = default_chunks or []

    async def retrieve_chunks(self, query, repository_id, top_k=None, file_prefix=None, db=None):
        self.calls.append({"query": query, "repository_id": repository_id, "file_prefix": file_prefix})
        return self._default_chunks


class _FakeResult:
    """Stands in for a SQLAlchemy Result: supports the two accessors the
    endpoints under test actually call (``scalar_one_or_none`` for the
    Repository lookup in qa.py, ``all`` for CodeRetriever's raw pgvector
    query in the untouched rag.py)."""

    def __init__(self, scalar=None, rows=None):
        self._scalar = scalar
        self._rows = rows if rows is not None else []

    def scalar_one_or_none(self):
        return self._scalar

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, repo: Optional[Repository] = None):
        self._repo = repo

    async def execute(self, stmt):
        return _FakeResult(scalar=self._repo, rows=[])


def _write(root: Path, rel_path: str, content: str) -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _override_db(repo: Optional[Repository]):
    async def _get_db():
        yield _FakeDB(repo)

    return _get_db


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    """Every test sets its own overrides; never leak them into other test
    modules sharing the same FastAPI ``app`` singleton."""
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 200: structured QAAnswer for a real (fixture) evidence-backed question
# ---------------------------------------------------------------------------
def test_ask_returns_structured_qaanswer(client, tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path, "src/cart.py", "def subtotal(items):\n    return sum(items)\n")

    repo = Repository(id=1, name="demo", local_path=str(tmp_path), remote_url=None, default_branch="main")
    fake_retriever = _FakeRetriever(
        default_chunks=[
            RetrievedChunk(
                file_path="src/cart.py", symbol_name="subtotal", symbol_type="function",
                start_line=1, end_line=2, source_code="def subtotal(items):\n    return sum(items)\n",
                similarity_score=0.9,
            )
        ]
    )

    app.dependency_overrides[get_db] = _override_db(repo)
    app.dependency_overrides[qa_module.get_code_retriever] = lambda: fake_retriever

    resp = client.post(
        "/api/v1/qa/ask",
        json={"question": "Where is the cart subtotal calculated?", "repository_id": 1},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {
        "summary", "details", "flow_trace", "evidence",
        "corrected_premise", "confidence", "projects_considered",
    }
    assert data["summary"]
    assert data["confidence"] in ("direct_evidence", "inferred", "no_evidence")
    assert data["evidence"], "evidence-backed question must return at least one citation"
    assert data["evidence"][0]["file_path"] == "src/cart.py"
    assert data["projects_considered"] == ["."]


# ---------------------------------------------------------------------------
# 404: unknown repository_id
# ---------------------------------------------------------------------------
def test_ask_unknown_repository_returns_404(client):
    app.dependency_overrides[get_db] = _override_db(None)
    app.dependency_overrides[qa_module.get_code_retriever] = lambda: _FakeRetriever()

    resp = client.post(
        "/api/v1/qa/ask",
        json={"question": "Where is the cart subtotal calculated?", "repository_id": 999},
    )

    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 422: blank / whitespace-only question
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_ask_blank_question_returns_422(client, tmp_path, question):
    repo = Repository(id=1, name="demo", local_path=str(tmp_path), remote_url=None, default_branch="main")
    app.dependency_overrides[get_db] = _override_db(repo)
    app.dependency_overrides[qa_module.get_code_retriever] = lambda: _FakeRetriever()

    resp = client.post("/api/v1/qa/ask", json={"question": question, "repository_id": 1})

    assert resp.status_code == 422


def test_ask_missing_question_field_returns_422(client):
    """Pydantic-level required-field validation (no repository lookup needed)."""
    app.dependency_overrides[get_db] = _override_db(None)
    app.dependency_overrides[qa_module.get_code_retriever] = lambda: _FakeRetriever()

    resp = client.post("/api/v1/qa/ask", json={"repository_id": 1})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# no_evidence: deterministic answer, no evidence gathered
# ---------------------------------------------------------------------------
def test_ask_no_evidence_response(client, tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")

    repo = Repository(id=1, name="demo", local_path=str(tmp_path), remote_url=None, default_branch="main")
    app.dependency_overrides[get_db] = _override_db(repo)
    app.dependency_overrides[qa_module.get_code_retriever] = lambda: _FakeRetriever(default_chunks=[])

    resp = client.post(
        "/api/v1/qa/ask",
        json={"question": "Where is the nonexistent payment gateway integrated?", "repository_id": 1},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence"] == "no_evidence"
    assert data["summary"] == NO_EVIDENCE_SUMMARY
    assert data["evidence"] == []


# ---------------------------------------------------------------------------
# Request schema never exposes a filesystem-path field
# ---------------------------------------------------------------------------
def test_request_schema_has_no_filesystem_path_fields():
    schema = qa_module.QAAskRequest.model_json_schema()
    field_names = set(schema.get("properties", {}).keys())

    assert field_names == {"question", "repository_id"}
    for forbidden in ("workspace_dir", "local_path", "path", "file_path", "directory"):
        assert forbidden not in field_names


def test_openapi_qa_ask_request_body_has_no_filesystem_path_fields():
    """Belt-and-suspenders: check the schema FastAPI actually publishes."""
    openapi = app.openapi()
    request_schema = openapi["paths"]["/api/v1/qa/ask"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    ref = request_schema["$ref"].split("/")[-1]
    properties = openapi["components"]["schemas"][ref]["properties"]

    assert set(properties.keys()) == {"question", "repository_id"}


# ---------------------------------------------------------------------------
# repository_id is resolved server-side; a client-supplied path-shaped field
# is never used, even if present in the raw request body.
# ---------------------------------------------------------------------------
def test_workspace_dir_is_always_server_resolved(client, tmp_path, monkeypatch):
    real_local_path = str(tmp_path)
    repo = Repository(id=1, name="demo", local_path=real_local_path, remote_url=None, default_branch="main")
    app.dependency_overrides[get_db] = _override_db(repo)
    app.dependency_overrides[qa_module.get_code_retriever] = lambda: _FakeRetriever()

    captured = {}

    async def fake_ask_codebase(question, workspace_dir, repository_id, db=None, retriever=None):
        captured["workspace_dir"] = workspace_dir
        captured["repository_id"] = repository_id
        from app.services.qa.models import QAAnswer
        return QAAnswer(summary="stub", confidence="inferred", evidence=[], projects_considered=[])

    monkeypatch.setattr(qa_module.qa_service, "ask_codebase", fake_ask_codebase)

    # Extra, browser-supplied path-shaped fields must be silently ignored --
    # QAAskRequest doesn't declare them, so Pydantic drops them.
    resp = client.post(
        "/api/v1/qa/ask",
        json={
            "question": "Where is the cart subtotal calculated?",
            "repository_id": 1,
            "workspace_dir": "C:/should/never/be/used",
            "local_path": "/etc/passwd",
        },
    )

    assert resp.status_code == 200
    assert captured["workspace_dir"] == real_local_path
    assert captured["repository_id"] == 1


# ---------------------------------------------------------------------------
# Existing /api/v1/rag/ask remains unaffected by this addition
# ---------------------------------------------------------------------------
def test_existing_rag_ask_endpoint_unaffected(client):
    app.dependency_overrides[get_db] = _override_db(None)

    resp = client.post(
        "/api/v1/rag/ask",
        json={"query": "Where is the cart subtotal calculated?", "repository_id": 1, "top_k": 5},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"answer", "retrieved_chunks", "citations"}
    assert data["retrieved_chunks"] == []
    assert data["citations"] == []
