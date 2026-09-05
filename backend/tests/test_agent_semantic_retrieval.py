"""Tests for Phase 6B: semantic candidate-file retrieval integrated into
app.services.agent.graph.retrieve_node.

No real Gemini/DB/network calls anywhere -- CodeRetriever and
_build_repository_evidence are patched/faked directly. This suite is about
the INTEGRATION contract (bounded, project-scoped semantic retrieval merged
ahead of the existing keyword ranking, degrading gracefully on any
failure) -- CodeRetriever/RetrievalError internals already have their own
dedicated suite (test_retriever.py), and project detection/selection has
its own (test_qa_investigator.py's project-selection tests, and
test_agent_baseline_integration.py's RepositoryAnalyzer-backed tests) --
neither is duplicated or weakened here.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.services.agent.graph import (
    RETRIEVAL_LIMIT,
    SEMANTIC_MAX_PROJECTS,
    SEMANTIC_TOP_K,
    _gather_semantic_candidate_files,
    _merge_candidate_files,
    _projects_for_semantic_retrieval,
    _semantic_candidate_files,
    _semantic_query_text,
    retrieve_node,
)
from app.services.baseline import RepositoryEvidence
from app.services.rag.retriever import RetrievalError, RetrievedChunk
from app.services.verification.project_analyzer import ProjectInfo


def _chunk(file_path: str, symbol_name: str = "x") -> RetrievedChunk:
    return RetrievedChunk(
        file_path=file_path,
        symbol_name=symbol_name,
        symbol_type="function",
        start_line=1,
        end_line=5,
        source_code="stub",
        similarity_score=0.9,
    )


def _project(root: str, ecosystem: str = "python") -> ProjectInfo:
    return ProjectInfo(root=root, ecosystem=ecosystem, languages=["Python"])


class _FakeRetriever:
    """Mirrors test_qa_investigator.py's _FakeRetriever convention: a
    minimal CodeRetriever stand-in matching retrieve_chunks's signature,
    recording every call for assertions."""

    def __init__(self, chunks_by_prefix=None, raise_retrieval_error=False, raise_generic_error=False):
        self.chunks_by_prefix = chunks_by_prefix or {}
        self.raise_retrieval_error = raise_retrieval_error
        self.raise_generic_error = raise_generic_error
        self.calls = []

    async def retrieve_chunks(self, query, repository_id, top_k=None, file_prefix=None, db=None):
        self.calls.append(
            {"query": query, "repository_id": repository_id, "top_k": top_k, "file_prefix": file_prefix, "db": db}
        )
        if self.raise_retrieval_error:
            raise RetrievalError("embedding generation failed")
        if self.raise_generic_error:
            raise RuntimeError("unexpected boom")
        return self.chunks_by_prefix.get(file_prefix, [])


def _make_workspace(tmpdir: str, files: dict) -> Path:
    workspace = Path(tmpdir)
    for name, content in files.items():
        (workspace / name).write_text(content, encoding="utf-8")
    return workspace


# ===========================================================================
# 1. Pure helpers: merge / dedup / cap
# ===========================================================================
def test_merge_candidate_files_ranks_semantic_first():
    merged = _merge_candidate_files(["b.py", "a.py"], ["c.py", "a.py"], limit=10)
    assert merged == ["b.py", "a.py", "c.py"]


def test_merge_candidate_files_deduplicates_first_occurrence_wins():
    merged = _merge_candidate_files(["a.py"], ["a.py", "b.py"], limit=10)
    assert merged == ["a.py", "b.py"]


def test_merge_candidate_files_caps_at_limit():
    merged = _merge_candidate_files(["a.py", "b.py"], ["c.py", "d.py", "e.py"], limit=3)
    assert merged == ["a.py", "b.py", "c.py"]


def test_merge_candidate_files_empty_semantic_is_byte_identical_to_keyword_only():
    keyword_ranked = ["z.py", "a.py", "m.py", "q.py", "r.py", "s.py"]
    assert _merge_candidate_files([], keyword_ranked, limit=RETRIEVAL_LIMIT) == keyword_ranked[:RETRIEVAL_LIMIT]


def test_semantic_candidate_files_deduplicates_preserving_order():
    chunks = [_chunk("a.py"), _chunk("b.py"), _chunk("a.py"), _chunk("c.py")]
    assert _semantic_candidate_files(chunks) == ["a.py", "b.py", "c.py"]


def test_semantic_candidate_files_empty_for_no_chunks():
    assert _semantic_candidate_files([]) == []


def test_semantic_query_text_includes_error_analysis_when_present():
    assert _semantic_query_text("Fix subtotal", None) == "Fix subtotal"
    query = _semantic_query_text("Fix subtotal", "AssertionError: got -1")
    assert "Fix subtotal" in query
    assert "AssertionError: got -1" in query


def test_semantic_query_text_changes_between_retries():
    first = _semantic_query_text("Fix subtotal", None)
    second = _semantic_query_text("Fix subtotal", "AssertionError: got -1")
    assert first != second


# ===========================================================================
# 2. Project scoping: bounded to SEMANTIC_MAX_PROJECTS
# ===========================================================================
def test_projects_for_semantic_retrieval_caps_at_max_projects():
    evidence = RepositoryEvidence(detected_projects=[_project("a"), _project("b"), _project("c"), _project("d")])
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        projects = _projects_for_semantic_retrieval({})
    assert len(projects) == SEMANTIC_MAX_PROJECTS
    assert [p.root for p in projects] == ["a", "b"]


def test_projects_for_semantic_retrieval_returns_all_when_below_cap():
    evidence = RepositoryEvidence(detected_projects=[_project(".")])
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        projects = _projects_for_semantic_retrieval({})
    assert [p.root for p in projects] == ["."]


# ===========================================================================
# 3. _gather_semantic_candidate_files: bounds, scoping, failure handling
# ===========================================================================
@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_uses_bounded_top_k_and_file_prefix():
    evidence = RepositoryEvidence(detected_projects=[_project(".")])
    fake = _FakeRetriever(chunks_by_prefix={None: [_chunk("a.py")]})
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        files = await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db="fake-db", retriever=fake)

    assert files == ["a.py"]
    assert len(fake.calls) == 1
    assert fake.calls[0]["top_k"] == SEMANTIC_TOP_K
    assert fake.calls[0]["file_prefix"] is None
    assert fake.calls[0]["repository_id"] == 1
    assert fake.calls[0]["db"] == "fake-db"


@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_never_queries_more_than_max_projects():
    evidence = RepositoryEvidence(
        detected_projects=[_project("a"), _project("b"), _project("c"), _project("d")]
    )
    fake = _FakeRetriever()
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db=None, retriever=fake)

    assert len(fake.calls) == SEMANTIC_MAX_PROJECTS
    assert {c["file_prefix"] for c in fake.calls} == {"a", "b"}


@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_scopes_query_per_project():
    evidence = RepositoryEvidence(detected_projects=[_project("frontend"), _project("backend")])
    fake = _FakeRetriever(chunks_by_prefix={"frontend": [_chunk("frontend/App.jsx")], "backend": [_chunk("backend/app.py")]})
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        files = await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db=None, retriever=fake)

    assert set(files) == {"frontend/App.jsx", "backend/app.py"}


@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_retrieval_error_degrades_to_empty():
    evidence = RepositoryEvidence(detected_projects=[_project(".")])
    fake = _FakeRetriever(raise_retrieval_error=True)
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        files = await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db=None, retriever=fake)
    assert files == []


@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_generic_exception_degrades_to_empty():
    evidence = RepositoryEvidence(detected_projects=[_project(".")])
    fake = _FakeRetriever(raise_generic_error=True)
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        files = await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db=None, retriever=fake)
    assert files == []


@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_empty_index_degrades_to_empty():
    evidence = RepositoryEvidence(detected_projects=[_project(".")])
    fake = _FakeRetriever(chunks_by_prefix={})  # real query, zero rows
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        files = await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db=None, retriever=fake)
    assert files == []


@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_project_scoping_failure_degrades_to_empty():
    with patch("app.services.agent.graph._build_repository_evidence", side_effect=RuntimeError("scan failed")):
        files = await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db=None, retriever=_FakeRetriever())
    assert files == []


@pytest.mark.asyncio
async def test_gather_semantic_candidate_files_no_detected_projects_makes_no_calls():
    evidence = RepositoryEvidence(detected_projects=[])
    fake = _FakeRetriever()
    with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence):
        files = await _gather_semantic_candidate_files({"task_description": "fix bug"}, repository_id=1, db=None, retriever=fake)
    assert files == []
    assert fake.calls == []


# ===========================================================================
# 4. retrieve_node full integration: semantic-first ordering, dedup, fallback
# ===========================================================================
@pytest.mark.asyncio
async def test_retrieve_node_ranks_semantic_hit_ahead_of_keyword_match():
    files = {
        "semantic_hit.py": "content one\n",
        "keyword_match.py": "cart discount applied\n",
        "irrelevant.py": "filler\n",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {
            "workspace_dir": str(workspace),
            "task_description": "Apply cart discount",
            "repository_id": 1,
            "keyword_matches": [{"file": "keyword_match.py", "line": 1, "content": "cart discount"}],
        }
        evidence = RepositoryEvidence(detected_projects=[_project(".")])
        fake_retriever = _FakeRetriever(chunks_by_prefix={None: [_chunk("semantic_hit.py")]})

        with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence), patch(
            "app.services.agent.graph.CodeRetriever", return_value=fake_retriever
        ), patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value="fake-session")
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            out = await retrieve_node(state)

    retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]
    assert retrieved_paths[0] == "semantic_hit.py"
    assert "keyword_match.py" in retrieved_paths


@pytest.mark.asyncio
async def test_retrieve_node_deduplicates_file_present_in_both_semantic_and_keyword_hits():
    files = {"shared.py": "content\n", "other.py": "cart discount\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {
            "workspace_dir": str(workspace),
            "task_description": "Apply cart discount",
            "repository_id": 1,
            "keyword_matches": [{"file": "shared.py", "line": 1, "content": "x"}],
        }
        evidence = RepositoryEvidence(detected_projects=[_project(".")])
        fake_retriever = _FakeRetriever(chunks_by_prefix={None: [_chunk("shared.py")]})

        with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence), patch(
            "app.services.agent.graph.CodeRetriever", return_value=fake_retriever
        ), patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value="fake-session")
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            out = await retrieve_node(state)

    retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]
    assert retrieved_paths.count("shared.py") == 1


@pytest.mark.asyncio
async def test_retrieve_node_missing_repository_id_never_opens_session_or_retriever():
    files = {"a.py": "content\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {"workspace_dir": str(workspace), "task_description": "fix bug"}

        with patch("app.services.agent.graph.open_session") as mock_open_session, patch(
            "app.services.agent.graph.CodeRetriever"
        ) as mock_retriever_cls:
            out = await retrieve_node(state)

    mock_open_session.assert_not_called()
    mock_retriever_cls.assert_not_called()
    assert [item["file_path"] for item in out["retrieved_context"]] == ["a.py"]


@pytest.mark.asyncio
async def test_retrieve_node_retrieval_error_falls_back_to_keyword_only_ranking():
    files = {"a_alpha.py": "filler\n", "zzz_vip.py": "cart discount applied\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {
            "workspace_dir": str(workspace),
            "task_description": "Apply cart discount",
            "repository_id": 1,
            "keyword_matches": [{"file": "zzz_vip.py", "line": 1, "content": "cart discount"}],
        }
        evidence = RepositoryEvidence(detected_projects=[_project(".")])
        fake_retriever = _FakeRetriever(raise_retrieval_error=True)

        with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence), patch(
            "app.services.agent.graph.CodeRetriever", return_value=fake_retriever
        ), patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value="fake-session")
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            out = await retrieve_node(state)

    retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]
    assert retrieved_paths[0] == "zzz_vip.py"


@pytest.mark.asyncio
async def test_retrieve_node_empty_index_falls_back_to_keyword_only_ranking():
    files = {"a_alpha.py": "filler\n", "zzz_vip.py": "cart discount applied\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {
            "workspace_dir": str(workspace),
            "task_description": "Apply cart discount",
            "repository_id": 1,
            "keyword_matches": [{"file": "zzz_vip.py", "line": 1, "content": "cart discount"}],
        }
        evidence = RepositoryEvidence(detected_projects=[_project(".")])
        fake_retriever = _FakeRetriever(chunks_by_prefix={})

        with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence), patch(
            "app.services.agent.graph.CodeRetriever", return_value=fake_retriever
        ), patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value="fake-session")
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            out = await retrieve_node(state)

    retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]
    assert retrieved_paths == ["zzz_vip.py", "a_alpha.py"]


@pytest.mark.asyncio
async def test_retrieve_node_session_unavailable_falls_back_to_keyword_only_ranking():
    """open_session() legitimately yields None (e.g. DB down) -- retrieve_node
    must degrade to keyword-only ranking, never raise."""
    files = {"a_alpha.py": "filler\n", "zzz_vip.py": "cart discount applied\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {
            "workspace_dir": str(workspace),
            "task_description": "Apply cart discount",
            "repository_id": 1,
            "keyword_matches": [{"file": "zzz_vip.py", "line": 1, "content": "cart discount"}],
        }

        with patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value=None)
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            out = await retrieve_node(state)

    retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]
    assert retrieved_paths == ["zzz_vip.py", "a_alpha.py"]


@pytest.mark.asyncio
async def test_retrieve_node_stale_semantic_hit_for_nonexistent_file_is_dropped():
    """A chunk-index entry pointing at a file that no longer exists on disk
    must never occupy a candidate slot."""
    files = {"a_alpha.py": "filler\n", "zzz_vip.py": "cart discount applied\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {
            "workspace_dir": str(workspace),
            "task_description": "Apply cart discount",
            "repository_id": 1,
            "keyword_matches": [{"file": "zzz_vip.py", "line": 1, "content": "cart discount"}],
        }
        evidence = RepositoryEvidence(detected_projects=[_project(".")])
        fake_retriever = _FakeRetriever(chunks_by_prefix={None: [_chunk("deleted_file.py")]})

        with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence), patch(
            "app.services.agent.graph.CodeRetriever", return_value=fake_retriever
        ), patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value="fake-session")
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            out = await retrieve_node(state)

    retrieved_paths = [item["file_path"] for item in out["retrieved_context"]]
    assert "deleted_file.py" not in retrieved_paths
    assert retrieved_paths == ["zzz_vip.py", "a_alpha.py"]


@pytest.mark.asyncio
async def test_retrieve_node_error_analysis_changes_semantic_query_on_retry():
    files = {"a.py": "content\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        evidence = RepositoryEvidence(detected_projects=[_project(".")])
        fake_retriever = _FakeRetriever(chunks_by_prefix={None: [_chunk("a.py")]})

        with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence), patch(
            "app.services.agent.graph.CodeRetriever", return_value=fake_retriever
        ), patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value="fake-session")
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            first_state = {"workspace_dir": str(workspace), "task_description": "Fix subtotal", "repository_id": 1, "error_analysis": None}
            await retrieve_node(first_state)

            second_state = {
                "workspace_dir": str(workspace),
                "task_description": "Fix subtotal",
                "repository_id": 1,
                "error_analysis": "AssertionError: subtotal() returned -1",
            }
            await retrieve_node(second_state)

    assert len(fake_retriever.calls) == 2
    assert fake_retriever.calls[0]["query"] == "Fix subtotal"
    assert "AssertionError: subtotal() returned -1" in fake_retriever.calls[1]["query"]
    assert fake_retriever.calls[0]["query"] != fake_retriever.calls[1]["query"]


# ===========================================================================
# 5. Phase 6A compatibility: retrieved_context shape unaffected, diagnosis
# still consumes it with zero changes.
# ===========================================================================
@pytest.mark.asyncio
async def test_retrieved_context_shape_unchanged_when_semantic_retrieval_contributes():
    files = {"a.py": "content\n"}
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = _make_workspace(tmpdir, files)
        state = {"workspace_dir": str(workspace), "task_description": "fix bug", "repository_id": 1}
        evidence = RepositoryEvidence(detected_projects=[_project(".")])
        fake_retriever = _FakeRetriever(chunks_by_prefix={None: [_chunk("a.py")]})

        with patch("app.services.agent.graph._build_repository_evidence", return_value=evidence), patch(
            "app.services.agent.graph.CodeRetriever", return_value=fake_retriever
        ), patch("app.services.agent.graph.open_session") as mock_open_session:
            mock_open_session.return_value.__aenter__ = AsyncMock(return_value="fake-session")
            mock_open_session.return_value.__aexit__ = AsyncMock(return_value=False)

            out = await retrieve_node(state)

    assert out["retrieved_context"] == [{"file_path": "a.py", "content": "content\n", "total_lines": 1}]


@pytest.mark.asyncio
async def test_diagnose_node_consumes_semantically_populated_retrieved_context_unchanged():
    """diagnose_node/diagnoser.py need zero changes: they only ever read
    retrieved_context's existing shape, regardless of how its files were
    selected."""
    from app.services.agent.graph import diagnose_node

    state = {
        "task_description": "fix bug",
        "retrieved_context": [{"file_path": "a.py", "content": "def f(): return 1\n", "total_lines": 1}],
        "error_analysis": None,
    }
    out = await diagnose_node(state)
    assert out["diagnosis_status"] in ("DIAGNOSED", "INSUFFICIENT_EVIDENCE", "DIAGNOSIS_FAILED")
