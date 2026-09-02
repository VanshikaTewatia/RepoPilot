"""Unit tests for the Deep Codebase Q&A investigation layer
(app.services.qa.investigator).

No real Docker, database, or Gemini calls are made: RAG retrieval is
exercised through a small fake CodeRetriever (duck-typed the same way the
real one is called) so these tests stay fast and deterministic while still
exercising real project detection and real filesystem search/read against
temporary fixture repositories.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from app.services.qa.investigator import (
    MAX_PROJECTS_INVESTIGATED,
    _DEPTH_CONFIGS,
    _MAX_TOTAL_SYMBOL_MATCHES,
    investigate,
)
from app.services.rag.retriever import RetrievedChunk


def _write(root: Path, rel_path: str, content: str = "") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _chunk(file_path: str, symbol_name: Optional[str] = None, symbol_type: str = "function") -> RetrievedChunk:
    return RetrievedChunk(
        file_path=file_path,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        start_line=1,
        end_line=3,
        source_code="stub",
        similarity_score=0.9,
    )


class _FakeRetriever:
    """Stands in for CodeRetriever: records every call's kwargs and returns
    pre-configured chunks keyed by the file_prefix it was scoped to."""

    def __init__(
        self,
        chunks_by_prefix: Optional[Dict[Optional[str], List[RetrievedChunk]]] = None,
        default_chunks: Optional[List[RetrievedChunk]] = None,
    ):
        self.calls: List[Dict] = []
        self._chunks_by_prefix = chunks_by_prefix or {}
        self._default_chunks = default_chunks or []

    async def retrieve_chunks(self, query, repository_id, top_k=None, file_prefix=None, db=None):
        self.calls.append(
            {"query": query, "repository_id": repository_id, "top_k": top_k, "file_prefix": file_prefix}
        )
        if file_prefix in self._chunks_by_prefix:
            return self._chunks_by_prefix[file_prefix]
        return self._default_chunks


# ---------------------------------------------------------------------------
# Single-project investigation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_project_investigation():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        _write(root, "src/cart.py", "def subtotal(items):\n    return sum(items)\n")

        fake = _FakeRetriever(default_chunks=[_chunk("src/cart.py", "subtotal")])
        result = await investigate(
            str(root), repository_id=1, question="Where is the cart subtotal calculated?",
            depth="shallow", db=None, retriever=fake,
        )

        assert result.investigated_projects == ["."]
        assert len(result.evidence) == 1
        assert result.evidence[0].ecosystem == "python"
        assert result.evidence[0].project_root == "."
        assert result.has_evidence is True


# ---------------------------------------------------------------------------
# Multi-project investigation: prefer the relevant project's evidence
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multi_project_investigation_selects_relevant_project_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "backend/pom.xml", "<project></project>")
        _write(root, "frontend/package.json", json.dumps({"dependencies": {"react": "^18.0.0"}}))
        _write(root, "frontend/src/ProductCard.jsx", "export function ProductCard() { return PRICE; }\n")

        fake = _FakeRetriever(chunks_by_prefix={
            "frontend": [_chunk("frontend/src/ProductCard.jsx", "ProductCard")],
        })

        result = await investigate(
            str(root), repository_id=1, question="Explain the React ProductCard component",
            depth="shallow", db=None, retriever=fake,
        )

        assert result.investigated_projects == ["frontend"]
        assert len(result.evidence) == 1
        assert result.evidence[0].ecosystem == "node"
        assert "React" in result.evidence[0].frameworks
        # RAG was scoped to the frontend project, not the whole repository
        assert fake.calls[0]["file_prefix"] == "frontend"


@pytest.mark.asyncio
async def test_project_selection_prefers_real_evidence_over_question_wording():
    """Mirrors select_relevant_projects' own philosophy (already relied on
    by the verification engine): a keyword match physically found in a
    project's files outweighs the question merely naming another project's
    language."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "backend/pom.xml", "<project></project>")
        _write(root, "frontend/package.json", json.dumps({"name": "frontend"}))
        _write(root, "frontend/src/promo.js", "const PromoCode = 'discount logic';\n")

        fake = _FakeRetriever(chunks_by_prefix={
            "frontend": [_chunk("frontend/src/promo.js", None)],
        })

        # Mentions "Java" (backend's language) but the real evidence for
        # "PromoCode" physically lives in the frontend project.
        result = await investigate(
            str(root), repository_id=1, question="Where is PromoCode implemented in this Java project?",
            depth="shallow", db=None, retriever=fake,
        )

        assert result.investigated_projects == ["frontend"]


@pytest.mark.asyncio
async def test_bounded_investigation_caps_ambiguous_project_count():
    """An ambiguous question across many projects investigates at most
    MAX_PROJECTS_INVESTIGATED -- never the whole repository unbounded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for i in range(5):
            _write(root, f"svc{i}/go.mod", f"module example.com/svc{i}\n")

        fake = _FakeRetriever(default_chunks=[])
        result = await investigate(
            str(root), repository_id=1, question="How does this work?",
            depth="shallow", db=None, retriever=fake,
        )

        assert len(result.investigated_projects) <= MAX_PROJECTS_INVESTIGATED
        assert len(result.detected_projects) == 5  # all detected, not all investigated


# ---------------------------------------------------------------------------
# Depth-aware behavior
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shallow_investigation_is_rag_only():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        _write(root, "src/cart.py", "def subtotal(items):\n    return sum(items)\n")

        fake = _FakeRetriever(default_chunks=[_chunk("src/cart.py", "subtotal")])
        result = await investigate(
            str(root), repository_id=1, question="Where is the cart subtotal calculated?",
            depth="shallow", db=None, retriever=fake,
        )

        ev = result.evidence[0]
        assert len(ev.chunks) == 1
        assert ev.files_inspected == []
        assert ev.symbol_matches == []
        assert fake.calls[0]["top_k"] == _DEPTH_CONFIGS["shallow"].top_k  # None


@pytest.mark.asyncio
async def test_targeted_investigation_inspects_top_files_and_searches():
    """Also exercises the Phase 0 search_code fix end-to-end: the project is
    plain JS, and a targeted search must still find a match in a .js file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"name": "x"}))
        _write(root, "src/auth.js", "function login(user) {\n  return AUTH_TOKEN;\n}\n")

        fake = _FakeRetriever(default_chunks=[_chunk("src/auth.js", "login")])
        result = await investigate(
            str(root), repository_id=1, question="How does login work?",
            depth="targeted", db=None, retriever=fake,
        )

        ev = result.evidence[0]
        assert fake.calls[0]["top_k"] == _DEPTH_CONFIGS["targeted"].top_k
        assert len(ev.files_inspected) == 1
        assert ev.files_inspected[0].file_path == "src/auth.js"
        assert any(m.file == "src/auth.js" for m in ev.symbol_matches)


@pytest.mark.asyncio
async def test_medium_investigation_traces_symbol_derived_search_terms():
    """Medium depth must search for symbol names discovered in the RAG
    hits themselves (one-hop tracing), not just words from the question --
    proving it finds a cross-file reference the question never mentioned."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        _write(root, "src/payment.py", "class PaymentProcessor:\n    def charge(self):\n        pass\n")
        _write(
            root, "src/order_flow.py",
            "from src.payment import PaymentProcessor\n\ndef process_order():\n    PaymentProcessor().charge()\n",
        )

        fake = _FakeRetriever(default_chunks=[_chunk("src/payment.py", "PaymentProcessor", "class")])
        result = await investigate(
            str(root), repository_id=1, question="How does checkout work?",
            depth="medium", db=None, retriever=fake,
        )

        ev = result.evidence[0]
        assert fake.calls[0]["top_k"] == _DEPTH_CONFIGS["medium"].top_k
        # "PaymentProcessor" never appears in the question -- only reachable
        # via the symbol name discovered in the RAG hit.
        assert any(m.file == "src/order_flow.py" for m in ev.symbol_matches)


@pytest.mark.asyncio
async def test_deep_investigation_reads_full_files_uncapped():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        long_file = "\n".join(f"line {i}" for i in range(1, 501))  # 500 lines, over the 400 medium/targeted cap
        _write(root, "src/big.py", long_file + "\n")

        fake = _FakeRetriever(default_chunks=[_chunk("src/big.py", "module")])
        result = await investigate(
            str(root), repository_id=1, question="Explain the architecture",
            depth="deep", db=None, retriever=fake,
        )

        ev = result.evidence[0]
        assert fake.calls[0]["top_k"] == _DEPTH_CONFIGS["deep"].top_k
        assert ev.files_inspected[0].total_lines == 500
        assert ev.files_inspected[0].truncated is False
        assert "line 500" in ev.files_inspected[0].content


@pytest.mark.asyncio
async def test_bounded_investigation_caps_files_inspected_and_symbol_matches():
    """Deep depth still stops at its configured caps even when far more
    candidate files/matches are available -- never an unbounded scan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        chunks = []
        for i in range(15):
            path = f"src/mod_{i}.py"
            _write(root, path, "TARGET_TOKEN = 1\n")
            chunks.append(_chunk(path, f"symbol_{i}"))

        fake = _FakeRetriever(default_chunks=chunks)
        result = await investigate(
            str(root), repository_id=1, question="Explain TARGET_TOKEN usage across the project",
            depth="deep", db=None, retriever=fake,
        )

        ev = result.evidence[0]
        cfg = _DEPTH_CONFIGS["deep"]
        assert len(ev.files_inspected) == cfg.max_files_inspected
        assert len(ev.symbol_matches) <= _MAX_TOTAL_SYMBOL_MATCHES


# ---------------------------------------------------------------------------
# No-evidence handling: investigate() never calls an LLM and must say so
# plainly rather than let a later step invent an answer.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_evidence_when_unsupported_ecosystem():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "README.md", "just docs, no recognizable project here\n")

        result = await investigate(
            str(root), repository_id=1, question="How does authentication work?",
            depth="deep", db=None, retriever=_FakeRetriever(),
        )

        assert result.has_evidence is False
        assert result.no_evidence_reason is not None
        assert result.evidence == []


@pytest.mark.asyncio
async def test_no_evidence_when_nothing_relevant_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        _write(root, "src/unrelated.py", "def totally_unrelated():\n    pass\n")

        # No RAG hits, and the question's own terms don't appear anywhere.
        fake = _FakeRetriever(default_chunks=[])
        result = await investigate(
            str(root), repository_id=1, question="How does the nonexistent payment gateway work?",
            depth="deep", db=None, retriever=fake,
        )

        assert result.has_evidence is False
        assert result.no_evidence_reason is not None
        assert result.evidence[0].has_evidence is False


def test_invalid_depth_raises():
    import asyncio

    async def _run():
        await investigate("/tmp/does-not-matter", repository_id=1, question="q", depth="ultra-deep")

    with pytest.raises(ValueError, match="Unknown investigation depth"):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# The user's terminology is a hypothesis, not ground truth
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_evidence_exposes_real_framework_despite_user_terminology():
    """The user asks about 'the React component' but the project is
    actually Vue -- Evidence must report the truth (Vue), never silently
    substitute or agree with the user's incorrect premise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"dependencies": {"vue": "^3.4.0"}}))
        _write(root, "src/AuthView.vue", "<template><div>login</div></template>\n")

        fake = _FakeRetriever(default_chunks=[_chunk("src/AuthView.vue", None, "block")])
        result = await investigate(
            str(root), repository_id=1,
            question="Explain the React component that handles authentication.",
            depth="shallow", db=None, retriever=fake,
        )

        ev = result.evidence[0]
        assert ev.ecosystem == "node"
        assert ev.frameworks == ["Vue"]
        assert "React" not in ev.frameworks
        assert ev.chunks[0].file_path == "src/AuthView.vue"
