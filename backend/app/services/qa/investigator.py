"""Depth-aware repository investigation for Deep Codebase Q&A.

Builds exclusively on already-existing, already-tested infrastructure:
project/ecosystem detection and multi-project relevance selection from the
verification engine (``RepositoryAnalyzer``, ``select_relevant_projects``),
semantic retrieval from the RAG engine (``CodeRetriever``), and the agent's
sandboxed filesystem tools (``list_files``/``read_file``/``search_code``).
No project-detection, embedding, or filesystem-safety logic is duplicated
here.

This module never calls an LLM and never invents facts. It only gathers
``Evidence`` from the real repository; question classification and answer
synthesis are separate, later stages that decide what to do with the
result -- including refusing to answer when
``InvestigationResult.has_evidence`` is False.

The user's own terminology is never trusted as ground truth: every
``Evidence`` carries the *actual* detected ecosystem/languages/frameworks
from real manifest evidence (via ``ProjectInfo``), regardless of what the
question assumed, and every retrieval/search is scoped to the *actual*
relevant project(s) rather than influenced by the question's wording.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.services.agent import tools
from app.services.rag.retriever import CodeRetriever, RetrievalError, RetrievedChunk
from app.services.verification.project_analyzer import (
    ProjectInfo,
    RepositoryAnalyzer,
    select_relevant_projects,
)

# ---------------------------------------------------------------------------
# Bounds. Investigation depth is question-aware, but every depth -- including
# "deep" -- is still a strictly bounded amount of work, never unrestricted
# autonomous exploration: a fixed cap on projects investigated, files read,
# and search terms issued, all independent of repository size.
# ---------------------------------------------------------------------------

VALID_DEPTHS = ("shallow", "targeted", "medium", "deep")

# Hard cap on how many detected projects a single question ever investigates,
# even when project selection is completely ambiguous (every project ties).
# Prevents a large monorepo from turning one question into an all-projects
# deep scan.
MAX_PROJECTS_INVESTIGATED = 3

# A bounded, single-pass, whole-repository grep used ONLY to gather real
# evidence for *which project* is relevant when the repository is a
# monorepo -- mirrors app.services.agent.graph.investigate_node's existing
# keyword-scan-for-ranking pattern, applied here for the identical purpose.
# Never run when the repository resolves to a single project (nothing to
# disambiguate), so this never touches "the entire repository" in the
# common case.
_PROJECT_SELECTION_SCAN_TERMS = 2
_PROJECT_SELECTION_SCAN_MATCH_CAP = 20

# Per-project bounds once investigation actually runs search_code() for a
# selected project.
_MAX_MATCHES_PER_TERM = 5
_MAX_TOTAL_SYMBOL_MATCHES = 25


@dataclass(frozen=True)
class _DepthConfig:
    """Per-depth investigation bounds.

    Maps directly to the four required tiers:
      shallow  -> RAG retrieval only, project-scoped.
      targeted -> + inspect the top few RAG-hit files, + a couple of
                   targeted searches derived from the question.
      medium   -> + more files, + search terms also derived from the RAG
                   hits' own symbol names (one-hop "who references this").
      deep     -> + full (uncapped) file reads and the widest search
                   budget, for bounded cross-file tracing of a flow/
                   architecture question.
    """

    top_k: Optional[int]  # RAG retrieval limit; None -> retriever's own default (settings.vector_top_k)
    max_files_inspected: int  # tools.read_file() calls
    max_question_terms: int  # search terms derived from the question text
    max_symbol_terms: int  # extra search terms derived from RAG hits' own symbol names
    file_read_line_cap: Optional[int]  # None = read the whole file


_DEPTH_CONFIGS: Dict[str, _DepthConfig] = {
    "shallow": _DepthConfig(
        top_k=None, max_files_inspected=0, max_question_terms=0, max_symbol_terms=0, file_read_line_cap=None
    ),
    "targeted": _DepthConfig(
        top_k=8, max_files_inspected=3, max_question_terms=3, max_symbol_terms=0, file_read_line_cap=400
    ),
    "medium": _DepthConfig(
        top_k=12, max_files_inspected=5, max_question_terms=4, max_symbol_terms=3, file_read_line_cap=400
    ),
    "deep": _DepthConfig(
        top_k=20, max_files_inspected=8, max_question_terms=6, max_symbol_terms=5, file_read_line_cap=None
    ),
}


@dataclass
class SymbolMatch:
    """One line-level text match from a targeted search_code() call."""

    file: str
    line: int
    content: str

    @property
    def citation(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class FileInspection:
    """A file read in full or in part during targeted/medium/deep investigation."""

    file_path: str
    total_lines: int
    content: str
    truncated: bool

    @property
    def citation(self) -> str:
        shown_lines = len(self.content.splitlines())
        end = shown_lines if self.truncated else self.total_lines
        return f"{self.file_path}:1-{end}"


@dataclass
class Evidence:
    """Everything gathered about one investigated project for a question.

    Carries enough provenance (file paths, line ranges, symbol names) for a
    later structured-answer step to cite concrete evidence rather than
    assert unsupported claims -- and enough truthful project metadata
    (ecosystem/languages/frameworks, sourced from real manifest evidence via
    ProjectInfo, never from the question's own wording) that a later answer
    step can correct a user's mistaken terminology instead of silently
    going along with it.
    """

    project_root: str
    ecosystem: str
    languages: List[str]
    frameworks: List[str]
    build_system: Optional[str]
    package_manager: Optional[str]
    test_system: Optional[str]
    project_evidence: List[str]
    file_count: int
    chunks: List[RetrievedChunk] = field(default_factory=list)
    files_inspected: List[FileInspection] = field(default_factory=list)
    symbol_matches: List[SymbolMatch] = field(default_factory=list)

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks or self.files_inspected or self.symbol_matches)

    @property
    def citations(self) -> List[str]:
        return (
            [c.citation for c in self.chunks]
            + [f.citation for f in self.files_inspected]
            + [m.citation for m in self.symbol_matches]
        )


@dataclass
class InvestigationResult:
    """Top-level result of investigate().

    ``evidence`` holds one ``Evidence`` per project actually investigated
    (bounded -- see MAX_PROJECTS_INVESTIGATED). ``detected_projects`` lists
    every project found in the repository regardless of whether it was
    investigated, for full transparency about what RepositoryAnalyzer saw.
    """

    question: str
    depth: str
    detected_projects: List[ProjectInfo]
    investigated_projects: List[str]
    evidence: List[Evidence] = field(default_factory=list)
    no_evidence_reason: Optional[str] = None

    @property
    def has_evidence(self) -> bool:
        return any(e.has_evidence for e in self.evidence)

    @property
    def all_citations(self) -> List[str]:
        result: List[str] = []
        for e in self.evidence:
            result.extend(e.citations)
        return result


def _question_terms(text: str, limit: int) -> List[str]:
    """Extract candidate search terms from free text.

    Deliberately the same simple heuristic already used by the existing
    agent's investigate_node (app.services.agent.graph) for the identical
    purpose -- words longer than 3 characters, in order, deduplicated --
    rather than inventing a second convention.
    """
    if limit <= 0:
        return []
    seen: List[str] = []
    for word in text.split():
        cleaned = word.strip(".,?!:;()[]{}\"'`")
        if len(cleaned) > 3 and cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= limit:
            break
    return seen


def _symbol_terms(chunks: List[RetrievedChunk], limit: int) -> List[str]:
    """Real, code-derived search terms from the top RAG hits' own symbol
    names -- used for one-hop "who else references this" tracing at
    medium/deep depth, distinct from (and more reliable than) guessing
    terms purely from the question's own wording."""
    if limit <= 0:
        return []
    seen: List[str] = []
    for chunk in chunks:
        if chunk.symbol_name and chunk.symbol_name not in seen:
            seen.append(chunk.symbol_name)
        if len(seen) >= limit:
            break
    return seen


def _to_repo_relative(project_root: str, path: str) -> str:
    """Rewrite a path returned by search_code()/list_files() (relative to
    whichever directory they were pointed at) into a path relative to the
    overall repository root -- matching CodeChunk.file_path's existing
    convention, so every citation in Evidence is comparable regardless of
    which project it came from."""
    if project_root in (".", ""):
        return path
    return f"{project_root.rstrip('/')}/{path}"


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: List[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def _cheap_keyword_scan(workspace: Path, question: str) -> List[Dict[str, Any]]:
    """One bounded, whole-repository grep pass used only to help
    select_relevant_projects() pick the right project(s) in an ambiguous
    (multi-project) repository. See _PROJECT_SELECTION_SCAN_TERMS."""
    matches: List[Dict[str, Any]] = []
    for term in _question_terms(question, _PROJECT_SELECTION_SCAN_TERMS):
        res = tools.search_code(str(workspace), term)
        matches.extend(res.get("matches", [])[:_PROJECT_SELECTION_SCAN_MATCH_CAP])
    return matches


async def _investigate_project(
    workspace: Path,
    repository_id: int,
    question: str,
    project: ProjectInfo,
    cfg: _DepthConfig,
    db: Optional[AsyncSession],
    retriever: CodeRetriever,
) -> Evidence:
    """Gather Evidence for one already-selected project, bounded by ``cfg``."""
    project_path = workspace if project.root == "." else (workspace / project.root)
    file_prefix = None if project.root == "." else project.root

    chunks: List[RetrievedChunk] = []
    try:
        chunks = await retriever.retrieve_chunks(
            query=question,
            repository_id=repository_id,
            top_k=cfg.top_k,
            file_prefix=file_prefix,
            db=db,
        )
    except RetrievalError as e:
        logger.warning(f"RAG retrieval failed for project '{project.root}': {e}")
        chunks = []

    files_inspected: List[FileInspection] = []
    if cfg.max_files_inspected > 0:
        candidate_files = _unique_preserve_order([c.file_path for c in chunks])
        for f in candidate_files[: cfg.max_files_inspected]:
            res = tools.read_file(str(workspace), f, end_line=cfg.file_read_line_cap)
            if res.get("success"):
                total_lines = res.get("total_lines", 0)
                truncated = cfg.file_read_line_cap is not None and total_lines > cfg.file_read_line_cap
                files_inspected.append(
                    FileInspection(
                        file_path=f,
                        total_lines=total_lines,
                        content=res.get("content", ""),
                        truncated=truncated,
                    )
                )

    symbol_matches: List[SymbolMatch] = []
    if cfg.max_question_terms > 0 or cfg.max_symbol_terms > 0:
        terms = _unique_preserve_order(
            _question_terms(question, cfg.max_question_terms) + _symbol_terms(chunks, cfg.max_symbol_terms)
        )
        for term in terms:
            res = tools.search_code(str(project_path), term)
            for m in res.get("matches", [])[:_MAX_MATCHES_PER_TERM]:
                symbol_matches.append(
                    SymbolMatch(
                        file=_to_repo_relative(project.root, m["file"]),
                        line=m["line"],
                        content=m["content"],
                    )
                )
            if len(symbol_matches) >= _MAX_TOTAL_SYMBOL_MATCHES:
                break
        symbol_matches = symbol_matches[:_MAX_TOTAL_SYMBOL_MATCHES]

    # Lightweight "repository/project context" (required even at shallow
    # depth) -- a directory listing, not file content, so it stays cheap.
    file_count = 0
    listing = tools.list_files(str(project_path))
    if listing.get("success"):
        file_count = listing.get("total", 0)

    return Evidence(
        project_root=project.root,
        ecosystem=project.ecosystem,
        languages=list(project.languages),
        frameworks=list(project.frameworks),
        build_system=project.build_system,
        package_manager=project.package_manager,
        test_system=project.test_system,
        project_evidence=list(project.evidence),
        file_count=file_count,
        chunks=chunks,
        files_inspected=files_inspected,
        symbol_matches=symbol_matches,
    )


async def investigate(
    workspace_dir: str,
    repository_id: int,
    question: str,
    depth: str = "shallow",
    db: Optional[AsyncSession] = None,
    retriever: Optional[CodeRetriever] = None,
    max_projects: int = MAX_PROJECTS_INVESTIGATED,
) -> InvestigationResult:
    """Investigate a repository for a question, at the requested depth.

    Never calls an LLM -- this only gathers Evidence from the real
    repository; question classification and answer synthesis are separate,
    later stages that decide what to do with the result, including
    refusing to answer when ``InvestigationResult.has_evidence`` is False
    (see module docstring).

    Reuses, rather than duplicates, project detection (RepositoryAnalyzer),
    multi-project relevance selection (select_relevant_projects), and
    semantic retrieval (CodeRetriever) -- all already built and tested for
    the verification engine and the existing RAG endpoint respectively.
    """
    if depth not in _DEPTH_CONFIGS:
        raise ValueError(f"Unknown investigation depth {depth!r}; expected one of {VALID_DEPTHS}.")

    workspace = Path(workspace_dir).resolve()
    cfg = _DEPTH_CONFIGS[depth]
    retriever = retriever or CodeRetriever()

    if not workspace.is_dir():
        return InvestigationResult(
            question=question,
            depth=depth,
            detected_projects=[],
            investigated_projects=[],
            no_evidence_reason=f"Workspace directory does not exist: {workspace}",
        )

    projects = RepositoryAnalyzer.analyze(workspace)
    if not projects:
        return InvestigationResult(
            question=question,
            depth=depth,
            detected_projects=[],
            investigated_projects=[],
            no_evidence_reason=(
                "No recognizable project ecosystem was found in this repository, "
                "so no evidence could be gathered."
            ),
        )

    if len(projects) == 1:
        selected = list(projects)
    else:
        keyword_matches = _cheap_keyword_scan(workspace, question)
        selected = select_relevant_projects(projects, task_description=question, keyword_matches=keyword_matches)

    investigated = selected[:max_projects]

    evidence: List[Evidence] = []
    for project in investigated:
        evidence.append(
            await _investigate_project(
                workspace=workspace,
                repository_id=repository_id,
                question=question,
                project=project,
                cfg=cfg,
                db=db,
                retriever=retriever,
            )
        )

    result = InvestigationResult(
        question=question,
        depth=depth,
        detected_projects=projects,
        investigated_projects=[p.root for p in investigated],
        evidence=evidence,
    )

    if not result.has_evidence:
        described = ", ".join(e.ecosystem for e in evidence) or "this repository"
        result.no_evidence_reason = (
            f"No relevant code evidence was found for this question in {described}. "
            "This is not evidence that the requested component/feature does or does "
            "not exist beyond what was investigated."
        )

    return result
