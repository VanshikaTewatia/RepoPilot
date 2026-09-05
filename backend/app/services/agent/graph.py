"""LangGraph multi-step reasoning agent with iterative self-correction."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from app.core.config import settings
from app.core.logging import logger
from app.services.agent.state import AgentState
from app.services.agent import tools
from app.services.baseline import (
    BridgeOutcome,
    EvidenceReference,
    ExitCodeSemantics,
    KnownCommand,
    ReproductionExpectation,
    ReproductionInput,
    RepositoryEvidence,
    build_reproduction_input,
    plan_reproduction,
    reproduce,
)
from app.services.verification.project_analyzer import ProjectInfo, RepositoryAnalyzer, select_relevant_projects
from app.services.diagnosis import diagnose, DiagnosisStatus
from app.services.agent.db import open_session
from app.services.rag.retriever import CodeRetriever, RetrievalError, RetrievedChunk


def validate_patch(patch: Any, workspace_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Validate a single patch dictionary for safe application against disk state."""
    if not isinstance(patch, dict):
        return None

    file_path = patch.get("file_path")
    code = patch.get("code")

    if not isinstance(file_path, str) or not file_path.strip():
        return None
    if not isinstance(code, str):
        return None

    clean_path = file_path.strip().replace("\\", "/")
    if clean_path.startswith("/") or ".." in clean_path.split("/"):
        logger.warning(f"Rejected patch with invalid path: {file_path}")
        return None

    start_line = patch.get("start_line")
    end_line = patch.get("end_line")

    if start_line is not None or end_line is not None:
        if not (isinstance(start_line, int) and isinstance(end_line, int)):
            return None
        if start_line < 1 or end_line < start_line:
            logger.warning(f"Rejected patch with invalid line range: {start_line}-{end_line}")
            return None

        # Validate against actual file length on disk if workspace is accessible
        if workspace_dir:
            try:
                target = Path(workspace_dir).resolve() / clean_path
                if target.is_file():
                    with open(target, "r", encoding="utf-8", errors="ignore") as f:
                        file_line_count = len(f.readlines())
                    if start_line > file_line_count + 1 or end_line > file_line_count:
                        logger.warning(
                            f"Rejected patch with out-of-bounds line range {start_line}-{end_line} "
                            f"(file '{clean_path}' has {file_line_count} lines)"
                        )
                        return None
            except Exception as e:
                logger.warning(f"Error checking file lines for patch validation: {e}")

    return {
        "file_path": clean_path,
        "code": code,
        "start_line": start_line,
        "end_line": end_line,
    }


def parse_and_validate_patches(raw_text: str, workspace_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse LLM JSON response and validate patch objects."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        try:
            start_idx = cleaned.find("[")
            end_idx = cleaned.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                parsed = json.loads(cleaned[start_idx : end_idx + 1])
            else:
                return []
        except Exception:
            return []

    if isinstance(parsed, dict):
        parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    valid_patches: List[Dict[str, Any]] = []
    for item in parsed:
        val = validate_patch(item, workspace_dir=workspace_dir)
        if val is not None:
            valid_patches.append(val)

    return valid_patches


# Markdown fence language per file extension for the patch-generation
# prompt's retrieved-context blocks. A fixed allow-list, not the raw
# extension string, so an unusual/malformed file extension can never become
# arbitrary prompt content -- unrecognized extensions fall back to "text".
_FENCE_LANGUAGE_BY_EXTENSION: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".dart": "dart",
    ".vue": "vue",
    ".html": "html",
    ".css": "css",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".sh": "bash",
}


def _fence_language_for_path(file_path: str) -> str:
    """Markdown fence language for a retrieved file's own extension.

    Always resolves to a fixed, known-safe token (never the raw extension
    string) -- an unrecognized extension falls back to "text" rather than
    the previous hardcoded "python" for every file regardless of its real
    language.
    """
    return _FENCE_LANGUAGE_BY_EXTENSION.get(Path(file_path).suffix.lower(), "text")


def _format_diagnosis_for_prompt(diagnosis: Dict[str, Any]) -> str:
    """Render a validated, DIAGNOSED ``Diagnosis.to_dict()`` as an advisory
    prompt section for ``_generate_patches_with_gemini``.

    Returns "" when there is nothing substantive to add (e.g. an empty
    summary and no hypotheses) so the caller never appends a hollow
    section -- see ``_generate_patches_with_gemini``'s byte-for-byte
    no-diagnosis guarantee.
    """
    summary = diagnosis.get("summary") or ""
    hypotheses = diagnosis.get("hypotheses") or []
    if not summary and not hypotheses:
        return ""

    lines = ["Root Cause Diagnosis (advisory -- verify against the context above before relying on it):"]
    if summary:
        lines.append(summary)
    for h in hypotheses:
        citations = h.get("citations") or []
        citation_str = ", ".join(
            f"{c.get('file_path')}:{c.get('start_line')}-{c.get('end_line')}" for c in citations
        )
        line = f"- Hypothesis {h.get('rank')}: {h.get('description', '')}"
        if citation_str:
            line += f" (evidence: {citation_str})"
        if h.get("suggested_fix_approach"):
            line += f" Suggested approach: {h['suggested_fix_approach']}"
        lines.append(line)
    return "\n".join(lines) + "\n\n"


def _generate_patches_with_gemini(
    task_description: str,
    retrieved_context: List[Dict[str, Any]],
    error_analysis: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    diagnosis: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate structured code patches using Gemini LLM."""
    if not settings.gemini_api_key or settings.gemini_api_key.startswith("test") or settings.gemini_api_key.startswith("mock"):
        return []

    context_parts = []
    for item in retrieved_context:
        fpath = item.get("file_path", "")
        content = item.get("content", "")
        total_lines = item.get("total_lines", 0)
        fence_lang = _fence_language_for_path(fpath)
        context_parts.append(f"### File: {fpath} ({total_lines} lines total)\n```{fence_lang}\n{content}\n```")
    context_str = "\n\n".join(context_parts)

    system_instruction = (
        "You are RepoPilot, an expert autonomous software engineer. "
        "Your task is to fix the described issue by proposing concrete, exact code patches. "
        "Return ONLY a JSON array of patch objects. Do not include markdown code fences or conversational text. "
        "Schema:\n"
        "[\n"
        "  {\n"
        "    \"file_path\": \"relative/path/to/file.py\",\n"
        "    \"code\": \"replacement or new code string\",\n"
        "    \"start_line\": 1,\n"
        "    \"end_line\": 10\n"
        "  }\n"
        "]\n"
        "If replacing the whole file, omit start_line and end_line or set them to null. "
        "Otherwise specify 1-based start_line and end_line inclusive based on the EXACT lines provided in the context.\n"
        "\n"
        "Strict minimality requirements:\n"
        "- Modify only files that are necessary to satisfy the requested task.\n"
        "- Make the smallest possible code change that fixes the issue.\n"
        "- Do not refactor unrelated code.\n"
        "- Do not rewrite docstrings or comments unless the task requires it.\n"
        "- Do not add unrelated improvements, cleanup, or extra features.\n"
        "- Do not modify behavior unrelated to the task.\n"
        "- Prefer targeted line replacements (start_line/end_line) over whole-file replacement.\n"
        "- If the requested behavior is already correctly implemented, return an empty array: []"
    )

    prompt = (
        f"Task Description:\n{task_description}\n\n"
        f"Context Code Files:\n{context_str}\n\n"
    )
    if error_analysis:
        prompt += f"Previous Attempt Test Failure Trace:\n{error_analysis}\n\nFix the failure in your new patch proposal.\n\n"

    if diagnosis and diagnosis.get("status") == DiagnosisStatus.DIAGNOSED.value:
        prompt += _format_diagnosis_for_prompt(diagnosis)

    prompt += "Generate the JSON patch array now:"

    try:
        from google import genai
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=prompt,
            config={"system_instruction": system_instruction},
        )
        raw_text = response.text or ""
        return parse_and_validate_patches(raw_text, workspace_dir=workspace_dir)
    except Exception as e:
        logger.error(f"Error generating patches with Gemini: {e}")
        return []


def _rank_candidate_files(files: List[str], keyword_matches: List[Dict[str, Any]]) -> List[str]:
    """Rank candidate files for retrieval using investigation keyword matches.

    Files with keyword matches rank first, ordered by descending match count;
    ties (and files with no matches) keep deterministic alphabetical order.
    """
    match_counts: Dict[str, int] = {}
    for match in keyword_matches:
        file_path = match.get("file") if isinstance(match, dict) else None
        if isinstance(file_path, str) and file_path:
            match_counts[file_path] = match_counts.get(file_path, 0) + 1

    matched = sorted(
        (f for f in files if f in match_counts),
        key=lambda f: (-match_counts[f], f),
    )
    unmatched = sorted(f for f in files if f not in match_counts)
    return matched + unmatched


def investigate_node(state: AgentState) -> Dict[str, Any]:
    """Inspect workspace files and search for relevant keywords and target test."""
    workspace = state["workspace_dir"]
    desc = state["task_description"]

    file_list = tools.list_files(workspace)
    files = file_list.get("files", [])

    keywords = [w for w in desc.split() if len(w) > 3]
    matches = []
    for kw in keywords[:3]:
        res = tools.search_code(workspace, kw)
        if res.get("matches"):
            matches.extend(res["matches"])

    # If test_target is not explicitly set, try to infer from task description
    test_target = state.get("test_target")
    if not test_target:
        test_pattern_match = re.search(r"(tests/[a-zA-Z0-9_\/]+\.py(?:::[a-zA-Z0-9_]+)?)", desc)
        if test_pattern_match:
            test_target = test_pattern_match.group(1)

    findings = f"Found {len(files)} files in workspace. Discovered {len(matches)} initial code matches."
    res_dict: Dict[str, Any] = {
        "status": "investigating",
        "investigation_findings": findings,
        "keyword_matches": matches,
        "messages": state.get("messages", []) + [{"role": "agent", "content": findings}],
    }
    if test_target:
        res_dict["test_target"] = test_target

    return res_dict


# ---------------------------------------------------------------------------
# Phase 4B-3: evidence-driven baseline reproduction.
#
# Runs once, between investigate and retrieve -- before any diagnosis/fix is
# attempted, and before the retry loop (analyze_failure -> retrieve) that
# never re-enters this node. Reuses the existing Phase 4B-1 planner, Phase
# 4B-2 bridge, and Phase 4A executor/classifier exactly as they already
# exist: no second executor, no direct Docker access, no independent
# repository scan -- RepositoryAnalyzer (the same deterministic analysis
# VerificationEngine.verify_repository already uses) is the sole source of
# detected-project evidence.
# ---------------------------------------------------------------------------

# Node's own test/build command is deliberately absent from
# project_analyzer's ECOSYSTEM_METADATA (it's read from each project's own
# package.json, not a fixed convention) -- derived here directly from that
# same manifest file, never invented, and bounded to the two script names a
# reproduction could plausibly need.
_NODE_KNOWN_SCRIPT_NAMES = ("test", "build")


def _known_commands_for_project(workspace: str, project: ProjectInfo) -> List[KnownCommand]:
    """Real, evidence-backed commands for one detected project -- never a
    guess. ``source_file`` is always a manifest RepositoryAnalyzer/
    ProjectDetector already used as evidence for this project's ecosystem,
    so plan_validator's evidence_refs check can verify against it."""
    commands: List[KnownCommand] = []
    source_file = project.evidence[0] if project.evidence else None

    if project.test_system and source_file:
        parts = project.test_system.split()
        if parts:
            commands.append(
                KnownCommand(
                    command=parts,
                    description=f"{project.ecosystem} project's conventional test command",
                    source_file=source_file,
                )
            )

    if project.ecosystem == "node":
        manifest_rel = "package.json" if project.root == "." else f"{project.root}/package.json"
        content_res = tools.read_file(workspace, manifest_rel)
        if content_res.get("success"):
            try:
                scripts = json.loads(content_res.get("content", "{}")).get("scripts") or {}
            except Exception:
                scripts = {}
            for name in _NODE_KNOWN_SCRIPT_NAMES:
                if isinstance(scripts.get(name), str):
                    commands.append(
                        KnownCommand(
                            command=["npm", "run", name],
                            description=f"package.json '{name}' script: {scripts[name]}",
                            source_file=manifest_rel,
                        )
                    )

    return commands


def _evidence_references_from_matches(
    keyword_matches: List[Dict[str, Any]], limit: int = 10
) -> List[EvidenceReference]:
    """Bounded EvidenceReferences from investigate_node's own search
    results -- the same keyword_matches already used to rank retrieval and
    select relevant projects, never a fresh search."""
    refs: List[EvidenceReference] = []
    for match in keyword_matches[:limit]:
        if not isinstance(match, dict):
            continue
        file_path = match.get("file")
        if not file_path:
            continue
        line = match.get("line")
        description = (match.get("content") or "").strip()[:200]
        refs.append(EvidenceReference(file_path=file_path, description=description, line_start=line, line_end=line))
    return refs


def _build_repository_evidence(state: AgentState) -> RepositoryEvidence:
    """Assemble RepositoryEvidence from the SAME deterministic repository
    analysis and investigation data the rest of the agent already produced
    -- never a fresh independent scan, and never derived from the user's
    own wording (task_description is used by select_relevant_projects only
    to narrow *which already-detected* project(s) are relevant, exactly as
    VerificationEngine.verify_repository already does -- it is never used to
    decide what ecosystems/projects exist)."""
    workspace = state["workspace_dir"]
    task_description = state.get("task_description", "")
    keyword_matches = state.get("keyword_matches") or []

    projects = RepositoryAnalyzer.analyze(Path(workspace))
    selected = (
        projects
        if len(projects) <= 1
        else select_relevant_projects(projects, task_description=task_description, keyword_matches=keyword_matches)
    )

    known_commands: List[KnownCommand] = []
    for project in selected:
        known_commands.extend(_known_commands_for_project(workspace, project))

    return RepositoryEvidence(
        detected_projects=selected,
        known_commands=known_commands,
        investigation_findings=state.get("investigation_findings", ""),
        evidence_references=_evidence_references_from_matches(keyword_matches),
    )


def _reproduction_input_to_state_dict(ri: ReproductionInput) -> Dict[str, Any]:
    """Minimal, JSON-safe serialization of the validated execution
    specification actually run for baseline reproduction (Phase 5) --
    deliberately excludes any command OUTPUT (that lives in baseline_result
    /post_fix_reproduction_result, each already bounded by Phase 4A) so only
    the small, reusable specification itself is retained in graph state."""
    return {
        "workspace_path": ri.workspace_path,
        "commands": ri.commands,
        "working_dir": ri.working_dir,
        "timeout_seconds": ri.timeout_seconds,
        "image": ri.image,
        "expectation": {
            "exit_code_semantics": ri.expectation.exit_code_semantics.value,
            "reproduced_output_pattern": ri.expectation.reproduced_output_pattern,
            "not_reproduced_output_pattern": ri.expectation.not_reproduced_output_pattern,
        },
        "task_context": ri.task_context,
    }


def _reproduction_input_from_state_dict(data: Dict[str, Any]) -> ReproductionInput:
    """Reconstruct the EXACT ReproductionInput previously validated and
    executed for baseline reproduction -- never regenerated, re-planned, or
    guessed; see ``_reproduction_input_to_state_dict``, its only producer."""
    expectation_data = data.get("expectation") or {}
    return ReproductionInput(
        workspace_path=data["workspace_path"],
        commands=data.get("commands") or [],
        working_dir=data.get("working_dir"),
        timeout_seconds=data.get("timeout_seconds"),
        image=data.get("image"),
        expectation=ReproductionExpectation(
            exit_code_semantics=ExitCodeSemantics(expectation_data.get("exit_code_semantics", "ignore")),
            reproduced_output_pattern=expectation_data.get("reproduced_output_pattern"),
            not_reproduced_output_pattern=expectation_data.get("not_reproduced_output_pattern"),
        ),
        task_context=data.get("task_context"),
    )


async def baseline_node(state: AgentState) -> Dict[str, Any]:
    """Establish, before any fix is attempted, whether the reported bug can
    be independently demonstrated against the isolated task workspace.

    This is evidence gathering ONLY -- it never itself declares the task
    fixed/unfixed. ``finalize_node`` is the sole place ``baseline_status``
    is allowed to affect the task's final outcome, and only to prevent a
    false FIXED claim; a planner/bridge failure is always surfaced as
    UNABLE_TO_REPRODUCE here, never silently reinterpreted as NOT_APPLICABLE
    (which is reserved for a genuine "no evidence-backed reproduction
    exists" verdict) or as any positive/negative claim about the bug.

    Whenever the bridge actually produces an executable ``ReproductionInput``
    (regardless of what executing it returns), its specification is
    retained verbatim in ``reproduction_spec`` (Phase 5) -- see
    ``post_fix_reproduction_node``, which reruns exactly this after an edit
    is applied, never a newly-planned reproduction.
    """
    workspace = state["workspace_dir"]
    task_description = state.get("task_description", "")
    reproduction_spec: Optional[Dict[str, Any]] = None

    try:
        evidence = _build_repository_evidence(state)
        plan = await plan_reproduction(task_description, evidence)

        if plan.planning_failed:
            status, result, detail = "UNABLE_TO_REPRODUCE", None, (plan.failure_reason or plan.reason)
        elif not plan.applicable:
            status, result, detail = "NOT_APPLICABLE", None, plan.reason
        else:
            bridge_result = build_reproduction_input(plan, evidence, workspace_path=workspace)
            if bridge_result.outcome == BridgeOutcome.PLANNING_FAILED:
                status, result, detail = "UNABLE_TO_REPRODUCE", None, bridge_result.detail
            elif bridge_result.outcome == BridgeOutcome.NOT_APPLICABLE:
                status, result, detail = "NOT_APPLICABLE", None, bridge_result.detail
            else:
                reproduction_spec = _reproduction_input_to_state_dict(bridge_result.reproduction_input)
                baseline_result = reproduce(bridge_result.reproduction_input)
                status = baseline_result.status.value
                result = baseline_result.to_dict()
                detail = baseline_result.detail
    except Exception as e:  # noqa: BLE001 -- a baseline-integration failure must never crash the task, and must never be silently treated as a verdict about the bug
        logger.warning(f"Baseline reproduction integration failed unexpectedly: {e}")
        status, result, detail = "UNABLE_TO_REPRODUCE", None, f"Baseline reproduction failed unexpectedly: {e}"

    message = f"Baseline reproduction: {status} -- {detail}"
    return {
        "status": "baseline_checked",
        "baseline_status": status,
        "baseline_result": result,
        "baseline_detail": detail,
        "reproduction_spec": reproduction_spec,
        "messages": state.get("messages", []) + [{"role": "agent", "content": message}],
    }


RETRIEVAL_LIMIT = 5

# ---------------------------------------------------------------------------
# Phase 6B: semantic candidate-file selection for retrieve_node.
#
# Upgrades WHICH files retrieve_node reads, never WHAT is stored about them
# or how they're read: retrieved_context's shape (full-file reads, real
# on-disk line numbers) is completely unchanged, so plan_node/edit_node/
# diagnose_node/diagnoser.py need no changes at all. Semantic retrieval is
# purely an additional, bounded, best-effort source of candidate file
# paths merged ahead of the existing keyword-derived ranking; any failure
# (no repository_id, no DB session, RAG/embedding failure, empty index)
# degrades to exactly the pre-Phase-6B keyword-only ranking, never raises,
# and never blocks retrieval.
# ---------------------------------------------------------------------------

# Small, fixed cap on how many detected projects a single retrieve_node pass
# ever issues a semantic query against -- mirrors app.services.qa.
# investigator.MAX_PROJECTS_INVESTIGATED's "small fixed budget, even when
# project selection is completely ambiguous" rationale, applied here for
# the agent's own (already more localized) retrieval.
SEMANTIC_MAX_PROJECTS = 2

# Fixed top_k for the agent's own semantic query -- matches
# app.services.qa.investigator's "targeted" depth tier, never the caller-
# configurable settings.vector_top_k default.
SEMANTIC_TOP_K = 8


def _semantic_query_text(task_description: str, error_analysis: Optional[str]) -> str:
    """Query text for semantic retrieval -- includes the CURRENT
    error_analysis when present, so a retry's semantic search reflects why
    the previous attempt failed, not just the original task description.
    Recomputed fresh on every retrieve_node call, never cached across
    retries."""
    if error_analysis:
        return f"{task_description}\n\n{error_analysis}"
    return task_description


def _semantic_candidate_files(chunks: List[RetrievedChunk]) -> List[str]:
    """Deduplicated, order-preserving file paths from semantic RAG hits."""
    seen: List[str] = []
    for chunk in chunks:
        if chunk.file_path and chunk.file_path not in seen:
            seen.append(chunk.file_path)
    return seen


def _merge_candidate_files(
    semantic_files: List[str], keyword_ranked_files: List[str], limit: int
) -> List[str]:
    """Merge semantic-hit file paths AHEAD OF the existing keyword-ranked
    ones, deduplicated (first occurrence wins) and capped at ``limit``.

    When ``semantic_files`` is empty (semantic retrieval unavailable or
    contributed nothing), this is byte-identical to the pre-Phase-6B
    keyword-only ranking sliced the same way -- the core degrade-
    gracefully guarantee.
    """
    merged: List[str] = []
    for f in list(semantic_files) + list(keyword_ranked_files):
        if f not in merged:
            merged.append(f)
    return merged[:limit]


def _projects_for_semantic_retrieval(
    state: AgentState, limit: int = SEMANTIC_MAX_PROJECTS
) -> List[ProjectInfo]:
    """Bounded, real project scoping for semantic retrieval -- reuses the
    SAME repository evidence baseline_node already computes
    (_build_repository_evidence), never a fresh/independent scan, capped
    to the first ``limit`` selected projects so semantic retrieval never
    issues more than ``limit`` embedding calls per retrieve_node pass
    regardless of how many projects a monorepo contains or how ambiguous
    project selection is (select_relevant_projects returns every detected
    project, unbounded, when nothing disambiguates them)."""
    evidence = _build_repository_evidence(state)
    return list(evidence.detected_projects)[:limit]


async def _gather_semantic_candidate_files(
    state: AgentState,
    repository_id: int,
    db: Optional[Any],
    retriever: Optional[CodeRetriever] = None,
) -> List[str]:
    """Bounded, best-effort semantic candidate file list for retrieve_node.

    Never raises: any failure (project scoping, RAG/embedding error, an
    unexpected exception) degrades to an empty list, which makes
    ``_merge_candidate_files`` fall back to byte-identical keyword-only
    ranking. ``db`` may legitimately be ``None`` (see
    app.services.agent.db.open_session) -- CodeRetriever.retrieve_chunks
    already treats that as "skip semantic retrieval" rather than an error.
    """
    retriever = retriever or CodeRetriever()
    query = _semantic_query_text(state.get("task_description", ""), state.get("error_analysis"))

    try:
        projects = _projects_for_semantic_retrieval(state)
    except Exception as e:  # noqa: BLE001 -- project scoping must never block retrieval
        logger.warning(f"Semantic retrieval project scoping failed unexpectedly: {e}")
        return []

    chunks: List[RetrievedChunk] = []
    for project in projects:
        file_prefix = None if project.root == "." else project.root
        try:
            chunks.extend(
                await retriever.retrieve_chunks(
                    query=query,
                    repository_id=repository_id,
                    top_k=SEMANTIC_TOP_K,
                    file_prefix=file_prefix,
                    db=db,
                )
            )
        except RetrievalError as e:
            logger.warning(f"Semantic retrieval failed for project '{project.root}': {e}")
            continue
        except Exception as e:  # noqa: BLE001 -- semantic retrieval is advisory-only, must never crash retrieve_node
            logger.warning(f"Semantic retrieval failed unexpectedly for project '{project.root}': {e}")
            continue

    return _semantic_candidate_files(chunks)


async def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve fresh code context directly from the workspace on disk.

    Candidate files are ranked by combining bounded, project-scoped
    semantic RAG retrieval (Phase 6B) with the existing investigation
    keyword matches -- semantic hits rank first, deduplicated, capped at
    RETRIEVAL_LIMIT. retrieved_context's own shape (full-file reads, real
    on-disk line numbers) is unchanged from before Phase 6B, so
    plan_node/edit_node/diagnose_node need no changes. When semantic
    retrieval is unavailable or contributes nothing for any reason (no
    repository_id in state, no DB session, RAG/embedding failure, empty
    index), ranking is byte-identical to the pre-Phase-6B keyword-only
    behavior -- see _merge_candidate_files.
    """
    workspace = state["workspace_dir"]
    file_list = tools.list_files(workspace)
    files = file_list.get("files", [])

    keyword_ranked = _rank_candidate_files(files, state.get("keyword_matches") or [])

    semantic_files: List[str] = []
    repository_id = state.get("repository_id")
    if repository_id is not None:
        try:
            async with open_session() as db:
                semantic_files = await _gather_semantic_candidate_files(state, repository_id, db)
        except Exception as e:  # noqa: BLE001 -- semantic retrieval must never block retrieval
            logger.warning(f"Semantic candidate retrieval failed unexpectedly: {e}")
            semantic_files = []

    # A stale/mismatched chunk-index entry must never introduce a file that
    # no longer exists (or never existed at this path) into the bounded
    # candidate list -- restrict semantic hits to files actually present in
    # the current workspace, exactly like the existing keyword ranking
    # already implicitly does by only ever reordering `files`.
    files_set = set(files)
    semantic_files = [f for f in semantic_files if f in files_set]

    ranked_files = _merge_candidate_files(semantic_files, keyword_ranked, RETRIEVAL_LIMIT)

    retrieved: List[Dict[str, Any]] = []
    for f in ranked_files:
        content_res = tools.read_file(workspace, f)
        if content_res.get("success"):
            retrieved.append({
                "file_path": f,
                "content": content_res.get("content", ""),
                "total_lines": content_res.get("total_lines", 0),
            })

    return {
        "status": "retrieved",
        "retrieved_context": retrieved,
    }


async def diagnose_node(state: AgentState) -> Dict[str, Any]:
    """Evidence-driven root-cause diagnosis (Phase 6A) -- advisory only.

    Runs once between retrieve_node and plan_node, and again on every retry
    (analyze_failure -> retrieve -> diagnose -> plan), always working from
    the freshly retrieved_context and the current error_analysis so a
    diagnosis never goes stale across attempts. This is evidence gathering
    ONLY -- it never itself proposes a patch, and a DIAGNOSIS_FAILED or
    INSUFFICIENT_EVIDENCE result must never block or alter plan_node's
    existing patch-generation behavior; see app.services.diagnosis's
    package docstring. Never calls reproduction, Docker, workspace
    creation, or remote code-hosting functionality, and never raises -- a
    failure here must never crash the task.
    """
    task_description = state.get("task_description", "")
    retrieved_context = state.get("retrieved_context") or []
    error_analysis = state.get("error_analysis")

    try:
        result = await diagnose(
            task_description=task_description,
            retrieved_context=retrieved_context,
            error_analysis=error_analysis,
        )
    except Exception as e:  # noqa: BLE001 -- diagnosis is advisory-only and must never crash the task
        logger.warning(f"Diagnosis failed unexpectedly: {e}")
        return {
            "diagnosis_status": DiagnosisStatus.DIAGNOSIS_FAILED.value,
            "diagnosis": None,
            "diagnosis_detail": f"Diagnosis failed unexpectedly: {e}",
        }

    detail = result.summary or result.failure_reason or result.status.value
    return {
        "diagnosis_status": result.status.value,
        "diagnosis": result.to_dict(),
        "diagnosis_detail": detail,
    }


def plan_node(state: AgentState) -> Dict[str, Any]:
    """Generate repair plan and structured proposed patches based on fresh context."""
    desc = state["task_description"]
    workspace = state.get("workspace_dir", "")
    error_analysis = state.get("error_analysis")
    diagnosis = state.get("diagnosis")

    # Always ensure retrieved_context contains the latest file content from disk
    fresh_retrieved: List[Dict[str, Any]] = []
    for item in state.get("retrieved_context", []):
        fpath = item.get("file_path", "")
        content_res = tools.read_file(workspace, fpath)
        if content_res.get("success"):
            fresh_retrieved.append({
                "file_path": fpath,
                "content": content_res.get("content", ""),
                "total_lines": content_res.get("total_lines", 0),
            })
        else:
            fresh_retrieved.append(item)

    # Generate new patches using fresh context
    if error_analysis:
        patches = _generate_patches_with_gemini(
            task_description=desc,
            retrieved_context=fresh_retrieved,
            error_analysis=error_analysis,
            workspace_dir=workspace,
            diagnosis=diagnosis,
        )
    else:
        existing_patches = state.get("proposed_patches")
        if existing_patches:
            patches = existing_patches
        else:
            patches = _generate_patches_with_gemini(
                task_description=desc,
                retrieved_context=fresh_retrieved,
                error_analysis=None,
                workspace_dir=workspace,
                diagnosis=diagnosis,
            )

    files_str = ", ".join(c["file_path"] for c in fresh_retrieved)
    patch_summary = f"Generated {len(patches)} patch(es)" if patches else "No valid patches generated"
    plan = f"Plan for '{desc}': Targets [{files_str}]. {patch_summary}."
    return {
        "status": "planning",
        "repair_plan": plan,
        "retrieved_context": fresh_retrieved,
        "proposed_patches": patches,
    }


def edit_node(state: AgentState) -> Dict[str, Any]:
    """Apply targeted patches to the repository workspace."""
    attempt = state.get("attempt_count", 0) + 1
    patches = state.get("proposed_patches", [])
    workspace = state["workspace_dir"]

    applied_count = 0
    for p in patches:
        fpath = p.get("file_path")
        code = p.get("code")
        s_line = p.get("start_line")
        e_line = p.get("end_line")
        if fpath and code:
            res = tools.apply_patch(workspace, fpath, code, s_line, e_line)
            if res.get("success"):
                applied_count += 1

    return {
        "status": "edited",
        "attempt_count": attempt,
        "applied_patch_count": applied_count,
    }


def test_node(state: AgentState) -> Dict[str, Any]:
    """Run the repository's own verification command(s) targeting the
    specific test or whole suite. Repository- and task-aware: detects every
    real project in the workspace and runs the one(s) relevant to this task
    rather than assuming the workspace root is a single project (see
    tools.run_tests_for_task / VerificationEngine.verify_repository)."""
    workspace = state["workspace_dir"]
    test_target = state.get("test_target")
    test_res = tools.run_tests_for_task(
        workspace,
        task_description=state.get("task_description", ""),
        keyword_matches=state.get("keyword_matches"),
        test_path=test_target,
    )

    return {
        "status": "tested",
        "test_results": test_res,
    }


def verify_node(state: AgentState) -> Dict[str, Any]:
    """Verify test outputs."""
    test_res = state.get("test_results") or {}
    success = test_res.get("success", False) and test_res.get("failed", 0) == 0

    return {
        "is_verified": success,
        "status": "verified" if success else "verification_failed",
    }


# ---------------------------------------------------------------------------
# Phase 5: targeted post-fix reproduction.
#
# Closes the evidence gap a REPRODUCED baseline leaves open: a passing full
# test suite establishes the *general* verification semantics already
# present in this project, but never re-confirms that the SPECIFIC failure
# baseline demonstrated is actually gone. This node reruns the EXACT
# ReproductionInput captured in reproduction_spec -- never a new plan, never
# regenerated commands/image/working_dir/timeout -- through the same Phase
# 4A `reproduce()` entry point baseline_node already uses. No second
# executor, no direct Docker access, no new workspace.
#
# Positioned after verify (not unconditionally after every edit) so it only
# ever does real work on an attempt whose own mechanical verification
# already passed AND whose baseline was REPRODUCED -- every other attempt/
# task is a single dict-lookup no-op. should_continue's retry-routing logic
# is intentionally NOT modified: it still routes solely on is_verified, so
# no new retry semantics are invented here (a still-reproducing post-fix
# result is instead caught by finalize_node, which is the only place this
# is allowed to prevent -- never manufacture -- a FIXED claim).
# ---------------------------------------------------------------------------
async def post_fix_reproduction_node(state: AgentState) -> Dict[str, Any]:
    """Rerun reproduction_spec (if any) against the isolated workspace after
    edit_node applied this attempt's candidate fix.

    A fast no-op unless baseline_status == "REPRODUCED" and this attempt's
    own verification already passed -- every task/attempt without a
    REPRODUCED baseline, and every failing attempt (which is already headed
    back to analyze_failure or to a FAILED finalize regardless of this
    node), is completely unaffected.
    """
    if state.get("baseline_status") != "REPRODUCED":
        return {}
    if not state.get("is_verified"):
        return {}
    spec = state.get("reproduction_spec")
    if not spec:
        return {}

    try:
        reproduction_input = _reproduction_input_from_state_dict(spec)
        result = reproduce(reproduction_input)
        status = result.status.value
        result_dict = result.to_dict()
        detail = result.detail
    except Exception as e:  # noqa: BLE001 -- a post-fix integration failure must never crash the task, and must never be silently treated as "the failure is gone"
        logger.warning(f"Post-fix reproduction failed unexpectedly: {e}")
        status, result_dict, detail = "UNABLE_TO_REPRODUCE", None, f"Post-fix reproduction failed unexpectedly: {e}"

    message = f"Post-fix reproduction: {status} -- {detail}"
    return {
        "post_fix_reproduction_status": status,
        "post_fix_reproduction_result": result_dict,
        "post_fix_reproduction_detail": detail,
        "messages": state.get("messages", []) + [{"role": "agent", "content": message}],
    }


def analyze_failure_node(state: AgentState) -> Dict[str, Any]:
    """Analyze test failure output to refine the next edit attempt."""
    test_res = state.get("test_results") or {}
    output = test_res.get("output", "No test output")

    analysis = f"Failure in attempt {state.get('attempt_count')}: {output[:300]}"
    return {
        "status": "analyzing_failure",
        "error_analysis": analysis,
        "messages": state.get("messages", []) + [{"role": "agent", "content": analysis}],
    }


def should_continue(state: AgentState) -> str:
    """Route based on verification result and retry budget."""
    if state.get("is_verified", False):
        return "human_approval"
    if state.get("attempt_count", 0) < state.get("max_attempts", 3):
        return "analyze_failure"
    return "failed"


def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Classify the run's real-world outcome.

    This is distinct from `is_verified`/`status`, which only describe the
    mechanical retry-loop result:

    - UNABLE_TO_VERIFY: verification tooling could not actually run (e.g. an
      unsupported ecosystem, or a required toolchain missing). Never
      conflated with "the bug doesn't exist" -- see
      VerificationEngine._run_adapter's tool-missing detection.
    - NO_CHANGE_NEEDED: verification passed without any code changes, i.e.
      the reported behavior was already correct or could not be
      substantiated as a bug (the plan step is explicitly instructed to
      return zero patches in that case) -- or one or more patches were
      *generated* but none of them actually *applied* to the workspace
      (see ``applied_patch_count`` below), so verification passing still
      reflects the repository's unmodified state, never a real fix.
    - FIXED: one or more patches were both generated AND actually applied
      to the workspace, and verification passed.
    - FAILED: the reported issue could not be verified as fixed within the
      attempt budget. Distinguishes "patches were generated but none of
      them could be applied" from a genuine post-edit verification failure
      via ``applied_patch_count``, so the two are never conflated in the
      reported detail.

    ``applied_patch_count`` (set by ``edit_node``) is deliberately treated
    as *unknown* (``None``), not zero, when absent from state -- only an
    explicit ``0`` downgrades an otherwise-FIXED result, so a caller that
    never ran a real edit pass (e.g. a unit test constructing state by
    hand) sees the same behavior as before this field existed.

    ``baseline_status`` (Phase 4B-3, set by ``baseline_node``) gates ONLY
    the one branch below that would otherwise claim FIXED -- a passing test
    suite plus an applied patch is not, by itself, evidence that the
    *reported* bug ever existed or is now gone; baseline reproduction is
    evidence gathering, never proof of correctness. Both "UNABLE_TO_REPRODUCE"
    and "NOT_REPRODUCED" downgrade this branch to UNABLE_TO_VERIFY -- never
    NO_CHANGE_NEEDED, which specifically (and here, falsely) means "no code
    change was actually made". The two remain distinguishable via
    ``baseline_status``/``baseline_detail`` even though they share this one
    conservative outcome category; see the NOT_REPRODUCED branch below for
    why NO_CHANGE_NEEDED would misrepresent it. Every other branch
    (NO_CHANGE_NEEDED, FAILED, tooling UNABLE_TO_VERIFY) already means "no
    fix is being claimed" and is left untouched regardless of
    ``baseline_status`` -- including when it is absent entirely (``None``,
    e.g. a hand-constructed state, or any task predating this integration),
    which behaves identically to before this field existed.

    ``post_fix_reproduction_status`` (Phase 5, set by
    ``post_fix_reproduction_node``) applies one further, narrower gate: when
    ``baseline_status == "REPRODUCED"`` (the reported failure WAS
    independently demonstrated before the edit), reaching FIXED additionally
    requires the SAME reproduction to have positively stopped reproducing
    after the edit (``post_fix_reproduction_status == "NOT_REPRODUCED"``).
    Still reproducing, unable to re-check, or never actually re-checked
    (``None`` -- e.g. ``post_fix_reproduction_node`` itself never ran for
    this attempt) all block FIXED here -- a general passing test suite is
    never sufficient by itself to claim the *specific* previously-reproduced
    failure is gone. This never widens which tasks can reach FIXED, only
    narrows the one REPRODUCED case further.
    """
    test_res = state.get("test_results") or {}
    patches = state.get("proposed_patches") or []
    is_verified = state.get("is_verified", False)
    applied_count = state.get("applied_patch_count")
    zero_applied = bool(patches) and applied_count == 0
    baseline_status = state.get("baseline_status")
    post_fix_status = state.get("post_fix_reproduction_status")

    if not test_res.get("available", True):
        outcome = "UNABLE_TO_VERIFY"
        detail = test_res.get("detail") or "Verification tooling was unavailable for this repository."
    elif is_verified and zero_applied:
        outcome = "NO_CHANGE_NEEDED"
        detail = (
            f"{len(patches)} patch(es) were generated but none could be applied to "
            "the workspace, so no code change was actually made -- verification "
            "passed against the repository's unmodified state."
        )
    elif is_verified and not patches:
        outcome = "NO_CHANGE_NEEDED"
        detail = (
            "Verification passed without any code changes; the reported behavior "
            "was already correct or the claimed issue could not be substantiated."
        )
    elif is_verified and baseline_status == "UNABLE_TO_REPRODUCE":
        # A patch applied and full verification passed, but baseline
        # reproduction never established that the reported issue actually
        # existed -- claiming FIXED here would be an unsupported guess.
        outcome = "UNABLE_TO_VERIFY"
        detail = (
            "The patch applied and verification passed, but baseline reproduction "
            "could not establish whether the reported issue actually existed "
            f"({state.get('baseline_detail') or 'inconclusive'}). This is not "
            "evidence the fix is correct or incorrect."
        )
    elif is_verified and baseline_status == "NOT_REPRODUCED":
        # NOT_REPRODUCED means the reproduction procedure executed
        # successfully and did not observe the reported behavior -- it does
        # NOT mean "verified as already correct". NO_CHANGE_NEEDED's actual
        # meaning elsewhere in this function is "no code change was
        # actually made" (zero patches, or none applied); that is false
        # here -- a patch WAS generated and applied, so labeling this
        # NO_CHANGE_NEEDED would misrepresent that a real change occurred.
        # A reproduction procedure can also fail to observe a real issue
        # because the procedure itself was incomplete or state-dependent,
        # not because the bug is absent -- so this must not be read as
        # confirmation the code was already correct either. UNABLE_TO_VERIFY
        # is the most conservative existing outcome that doesn't claim
        # either: it already means "no confident determination could be
        # made", which is exactly this situation. baseline_status in state
        # (and the detail text below) keeps NOT_REPRODUCED distinguishable
        # from UNABLE_TO_REPRODUCE even though both map to this outcome.
        outcome = "UNABLE_TO_VERIFY"
        detail = (
            "The patch applied and verification passed, but baseline reproduction "
            "did not observe the reported issue before the fix "
            f"({state.get('baseline_detail') or 'no evidence of the reported behavior'}). "
            "This does not confirm the reported issue was already absent -- the "
            "reproduction procedure may not have captured it -- so this is not "
            "reported as a confirmed fix."
        )
    elif is_verified and baseline_status == "REPRODUCED" and post_fix_status != "NOT_REPRODUCED":
        # Baseline independently demonstrated the reported failure before
        # the edit. A passing general test suite alone does not establish
        # that THIS SPECIFIC failure is gone -- only rerunning the exact
        # same reproduction (post_fix_reproduction_node) can. Still
        # reproducing is a definitive "not fixed"; unable to re-check (or
        # never re-checked at all) is an inconclusive "not fixed either" --
        # neither may become FIXED.
        if post_fix_status == "REPRODUCED":
            outcome = "FAILED"
            detail = (
                "The patch applied and the full test suite passed, but the SAME "
                "targeted reproduction that established this issue before the fix "
                "still demonstrates it afterward "
                f"({state.get('post_fix_reproduction_detail') or 'failure still observed'}). "
                "The reported issue has not been fixed."
            )
        else:
            outcome = "UNABLE_TO_VERIFY"
            detail = (
                "The patch applied and the full test suite passed, but the targeted "
                "post-fix reproduction could not confirm the previously-established "
                "failure is gone "
                f"({state.get('post_fix_reproduction_detail') or 'post-fix reproduction did not run or was inconclusive'}). "
                "This is not evidence the fix is correct or incorrect."
            )
    elif is_verified:
        outcome = "FIXED"
        applied_display = applied_count if applied_count is not None else len(patches)
        detail = f"Applied {applied_display} patch(es) and verification passed."
    elif zero_applied:
        outcome = "FAILED"
        detail = (
            f"{len(patches)} patch(es) were generated but none could be applied to "
            "the workspace; the reported issue could not be verified as fixed "
            "within the attempt budget."
        )
    else:
        outcome = "FAILED"
        detail = "The reported issue could not be verified as fixed within the attempt budget."

    return {
        "outcome": outcome,
        "outcome_detail": detail,
    }


def build_agent_graph():
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("investigate", investigate_node)
    builder.add_node("baseline", baseline_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("diagnose", diagnose_node)
    builder.add_node("plan", plan_node)
    builder.add_node("edit", edit_node)
    builder.add_node("test", test_node)
    builder.add_node("verify", verify_node)
    builder.add_node("post_fix_reproduction", post_fix_reproduction_node)
    builder.add_node("analyze_failure", analyze_failure_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "baseline")
    builder.add_edge("baseline", "retrieve")
    builder.add_edge("retrieve", "diagnose")
    builder.add_edge("diagnose", "plan")
    builder.add_edge("plan", "edit")
    builder.add_edge("edit", "test")
    builder.add_edge("test", "verify")
    builder.add_edge("verify", "post_fix_reproduction")

    builder.add_conditional_edges(
        "post_fix_reproduction",
        should_continue,
        {
            "human_approval": "finalize",
            "analyze_failure": "analyze_failure",
            "failed": "finalize",
        },
    )
    builder.add_edge("analyze_failure", "retrieve")
    builder.add_edge("finalize", END)

    return builder.compile()


agent_app = build_agent_graph()
