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


def _generate_patches_with_gemini(
    task_description: str,
    retrieved_context: List[Dict[str, Any]],
    error_analysis: Optional[str] = None,
    workspace_dir: Optional[str] = None,
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


RETRIEVAL_LIMIT = 5


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Retrieve fresh code context directly from the workspace on disk.

    Candidate files are ranked so those with investigation keyword matches are
    retrieved first instead of relying on alphabetical order alone.
    """
    workspace = state["workspace_dir"]
    file_list = tools.list_files(workspace)
    files = file_list.get("files", [])

    ranked_files = _rank_candidate_files(files, state.get("keyword_matches") or [])[:RETRIEVAL_LIMIT]

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


def plan_node(state: AgentState) -> Dict[str, Any]:
    """Generate repair plan and structured proposed patches based on fresh context."""
    desc = state["task_description"]
    workspace = state.get("workspace_dir", "")
    error_analysis = state.get("error_analysis")

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
    """
    test_res = state.get("test_results") or {}
    patches = state.get("proposed_patches") or []
    is_verified = state.get("is_verified", False)
    applied_count = state.get("applied_patch_count")
    zero_applied = bool(patches) and applied_count == 0

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
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("plan", plan_node)
    builder.add_node("edit", edit_node)
    builder.add_node("test", test_node)
    builder.add_node("verify", verify_node)
    builder.add_node("analyze_failure", analyze_failure_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "retrieve")
    builder.add_edge("retrieve", "plan")
    builder.add_edge("plan", "edit")
    builder.add_edge("edit", "test")
    builder.add_edge("test", "verify")

    builder.add_conditional_edges(
        "verify",
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
