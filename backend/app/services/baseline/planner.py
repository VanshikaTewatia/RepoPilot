"""Evidence-driven reproduction planner (Phase 4B-1).

Answers exactly one question: "what is the safest, most evidence-supported
procedure we could execute to determine whether this reported bug actually
exists?" -- and returns a *proposal* (``ReproductionPlan``), never a verdict.
It must never itself claim the bug exists, execute anything, or classify an
outcome; that division of labor (execute -> classify) already belongs to
``app.services.baseline.executor``/``classifier``, which this module never
calls (Phase 4B-2 is responsible for wiring a validated plan into an actual
``ReproductionInput``/``reproduce()`` call).

Reuses the exact Gemini call/parsing convention already used throughout this
codebase (see ``app.services.qa.classifier.classify_question`` and
``app.services.agent.graph._generate_patches_with_gemini``): a plain
``genai.Client(...).models.generate_content(...)`` call with a JSON-only
system instruction, parsed via the shared ``app.services.qa.json_utils``
helper (no second JSON parser), with a real/test/mock API-key check
performed before ever attempting a network call, and a safe, deterministic
NOT_APPLICABLE fallback on *any* failure (network, quota, malformed JSON, an
LLM-proposed value that doesn't validate) -- this function never raises.

Every plan -- whether freshly parsed from Gemini or a fallback -- is run
through ``plan_validator.validate_plan`` before being returned. A plan that
fails validation is never returned as-is; it is downgraded to a
NOT_APPLICABLE plan carrying the validator's reasons, so a caller can never
receive something unsafe by relying on this function alone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from google import genai
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.services.qa.json_utils import parse_json_object

from .executor import bound_output
from .models import (
    ExitCodeSemantics,
    ReproductionPlan,
    ReproductionType,
    RepositoryEvidence,
)
from .plan_validator import validate_plan

# Keeps the prompt itself bounded regardless of how much investigation text
# a caller passes in -- reuses Phase 4A's own output-bounding helper rather
# than introducing a second truncation utility.
MAX_FINDINGS_CHARS = 3000


class _RawReproductionPlan(BaseModel):
    """Exact JSON shape the planner asks Gemini to return.

    Deliberately untyped-as-enum here (``reproduction_type``/
    ``exit_code_semantics`` are plain strings) so a malformed/unsupported
    value fails at the *enum construction* step in ``_to_plan`` -- caught by
    the same broad exception handler that already covers network/JSON
    failures, producing one single, deterministic fallback path instead of
    two.
    """

    applicable: bool
    reason: str
    reproduction_type: str
    commands: List[List[str]] = Field(default_factory=list)
    working_dir: Optional[str] = None
    expected_observation: Optional[str] = None
    exit_code_semantics: str = "ignore"
    reproduced_output_pattern: Optional[str] = None
    not_reproduced_output_pattern: Optional[str] = None
    confidence: float = 0.0
    evidence_refs: List[str] = Field(default_factory=list)
    project_root: Optional[str] = None
    ecosystem: Optional[str] = None
    image: Optional[str] = None
    timeout_seconds: Optional[int] = None


_PLANNER_SYSTEM_INSTRUCTION = (
    "You are RepoPilot's reproduction planner. You do not fix bugs, execute "
    "commands, or decide whether a bug exists -- you propose a single, "
    "safe, evidence-backed PROCEDURE that a separate, deterministic system "
    "will later execute and classify. Return ONLY a JSON object, no "
    "markdown fences, no commentary. Schema:\n"
    '{"applicable": boolean, '
    '"reason": string, '
    '"reproduction_type": "command_failure"|"test_failure"|"runtime_behavior"'
    '|"static_observation"|"build_failure"|"api_behavior"|"not_applicable", '
    '"commands": [[string, ...], ...], '
    '"working_dir": string|null, '
    '"expected_observation": string|null, '
    '"exit_code_semantics": "zero_is_reproduced"|"nonzero_is_reproduced"|"ignore", '
    '"reproduced_output_pattern": string|null, '
    '"not_reproduced_output_pattern": string|null, '
    '"confidence": number between 0.0 and 1.0, '
    '"evidence_refs": [string, ...], '
    '"project_root": string|null, '
    '"ecosystem": string|null, '
    '"image": string|null, '
    '"timeout_seconds": number|null}\n'
    "\n"
    "AUTHORITY OF EVIDENCE:\n"
    "The REPOSITORY EVIDENCE section below is authoritative and factual. "
    "The USER REPORT is only a hypothesis about what is wrong -- it may "
    "name the wrong framework, file, or component entirely. If the user's "
    "wording conflicts with the repository evidence (e.g. they say "
    '"React" but the evidence shows Vue, or they say "MongoDB" but the '
    "evidence shows PostgreSQL with no MongoDB reference anywhere), you "
    "MUST follow the evidence, not the user's wording. If the evidence "
    "does not establish that the component the user describes exists at "
    "all, do not invent a reproduction for it.\n"
    "\n"
    "WHEN TO RETURN NOT_APPLICABLE (applicable=false, "
    'reproduction_type="not_applicable", commands=[]):\n'
    "- the repository evidence does not support any specific, executable "
    "procedure for this report\n"
    "- the only way to attempt a procedure would require guessing at "
    "files, endpoints, commands, or framework APIs not shown in the evidence\n"
    "- the report describes something evidence shows does not exist in "
    "this repository\n"
    "Returning NOT_APPLICABLE is the CORRECT, preferred answer whenever a "
    "trustworthy procedure cannot be constructed -- never invent one merely "
    "to avoid it.\n"
    "\n"
    "WHAT MAKES A GOOD PROCEDURE:\n"
    "- Strongly prefer an existing command already listed under KNOWN "
    "COMMANDS below (e.g. an actual package.json script, an existing test "
    "file) over any invented shell logic.\n"
    "- Every command is an argv array (e.g. [\"npm\", \"test\"]), never a "
    "single shell string -- these are executed directly, never via a shell.\n"
    "- reason must explain WHY the chosen exit_code_semantics/pattern "
    "actually demonstrates the reported bug (e.g. \"the existing regression "
    "test at tests/test_cart.py::test_subtotal fails today because of this "
    "bug, so a nonzero exit specifically from that test is the reproduction\"). "
    "Do not assume a generic nonzero exit means a bug exists, or that a "
    "generic zero exit means it doesn't -- justify the specific mapping.\n"
    "- evidence_refs must be file paths (optionally file:start-end) that "
    "literally appear in the DETECTED PROJECTS / KNOWN COMMANDS / "
    "INVESTIGATION FINDINGS / EVIDENCE REFERENCES sections below -- never a "
    "path you have not actually seen there.\n"
    "\n"
    "HARD SAFETY RULES (a plan violating any of these will be rejected "
    "before it can ever run, so do not propose them):\n"
    "- working_dir, if given, must be a repository-relative path -- never "
    "an absolute path, never containing '..'\n"
    "- never propose installing software (pip/npm/yarn/apt/etc install, "
    "go get/install, cargo install, ...)\n"
    "- never propose curl/wget or downloading/fetching arbitrary code\n"
    "- never propose reading credentials, secrets, .env files, or SSH keys\n"
    "- never propose modifying source files, or any git command that "
    "changes history/branches/remotes (push, commit, reset, clean, "
    "checkout, branch, rebase, merge, tag)\n"
    "- never propose deleting files or other destructive commands (rm, "
    "del, format, dd, shutdown, ...)\n"
    "- never invoke a shell interpreter directly (sh, bash, cmd, "
    "powershell, ...) -- each command is one real program plus its "
    "arguments, never a shell one-liner\n"
    "- never assume network access is available\n"
    "- only set \"image\" if a specific Docker image is genuinely needed "
    "and it is one of this project's own known ecosystem images (e.g. "
    "node:20-slim, golang:1.22-alpine) -- leave it null in every other "
    "case and the executor will choose the correct one for the detected "
    "ecosystem automatically\n"
    "- confidence is your own honest confidence (0.0-1.0) that this "
    "specific procedure, if run, would reliably demonstrate the reported "
    "bug -- not confidence that the bug exists at all"
)


def _gemini_unavailable() -> bool:
    """Mirrors the identical test/mock/empty-key check already used by
    app.services.qa.classifier.classify_question and
    app.services.agent.graph's own structured Gemini calls."""
    key = settings.gemini_api_key
    return not key or key.startswith("test") or key.startswith("mock")


def _summarize_project(project: Any) -> str:
    # `project` is a verification.project_analyzer.ProjectInfo, but kept
    # loosely typed here (duck-typed) to avoid a hard import-time coupling
    # in this module beyond what's already implied by RepositoryEvidence.
    root = getattr(project, "root", ".")
    ecosystem = getattr(project, "ecosystem", "unknown")
    languages = ", ".join(getattr(project, "languages", []) or [])
    frameworks = ", ".join(getattr(project, "frameworks", []) or [])
    test_system = getattr(project, "test_system", None)
    evidence_files = ", ".join(getattr(project, "evidence", []) or [])
    return (
        f'- root="{root}" ecosystem="{ecosystem}" languages=[{languages}] '
        f'frameworks=[{frameworks}] test_system="{test_system}" evidence=[{evidence_files}]'
    )


def _summarize_evidence(evidence: RepositoryEvidence) -> str:
    sections: List[str] = []

    if evidence.detected_projects:
        sections.append(
            "DETECTED PROJECTS:\n"
            + "\n".join(_summarize_project(p) for p in evidence.detected_projects)
        )
    else:
        sections.append("DETECTED PROJECTS:\n(none detected)")

    if evidence.known_commands:
        commands_lines = [
            f'- {kc.command} -- {kc.description} (source: {kc.source_file})'
            for kc in evidence.known_commands
        ]
        sections.append(
            "KNOWN COMMANDS (already exist in the repository -- strongly "
            "prefer these over inventing new ones):\n" + "\n".join(commands_lines)
        )
    else:
        sections.append(
            "KNOWN COMMANDS:\n(none known -- do not invent a test/build "
            "command that isn't backed by evidence elsewhere in this prompt)"
        )

    if evidence.investigation_findings:
        sections.append(
            "INVESTIGATION FINDINGS:\n"
            + bound_output(evidence.investigation_findings, MAX_FINDINGS_CHARS)
        )

    if evidence.evidence_references:
        ref_lines = [f"- {ref.citation} -- {ref.description}" for ref in evidence.evidence_references]
        sections.append("EVIDENCE REFERENCES:\n" + "\n".join(ref_lines))

    return "\n\n".join(sections)


def _build_planner_prompt(task_description: str, evidence: RepositoryEvidence) -> str:
    return (
        f"USER REPORT (a hypothesis about what is wrong, not a verified fact):\n"
        f'"{task_description}"\n\n'
        f"{_summarize_evidence(evidence)}\n\n"
        "Propose a reproduction plan now as the JSON object described in "
        "your instructions."
    )


def _not_applicable_plan(reason: str) -> ReproductionPlan:
    """A genuine, confident verdict: there is no evidence-backed executable
    reproduction procedure for this task. Planning itself succeeded (or
    wasn't even necessary, as when no task description was given) --
    ``planning_failed`` stays False, so a caller can trust this as a real
    NOT_APPLICABLE result."""
    return ReproductionPlan(
        applicable=False,
        reason=reason,
        reproduction_type=ReproductionType.NOT_APPLICABLE,
        confidence=0.0,
        planning_failed=False,
    )


def _planning_failed_plan(reason: str) -> ReproductionPlan:
    """The planner could not produce or validate a usable plan -- a Gemini/
    network/API failure, malformed JSON, an unsupported enum value, or an
    LLM-proposed plan that failed safety validation. This is NEVER a
    determination that reproduction is inapplicable: ``applicable=False``/
    ``reproduction_type=NOT_APPLICABLE`` here are structural placeholders
    only (``ReproductionType`` has no other non-executable value to use),
    and ``planning_failed=True`` is the field a caller MUST check first --
    see the ``ReproductionPlan`` docstring. A future caller (Phase 4B-2)
    should map this to something like ``BaselineStatus.UNABLE_TO_REPRODUCE``.
    """
    return ReproductionPlan(
        applicable=False,
        reason=reason,
        reproduction_type=ReproductionType.NOT_APPLICABLE,
        confidence=0.0,
        planning_failed=True,
        failure_reason=reason,
    )


def _to_plan(data: _RawReproductionPlan) -> ReproductionPlan:
    """Convert the raw, Gemini-parsed shape into a ``ReproductionPlan``.

    Raises (ValueError, via the enum constructors) for an unsupported
    ``reproduction_type``/``exit_code_semantics`` string -- the caller
    catches this the same way it catches every other parsing failure.
    """
    return ReproductionPlan(
        applicable=data.applicable,
        reason=data.reason,
        reproduction_type=ReproductionType(data.reproduction_type),
        commands=data.commands,
        working_dir=data.working_dir,
        expected_observation=data.expected_observation,
        exit_code_semantics=ExitCodeSemantics(data.exit_code_semantics),
        reproduced_output_pattern=data.reproduced_output_pattern,
        not_reproduced_output_pattern=data.not_reproduced_output_pattern,
        confidence=data.confidence,
        evidence_refs=data.evidence_refs,
        project_root=data.project_root,
        ecosystem=data.ecosystem,
        image=data.image,
        timeout_seconds=data.timeout_seconds,
    )


async def plan_reproduction(
    task_description: str,
    evidence: RepositoryEvidence,
) -> ReproductionPlan:
    """Propose a ``ReproductionPlan`` for ``task_description`` given
    ``evidence``. Never executes anything, never raises, and never returns
    a plan that failed ``plan_validator.validate_plan``.
    """
    if not task_description or not task_description.strip():
        return _not_applicable_plan("No task/bug description was supplied.")

    if _gemini_unavailable():
        return _planning_failed_plan("Gemini not configured (missing/test/mock API key).")

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=_build_planner_prompt(task_description, evidence),
            config={"system_instruction": _PLANNER_SYSTEM_INSTRUCTION},
        )
        raw_text = response.text or ""
        data: Dict[str, Any] = parse_json_object(raw_text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

        raw_plan = _RawReproductionPlan(**data)
        plan = _to_plan(raw_plan)
    except Exception as e:  # noqa: BLE001 -- any failure (network, quota, malformed JSON, invalid schema/enum) is a planning failure, never a NOT_APPLICABLE verdict
        logger.warning(f"Reproduction planning failed (infrastructure/parsing failure, not NOT_APPLICABLE): {e}")
        return _planning_failed_plan(f"Reproduction planning failed: {e}")

    result = validate_plan(plan, evidence)
    if not result.valid:
        logger.warning(f"Reproduction plan failed validation (planning failure, not NOT_APPLICABLE): {result.errors}")
        return _planning_failed_plan(
            "The proposed reproduction plan failed safety validation and was "
            "rejected: " + "; ".join(result.errors)
        )

    return plan
