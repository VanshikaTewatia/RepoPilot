"""Evidence-grounded patch planning synthesis (Phase 6C).

Turns an already-established diagnosis (Phase 6A) plus already-gathered
``retrieved_context`` (Phase 6B) into a structured ``PatchPlan`` via a
single Gemini call. Mirrors the exact Gemini-call/test-mode/deterministic-
override conventions already established by ``app.services.diagnosis.
diagnoser`` -- see this module's functions below for the direct analogues.

Never fabricates:
  - Diagnosis not DIAGNOSED (absent, INSUFFICIENT_EVIDENCE, DIAGNOSIS_FAILED)
    -> INSUFFICIENT_DIAGNOSIS returned immediately, WITHOUT ever calling
    Gemini. This is the core no-fabricated-fix protection: planning never
    even attempts to reason about a fix when there is nothing solid to
    reason FROM.
  - No retrieved_context -> INSUFFICIENT_DIAGNOSIS, same rule, independent
    of diagnosis status (redundant-but-cheap extra safety net).
  - Every citation Gemini returns is checked against the real retrieved
    file paths; anything else is dropped (see patch_plan_validator).
  - Any Gemini/network/JSON failure produces PLANNING_FAILED with a
    failure_reason -- never silently reinterpreted as NOT_APPLICABLE or
    INSUFFICIENT_DIAGNOSIS (a process failure is not a verdict).

Deliberately has NO dependency on reproduction, Docker, workspace
creation, or remote code-hosting functionality, and NEVER modifies files
-- a PatchPlan is guidance for a later Gemini patch-generation call, never
itself a diff, never passed to edit_node/apply_patch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from google import genai

from app.core.config import settings
from app.core.logging import logger
from app.services.diagnosis.models import DiagnosisStatus
from app.services.qa.json_utils import parse_json_object
from app.services.qa.models import CitationRef

from .models import PatchPlan, PatchPlanStatus, PlannedChange
from .patch_plan_validator import validate_patch_plan

_PATCH_PLAN_SYSTEM_INSTRUCTION = (
    "You are RepoPilot, an expert AI software engineer. You are given a "
    "root-cause diagnosis of a reported bug and the actual retrieved code. "
    "Produce a structured PATCH PLAN describing what should change and why "
    "-- NOT the actual code diff. Follow these rules strictly:\n"
    "1. The supplied Context Code Files are authoritative.\n"
    "2. Base your plan on the supplied diagnosis. Do not invent an "
    "unrelated cause or propose changes the diagnosis does not support.\n"
    "3. Only propose changes to files that are present in the supplied "
    "context, or clearly justified new files.\n"
    "4. Each change must include a rationale explicitly tying it back to "
    "the diagnosed cause.\n"
    "5. If the diagnosis indicates the behavior is already correct, or no "
    'code change is warranted, set "applicable" to false and return an '
    'empty "changes" array.\n'
    "6. Do not propose unrelated refactors, cleanup, or broad rewrites.\n"
    '7. change_type must be "modify" or "create" only.\n'
    "\n"
    "Return ONLY a JSON object, no markdown fences, no commentary. Schema:\n"
    "{\n"
    '  "applicable": true|false,\n'
    '  "summary": "1-3 sentence overview",\n'
    '  "changes": [{"file_path": "...", "change_type": "modify"|"create", '
    '"description": "...", "rationale": "...", '
    '"citations": [{"file_path": "...", "start_line": 1, "end_line": 10, '
    '"symbol_name": "..." or null}], "symbols_affected": ["..."]}],\n'
    '  "diagnosis_alignment": "how this plan addresses the diagnosed cause",\n'
    '  "confidence": "direct_evidence" | "inferred" | "no_evidence"\n'
    "}"
)


def _gemini_unavailable() -> bool:
    key = settings.gemini_api_key
    return not key or key.startswith("test") or key.startswith("mock")


def insufficient_diagnosis_plan(reason: str) -> PatchPlan:
    """The deterministic result whenever there is nothing solid to plan
    from -- never produced by asking Gemini to plan from a diagnosis that
    was never established (mirrors diagnoser.insufficient_evidence_diagnosis
    one level up)."""
    return PatchPlan(status=PatchPlanStatus.INSUFFICIENT_DIAGNOSIS, confidence="no_evidence", failure_reason=reason)


def _real_evidence_files(retrieved_context: List[Dict[str, Any]]) -> Set[str]:
    return {item.get("file_path") for item in retrieved_context if item.get("file_path")}


def _diagnosis_citation_files(diagnosis: Dict[str, Any]) -> Set[str]:
    files: Set[str] = set()
    for h in diagnosis.get("hypotheses") or []:
        for c in h.get("citations") or []:
            fpath = c.get("file_path")
            if fpath:
                files.add(fpath)
    return files


def _format_diagnosis_context(diagnosis: Dict[str, Any]) -> str:
    summary = diagnosis.get("summary") or ""
    hypotheses = diagnosis.get("hypotheses") or []
    lines: List[str] = []
    if summary:
        lines.append(f"Diagnosis summary: {summary}")
    for h in hypotheses:
        citations = h.get("citations") or []
        citation_str = ", ".join(
            f"{c.get('file_path')}:{c.get('start_line')}-{c.get('end_line')}" for c in citations
        )
        line = f"- Hypothesis {h.get('rank')}: {h.get('description', '')}"
        if citation_str:
            line += f" (evidence: {citation_str})"
        lines.append(line)
    return "\n".join(lines)


def _format_retrieved_context(retrieved_context: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in retrieved_context:
        fpath = item.get("file_path", "")
        content = item.get("content", "")
        total_lines = item.get("total_lines", 0)
        parts.append(f"### File: {fpath} ({total_lines} lines total)\n```\n{content}\n```")
    return "\n\n".join(parts)


def _build_patch_plan_prompt(
    task_description: str,
    retrieved_context: List[Dict[str, Any]],
    diagnosis: Dict[str, Any],
    error_analysis: Optional[str],
) -> str:
    prompt = (
        f"Task Description:\n{task_description}\n\n"
        f"Root Cause Diagnosis:\n{_format_diagnosis_context(diagnosis)}\n\n"
        f"Context Code Files:\n{_format_retrieved_context(retrieved_context)}\n\n"
    )
    if error_analysis:
        prompt += f"Previous Attempt Test Failure Trace:\n{error_analysis}\n\n"
    prompt += "Produce the JSON patch plan now:"
    return prompt


def _parse_citation(data: dict) -> CitationRef:
    return CitationRef(
        file_path=str(data.get("file_path", "")),
        start_line=int(data.get("start_line", 1)),
        end_line=int(data.get("end_line", data.get("start_line", 1))),
        symbol_name=data.get("symbol_name"),
    )


def _patch_plan_from_data(data: dict) -> PatchPlan:
    applicable = bool(data.get("applicable", True))
    confidence = data.get("confidence")
    if confidence not in ("direct_evidence", "inferred", "no_evidence"):
        confidence = "inferred"

    if not applicable:
        return PatchPlan(
            status=PatchPlanStatus.NOT_APPLICABLE,
            summary=str(data.get("summary") or ""),
            confidence=confidence,
        )

    changes: List[PlannedChange] = []
    for c in data.get("changes") or []:
        change_type = c.get("change_type")
        changes.append(
            PlannedChange(
                file_path=str(c.get("file_path", "")),
                change_type=change_type if change_type in ("modify", "create") else "modify",
                description=str(c.get("description", "")),
                rationale=str(c.get("rationale", "")),
                citations=[_parse_citation(ci) for ci in (c.get("citations") or [])],
                symbols_affected=[str(s) for s in (c.get("symbols_affected") or [])],
            )
        )

    return PatchPlan(
        status=PatchPlanStatus.PLANNED,
        summary=str(data.get("summary") or ""),
        changes=changes,
        diagnosis_alignment=data.get("diagnosis_alignment"),
        confidence=confidence,
    )


def _mock_patch_plan(diagnosis: Dict[str, Any]) -> PatchPlan:
    """Deterministic plan used in test/dev environments (no real Gemini key
    configured) -- grounded in the diagnosis's own top hypothesis/citation,
    mirroring diagnoser._mock_diagnosis's convention of a deterministic
    placeholder rather than a network call, never inventing new evidence."""
    hypotheses = diagnosis.get("hypotheses") or []
    if not hypotheses:
        return PatchPlan(status=PatchPlanStatus.NOT_APPLICABLE, summary="No hypotheses to plan from.", confidence="no_evidence")

    top = hypotheses[0]
    citations_data = top.get("citations") or []
    if not citations_data:
        return PatchPlan(status=PatchPlanStatus.NOT_APPLICABLE, summary="No cited file to plan from.", confidence="no_evidence")

    citation = _parse_citation(citations_data[0])
    change = PlannedChange(
        file_path=citation.file_path,
        change_type="modify",
        description=f"Address: {top.get('description', '')}",
        rationale=top.get("description", ""),
        citations=[citation],
    )
    return PatchPlan(
        status=PatchPlanStatus.PLANNED,
        summary="Based on the diagnosed root cause, here is a candidate patch plan.",
        changes=[change],
        diagnosis_alignment=f"Addresses hypothesis: {top.get('description', '')}",
        confidence="inferred",
    )


async def plan_patches(
    task_description: str,
    diagnosis: Optional[Dict[str, Any]],
    retrieved_context: List[Dict[str, Any]],
    error_analysis: Optional[str] = None,
) -> PatchPlan:
    """Synthesize a structured, cited ``PatchPlan`` from an already-DIAGNOSED
    root cause and already-gathered ``retrieved_context``.

    Never calls Gemini unless diagnosis.status == DIAGNOSED AND
    retrieved_context is non-empty (returns insufficient_diagnosis_plan()
    immediately otherwise) -- this is the core no-fabricated-fix guarantee.
    Any Gemini/parsing failure returns PLANNING_FAILED with failure_reason
    set, never raised. The result is always passed through
    validate_patch_plan before being returned.
    """
    diagnosis = diagnosis or {}
    diagnosis_status = diagnosis.get("status")
    if diagnosis_status != DiagnosisStatus.DIAGNOSED.value:
        reason = f"diagnosis status is {diagnosis_status!r}" if diagnosis_status else "no diagnosis is available"
        return insufficient_diagnosis_plan(f"Patch planning requires a DIAGNOSED root cause; {reason}.")

    if not retrieved_context:
        return insufficient_diagnosis_plan("No retrieved evidence to plan changes from.")

    real_files = _real_evidence_files(retrieved_context)
    diagnosis_citation_files = _diagnosis_citation_files(diagnosis)

    if _gemini_unavailable():
        raw = _mock_patch_plan(diagnosis)
    else:
        prompt = _build_patch_plan_prompt(task_description, retrieved_context, diagnosis, error_analysis)
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model_name,
                contents=prompt,
                config={"system_instruction": _PATCH_PLAN_SYSTEM_INSTRUCTION},
            )
            raw_text = response.text or ""
            data = parse_json_object(raw_text)
            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
            raw = _patch_plan_from_data(data)
        except Exception as e:  # noqa: BLE001 -- any failure degrades to PLANNING_FAILED, never crashes
            logger.error(f"Error generating patch plan with Gemini: {e}")
            return PatchPlan(status=PatchPlanStatus.PLANNING_FAILED, confidence="no_evidence", failure_reason=str(e))

    result = validate_patch_plan(raw, real_files, diagnosis_citation_files)
    return result.plan
