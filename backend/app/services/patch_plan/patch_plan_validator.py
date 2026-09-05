"""Deterministic validation/filtering for a raw ``PatchPlan`` parsed from
Gemini output.

Mirrors ``app.services.diagnosis.diagnosis_validator``'s own
deterministic-override philosophy (filter/repair, not reject-wholesale):
a ``PatchPlan`` is advisory-only and never executes anything, so a
partially-untrustworthy plan is repaired in place instead of discarded --
except when EVERY proposed change is rejected, which downgrades the whole
plan to PLANNING_FAILED (see ``validate_patch_plan``) rather than leaving
an incoherent "PLANNED with nothing to do" plan.

The LLM's structured output is never trusted directly: every citation is
checked against the real evidence file set actually retrieved, every
change's file_path is checked the same way ``app.services.agent.graph.
validate_patch`` already checks a real generated patch's path (relative,
non-traversing), every change_type is checked against a fixed allow-list,
change counts/text lengths are bounded, and confidence is sanity-checked
against how much real evidence survived filtering.
"""

from __future__ import annotations

import re
from typing import List, Set

from app.services.baseline.executor import bound_output
from app.services.qa.models import CitationRef

from .models import PatchPlan, PatchPlanStatus, PatchPlanValidationResult, PlannedChange

MAX_CHANGES = 5
MAX_CITATIONS_PER_CHANGE = 10
MAX_SYMBOLS_PER_CHANGE = 10
MAX_SUMMARY_CHARS = 2000
MAX_DESCRIPTION_CHARS = 2000
MAX_RATIONALE_CHARS = 2000
MAX_DIAGNOSIS_ALIGNMENT_CHARS = 2000

# The only change_type values a PlannedChange may carry -- see
# models.PlannedChange's own docstring: this is the schema's actual safety
# boundary (no delete/execute/run representable at all), this allow-list is
# defense-in-depth against a value outside even the schema's own Literal.
_ALLOWED_CHANGE_TYPES = {"modify", "create"}

# Identical lexical check to app.services.agent.graph.validate_patch's own
# path-safety rule (absolute or traversing paths rejected) and to
# app.services.baseline.plan_validator's _ABSOLUTE_PATH_PATTERN -- deliberately
# duplicated (not imported) as a tiny, pure, sibling-package check rather
# than adding a cross-package dependency for a three-line rule. This is
# defense-in-depth, not the actual safety boundary: a PatchPlan is never
# passed to apply_patch, so a bad path here can only ever become prompt
# text a later Gemini call must still pass through the UNCHANGED
# validate_patch/parse_and_validate_patches gate before any file is touched.
_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:[a-zA-Z]:[\\/]|[\\/])")


def _file_path_is_unsafe(file_path: str) -> bool:
    if not file_path or not isinstance(file_path, str):
        return True
    if _ABSOLUTE_PATH_PATTERN.match(file_path):
        return True
    clean = file_path.strip().replace("\\", "/")
    if clean.startswith("/") or ".." in clean.split("/"):
        return True
    return False


def _filter_hallucinated_citations(citations: List[CitationRef], real_files: Set[str]) -> List[CitationRef]:
    """Drop any citation for a file that was never actually gathered as
    evidence -- identical policy to diagnosis_validator's own filter."""
    return [c for c in citations if c.file_path in real_files]


def _sanity_check_confidence(confidence: str, total_evidence_pieces: int) -> str:
    """Never let a "direct_evidence" verdict stand on objectively thin
    evidence -- same rule as diagnosis_validator's own."""
    if confidence != "direct_evidence":
        return confidence
    return "inferred" if total_evidence_pieces <= 1 else confidence


def validate_patch_plan(
    plan: PatchPlan, real_files: Set[str], diagnosis_citation_files: Set[str]
) -> PatchPlanValidationResult:
    """Deterministically filter/repair a raw ``PatchPlan``.

    ``real_files`` is the set of file paths actually present in
    retrieved_context -- used to drop hallucinated citations. Note a
    change's OWN file_path is not required to be in ``real_files`` (a
    change may legitimately propose creating a new file not yet
    retrieved); only its citations are filtered against it.
    ``diagnosis_citation_files`` is the set of file paths the diagnosis
    itself cited -- used only for a soft, non-rejecting alignment signal.

    Never rejects a non-PLANNED plan (nothing to filter). A PLANNED plan
    that ends up with zero surviving changes after filtering is downgraded
    to PLANNING_FAILED -- a PLANNED status with no actual changes would be
    an applicable<->content inconsistency, never returned as-is.
    """
    notes: List[str] = []

    if plan.status != PatchPlanStatus.PLANNED:
        return PatchPlanValidationResult(plan=plan, notes=notes)

    changes = plan.changes[:MAX_CHANGES]
    if len(plan.changes) > MAX_CHANGES:
        notes.append(f"truncated changes to {MAX_CHANGES} (had {len(plan.changes)})")

    filtered_changes: List[PlannedChange] = []
    for change in changes:
        if change.change_type not in _ALLOWED_CHANGE_TYPES:
            notes.append(f"dropped change with unsupported change_type {change.change_type!r}")
            continue
        if _file_path_is_unsafe(change.file_path):
            notes.append(f"dropped change with unsafe file_path {change.file_path!r}")
            continue

        citations = _filter_hallucinated_citations(change.citations, real_files)
        if len(citations) != len(change.citations):
            notes.append(f"dropped {len(change.citations) - len(citations)} hallucinated citation(s) from a change")
        citations = citations[:MAX_CITATIONS_PER_CHANGE]

        filtered_changes.append(
            PlannedChange(
                file_path=change.file_path,
                change_type=change.change_type,
                description=bound_output(change.description, MAX_DESCRIPTION_CHARS),
                rationale=bound_output(change.rationale, MAX_RATIONALE_CHARS),
                citations=citations,
                symbols_affected=list(change.symbols_affected)[:MAX_SYMBOLS_PER_CHANGE],
            )
        )

    if not filtered_changes:
        notes.append("all proposed changes were rejected by validation; downgrading to PLANNING_FAILED")
        failed_plan = PatchPlan(
            status=PatchPlanStatus.PLANNING_FAILED,
            confidence="no_evidence",
            failure_reason="All proposed changes were rejected by deterministic validation.",
        )
        return PatchPlanValidationResult(plan=failed_plan, notes=notes)

    total_citations = sum(len(c.citations) for c in filtered_changes)
    confidence = _sanity_check_confidence(plan.confidence, total_citations)
    if confidence != plan.confidence:
        notes.append(f"downgraded confidence from {plan.confidence!r} to {confidence!r} (thin evidence)")

    # Diagnosis-alignment heuristic: a soft signal, never a rejection -- a
    # legitimate fix may touch a caller file the diagnosis never explicitly
    # cited, so zero overlap only downgrades confidence one notch rather
    # than failing the plan.
    if diagnosis_citation_files:
        changed_files = {c.file_path for c in filtered_changes}
        if not (changed_files & diagnosis_citation_files):
            notes.append("no proposed change's file_path overlaps with any diagnosis citation -- alignment unconfirmed")
            if confidence == "direct_evidence":
                confidence = "inferred"
                notes.append("downgraded confidence due to unconfirmed diagnosis alignment")

    validated_plan = PatchPlan(
        status=plan.status,
        summary=bound_output(plan.summary, MAX_SUMMARY_CHARS),
        changes=filtered_changes,
        diagnosis_alignment=(
            bound_output(plan.diagnosis_alignment, MAX_DIAGNOSIS_ALIGNMENT_CHARS) if plan.diagnosis_alignment else None
        ),
        confidence=confidence,
        failure_reason=plan.failure_reason,
    )
    return PatchPlanValidationResult(plan=validated_plan, notes=notes)
