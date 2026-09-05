"""Typed models for Phase 6C validated patch planning.

Sits between diagnosis (Phase 6A, "what is wrong and why") and Gemini patch
generation ("the actual diff"): a ``PatchPlan`` describes WHICH files should
change and WHY, grounded in the diagnosed root cause and the actual
retrieved evidence -- never itself a diff, and never executed. See the
package docstring in ``app.services.patch_plan.__init__`` for the
integration invariants this implies.

Reuses ``CitationRef``/``ConfidenceLevel`` directly from
``app.services.qa.models`` rather than duplicating them -- a planned
change's evidence is the same "file/line/symbol provenance" concept a
diagnosis hypothesis's evidence already is.

``PatchPlanStatus`` deliberately mirrors ``DiagnosisStatus``'s single-enum
design (the more recent, refined precedent in this codebase) rather than
``ReproductionPlan``'s older dual-bool (``applicable`` + ``planning_failed``)
one: a single status avoids the "check two fields, in the right order"
burden the older pattern required. ``INSUFFICIENT_DIAGNOSIS`` and
``PLANNING_FAILED`` are process/upstream failures, never a verdict about
whether a fix is needed; ``NOT_APPLICABLE`` is a genuine, evidence-grounded
conclusion that no code change is warranted. Critically, the graph
integration (see ``app.services.agent.graph.plan_node``) treats ONLY
``PLANNED`` as license to call Gemini for patch generation -- an allow-list,
not a denylist, so no status (existing or future) can accidentally fall
through to patch generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional

from app.services.qa.models import CitationRef, ConfidenceLevel

ChangeType = Literal["modify", "create"]


class PatchPlanStatus(str, Enum):
    """Outcome of attempting to plan a patch from a diagnosed root cause.

    - PLANNED: a real, validated plan is ready to guide patch generation.
      The ONLY status that may result in a Gemini patch-generation call --
      see app.services.agent.graph.plan_node's allow-list gate.
    - NOT_APPLICABLE: diagnosis was usable, but planning concluded, from
      real evidence, that no code change is warranted. Distinct in MEANING
      from INSUFFICIENT_DIAGNOSIS/PLANNING_FAILED (this is a confident
      conclusion, not a failure) but identical in EFFECT: no Gemini call.
    - INSUFFICIENT_DIAGNOSIS: diagnosis itself never established a usable
      root cause (DIAGNOSIS_FAILED or INSUFFICIENT_EVIDENCE), or there was
      no retrieved evidence to plan from -- planning correctly declines
      rather than guessing, and NEVER calls Gemini for planning.
    - PLANNING_FAILED: the planning process itself failed (a Gemini error,
      malformed JSON, or a proposed plan that failed validation entirely).
      A process failure, never a verdict.
    """

    PLANNED = "PLANNED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_DIAGNOSIS = "INSUFFICIENT_DIAGNOSIS"
    PLANNING_FAILED = "PLANNING_FAILED"


@dataclass
class PlannedChange:
    """One proposed file-level change, grounded in cited evidence.

    ``change_type`` is deliberately restricted to "modify"/"create" --
    there is no field capable of encoding a delete, a shell command, a Git
    operation, or a dependency install anywhere in this model. This is the
    schema's own safety boundary: destructive/executable actions are
    impossible to represent, not merely discouraged.
    """

    file_path: str
    change_type: ChangeType
    description: str
    rationale: str
    citations: List[CitationRef] = field(default_factory=list)
    symbols_affected: List[str] = field(default_factory=list)


@dataclass
class PatchPlan:
    """A structured, cited plan for what should change and why -- NEVER a
    diff, NEVER executed, and NEVER passed to edit_node/apply_patch.

    ``changes`` is only meaningful when ``status == PLANNED``; callers must
    always check ``status`` first, never infer it from whether ``changes``
    is empty (mirrors ``Diagnosis.hypotheses``'s identical rule).
    """

    status: PatchPlanStatus
    summary: str = ""
    changes: List[PlannedChange] = field(default_factory=list)
    diagnosis_alignment: Optional[str] = None
    confidence: ConfidenceLevel = "no_evidence"
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a plain, JSON-safe dict -- the shape stored in
        ``AgentState["patch_plan"]``. Never contains raw file content or a
        copy of retrieved_context -- citations are file/line pointers only."""
        return {
            "status": self.status.value,
            "summary": self.summary,
            "changes": [
                {
                    "file_path": c.file_path,
                    "change_type": c.change_type,
                    "description": c.description,
                    "rationale": c.rationale,
                    "citations": [
                        {
                            "file_path": ci.file_path,
                            "start_line": ci.start_line,
                            "end_line": ci.end_line,
                            "symbol_name": ci.symbol_name,
                        }
                        for ci in c.citations
                    ],
                    "symbols_affected": c.symbols_affected,
                }
                for c in self.changes
            ],
            "diagnosis_alignment": self.diagnosis_alignment,
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
        }


@dataclass
class PatchPlanValidationResult:
    """Outcome of deterministically validating/filtering a raw ``PatchPlan``
    parsed from Gemini output.

    Filter-based, not reject-wholesale (mirrors ``DiagnosisValidationResult``,
    not ``PlanValidationResult``): a ``PatchPlan`` never executes anything,
    so a partially-invalid plan is repaired in place (hallucinated
    citations dropped, unsafe changes dropped, confidence downgraded when
    thin) rather than discarded entirely -- UNLESS every change is rejected,
    in which case the plan is downgraded to PLANNING_FAILED rather than
    left as an incoherent "PLANNED with zero changes" (see
    patch_plan_validator.validate_patch_plan)."""

    plan: PatchPlan
    notes: List[str] = field(default_factory=list)
