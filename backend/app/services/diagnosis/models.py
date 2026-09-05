"""Typed models for Phase 6A evidence-driven root-cause diagnosis.

Diagnosis sits between retrieval and patch planning: it turns the
already-gathered ``retrieved_context`` (and, on a retry, the current
``error_analysis``) into a structured, cited hypothesis about *why* the
reported bug is happening, before the planner asks Gemini to generate a
patch. It is advisory only -- see the package docstring in
``app.services.diagnosis.__init__`` for the invariants this implies.

Reuses ``CitationRef``/``ConfidenceLevel`` directly from
``app.services.qa.models`` rather than duplicating them: a diagnosis
citation is the same "file/line/symbol provenance" concept a QA answer's
evidence already is.

``diagnosis_failed``/``failure_reason`` mirrors the exact pattern already
established by ``QuestionClass.classification_failed`` (QA classifier) and
``ReproductionPlan.planning_failed`` (Phase 4B-1): a process/infrastructure
failure (a Gemini error, malformed JSON, an unsupported enum value) must
never be silently reinterpreted as a genuine diagnostic verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from app.services.qa.models import CitationRef, ConfidenceLevel


class DiagnosisStatus(str, Enum):
    """Outcome of attempting to diagnose a reported bug's root cause.

    Every value here is evidence-gathering output only -- never a verdict
    about the task's final outcome, and never allowed to gate, block, or
    influence ``finalize_node``/``should_continue`` (see the package
    docstring). ``plan_node`` may use a ``DIAGNOSED`` result to inform its
    Gemini prompt; ``INSUFFICIENT_EVIDENCE`` and ``DIAGNOSIS_FAILED`` must
    both fall back to the exact pre-Phase-6A prompt.

    - DIAGNOSED: a structured hypothesis was produced from real evidence.
    - INSUFFICIENT_EVIDENCE: diagnosis ran (no infrastructure failure) but
      there was no usable evidence to reason from -- a genuine, deterministic
      result, never sent to Gemini (mirrors ``qa.answerer.no_evidence_answer``).
    - DIAGNOSIS_FAILED: the diagnosis process itself did not complete (a
      Gemini/network error, malformed JSON, an unsupported enum value). This
      is NOT a claim that no root cause exists -- it must never be treated as
      INSUFFICIENT_EVIDENCE or as any kind of negative signal.
    """

    DIAGNOSED = "DIAGNOSED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DIAGNOSIS_FAILED = "DIAGNOSIS_FAILED"


@dataclass
class RootCauseHypothesis:
    """One candidate explanation for the reported bug, grounded in cited
    evidence. Multiple hypotheses may be returned when the evidence
    doesn't clearly point to a single cause; ``rank`` orders them from
    most to least likely (1 = most likely)."""

    rank: int
    description: str
    citations: List[CitationRef] = field(default_factory=list)
    suggested_fix_approach: Optional[str] = None


@dataclass
class Diagnosis:
    """A structured, cited root-cause diagnosis for the reported bug.

    This is a PROPOSAL only: advisory input to ``plan_node``'s Gemini patch
    prompt, never itself a patch, never executed, and never allowed to
    affect verification, reproduction, or the task's final outcome.

    ``status``/``failure_reason`` follow the ``DiagnosisStatus`` semantics
    above. ``hypotheses``/``summary`` are only meaningful when
    ``status == DIAGNOSED``; callers must always check ``status`` first,
    never infer it from whether ``hypotheses`` is empty.
    """

    status: DiagnosisStatus
    summary: str = ""
    hypotheses: List[RootCauseHypothesis] = field(default_factory=list)
    confidence: ConfidenceLevel = "no_evidence"
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a plain, JSON-safe dict -- the shape stored in
        ``AgentState["diagnosis"]``. Text fields are expected to already be
        bounded by the caller (see ``diagnoser.bound_output`` reuse)."""
        return {
            "status": self.status.value,
            "summary": self.summary,
            "hypotheses": [
                {
                    "rank": h.rank,
                    "description": h.description,
                    "citations": [
                        {
                            "file_path": c.file_path,
                            "start_line": c.start_line,
                            "end_line": c.end_line,
                            "symbol_name": c.symbol_name,
                        }
                        for c in h.citations
                    ],
                    "suggested_fix_approach": h.suggested_fix_approach,
                }
                for h in self.hypotheses
            ],
            "confidence": self.confidence,
            "failure_reason": self.failure_reason,
        }


@dataclass
class DiagnosisValidationResult:
    """Outcome of deterministically validating/filtering a raw ``Diagnosis``
    parsed from Gemini output.

    Unlike ``PlanValidationResult`` (Phase 4B-1), this is filter-based, not
    reject-wholesale: diagnosis is advisory-only and low-stakes, so a
    partially-invalid result is repaired in place (hallucinated citations
    dropped, confidence downgraded when thin) rather than discarded
    entirely. ``diagnosis`` is always populated; ``notes`` records what, if
    anything, was filtered or adjusted."""

    diagnosis: Diagnosis
    notes: List[str] = field(default_factory=list)
