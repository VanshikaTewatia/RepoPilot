"""Deterministic validation/filtering for a raw ``Diagnosis`` parsed from
Gemini output.

Mirrors ``app.services.qa.answerer``'s own deterministic-override
philosophy (``_filter_hallucinated_citations``, ``_sanity_check_confidence``)
rather than ``app.services.baseline.plan_validator``'s reject-wholesale one:
diagnosis is advisory-only and never executes anything, so a partially
untrustworthy result is repaired in place instead of discarded -- see
``DiagnosisValidationResult``'s docstring for why.

The LLM's structured output is never trusted directly: every citation is
checked against the real evidence file set actually gathered, hypothesis
counts/text lengths are bounded, and confidence is sanity-checked against
how much real evidence was gathered.
"""

from __future__ import annotations

from typing import List, Set

from app.services.baseline.executor import bound_output
from app.services.qa.models import CitationRef

from .models import Diagnosis, DiagnosisStatus, DiagnosisValidationResult, RootCauseHypothesis

MAX_HYPOTHESES = 5
MAX_CITATIONS_PER_HYPOTHESIS = 10
MAX_SUMMARY_CHARS = 2000
MAX_DESCRIPTION_CHARS = 2000
MAX_FIX_APPROACH_CHARS = 2000


def _filter_hallucinated_citations(
    citations: List[CitationRef], real_files: Set[str]
) -> List[CitationRef]:
    """Drop any citation for a file that was never actually gathered as
    evidence -- identical policy to ``qa.answerer._filter_hallucinated_citations``,
    duplicated here (rather than imported) because it is a tiny, pure
    function with no shared state, and importing across sibling service
    packages for a three-line filter would add a coupling neither package
    needs."""
    return [c for c in citations if c.file_path in real_files]


def _sanity_check_confidence(confidence: str, total_evidence_pieces: int) -> str:
    """Never let a "direct_evidence" verdict stand on objectively thin
    evidence -- same rule as ``qa.answerer._sanity_check_confidence``."""
    if confidence != "direct_evidence":
        return confidence
    return "inferred" if total_evidence_pieces <= 1 else confidence


def validate_diagnosis(diagnosis: Diagnosis, real_files: Set[str], total_evidence_pieces: int) -> DiagnosisValidationResult:
    """Deterministically filter/repair a raw ``Diagnosis``.

    ``real_files`` is the set of file paths actually present in the
    evidence diagnosis was run against (retrieved_context file paths) --
    used to drop hallucinated citations. ``total_evidence_pieces`` is the
    count of real evidence items gathered, used for the confidence sanity
    check.

    Never rejects a diagnosis outright: ``status``/``failure_reason`` are
    passed through unchanged (a DIAGNOSIS_FAILED/INSUFFICIENT_EVIDENCE
    diagnosis has nothing to filter), and a DIAGNOSED diagnosis is repaired
    in place rather than discarded.
    """
    notes: List[str] = []

    if diagnosis.status != DiagnosisStatus.DIAGNOSED:
        return DiagnosisValidationResult(diagnosis=diagnosis, notes=notes)

    hypotheses = diagnosis.hypotheses[:MAX_HYPOTHESES]
    if len(diagnosis.hypotheses) > MAX_HYPOTHESES:
        notes.append(f"truncated hypotheses to {MAX_HYPOTHESES} (had {len(diagnosis.hypotheses)})")

    filtered_hypotheses: List[RootCauseHypothesis] = []
    for h in hypotheses:
        citations = _filter_hallucinated_citations(h.citations, real_files)
        if len(citations) != len(h.citations):
            notes.append(f"dropped {len(h.citations) - len(citations)} hallucinated citation(s) from a hypothesis")
        citations = citations[:MAX_CITATIONS_PER_HYPOTHESIS]

        filtered_hypotheses.append(
            RootCauseHypothesis(
                rank=h.rank,
                description=bound_output(h.description, MAX_DESCRIPTION_CHARS),
                citations=citations,
                suggested_fix_approach=(
                    bound_output(h.suggested_fix_approach, MAX_FIX_APPROACH_CHARS)
                    if h.suggested_fix_approach
                    else None
                ),
            )
        )

    confidence = _sanity_check_confidence(diagnosis.confidence, total_evidence_pieces)
    if confidence != diagnosis.confidence:
        notes.append(f"downgraded confidence from {diagnosis.confidence!r} to {confidence!r} (thin evidence)")

    validated = Diagnosis(
        status=diagnosis.status,
        summary=bound_output(diagnosis.summary, MAX_SUMMARY_CHARS),
        hypotheses=filtered_hypotheses,
        confidence=confidence,
        failure_reason=diagnosis.failure_reason,
    )
    return DiagnosisValidationResult(diagnosis=validated, notes=notes)
