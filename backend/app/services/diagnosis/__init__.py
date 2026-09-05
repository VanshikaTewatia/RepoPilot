"""Phase 6A: evidence-driven root-cause diagnosis.

Sits between retrieval and patch planning in the agent graph (see
``app.services.agent.graph.diagnose_node``): turns already-gathered
``retrieved_context`` into a structured, cited ``Diagnosis`` that
``plan_node`` may use to inform its Gemini patch prompt.

Advisory only, by design:
  - Diagnosis never calls reproduction, Docker, workspace creation, or
    GitHub functionality -- it works purely from state already gathered by
    ``retrieve_node``/``analyze_failure_node``.
  - Diagnosis never blocks, gates, or otherwise influences ``finalize_node``
    or ``should_continue``. A ``DIAGNOSIS_FAILED`` or
    ``INSUFFICIENT_EVIDENCE`` result must never prevent, delay, or alter
    patch generation -- ``plan_node`` falls back to the exact
    pre-Phase-6A prompt whenever a valid ``DIAGNOSED`` result isn't
    available.
  - Like ``ReproductionPlan.planning_failed`` (Phase 4B-1) and
    ``QuestionClass.classification_failed`` (QA classifier),
    ``DiagnosisStatus.DIAGNOSIS_FAILED`` is a process failure, never a
    verdict -- it must never be treated as equivalent to
    ``INSUFFICIENT_EVIDENCE``.

Public API: ``diagnose`` (diagnoser.py), the ``Diagnosis``/``DiagnosisStatus``/
``RootCauseHypothesis``/``DiagnosisValidationResult`` models (models.py), and
``validate_diagnosis`` (diagnosis_validator.py).
"""

from .diagnoser import diagnose, insufficient_evidence_diagnosis
from .diagnosis_validator import validate_diagnosis
from .models import Diagnosis, DiagnosisStatus, DiagnosisValidationResult, RootCauseHypothesis

__all__ = [
    "diagnose",
    "insufficient_evidence_diagnosis",
    "validate_diagnosis",
    "Diagnosis",
    "DiagnosisStatus",
    "DiagnosisValidationResult",
    "RootCauseHypothesis",
]
