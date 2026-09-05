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
    or ``should_continue`` -- those two remain entirely unaware of
    diagnosis. Patch generation itself, however, IS gated on diagnosis as
    of Phase 6C: a ``DIAGNOSIS_FAILED`` or ``INSUFFICIENT_EVIDENCE`` result
    is converted by ``app.services.patch_plan.plan_patches`` into
    ``PatchPlanStatus.INSUFFICIENT_DIAGNOSIS``, which closes ``plan_node``'s
    PLANNED-only allow-list gate on the Gemini patch-generation call --
    see ``app.services.patch_plan``'s package docstring. There is no
    fallback to a pre-Phase-6A/6C prompt; this is the intended
    no-fabricated-fix behavior, not a bug.
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
