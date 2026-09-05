"""Phase 6C: validated patch planning.

Sits between diagnosis (Phase 6A) and Gemini patch generation in the agent
graph (see ``app.services.agent.graph.patch_plan_node``): turns an
already-DIAGNOSED root cause and already-gathered ``retrieved_context``
into a structured, cited ``PatchPlan`` that ``plan_node`` may use to guide
its Gemini patch-generation prompt.

Never modifies files, by design: a ``PatchPlan`` is guidance for a LATER
Gemini call, never itself a diff. It is never passed to
``edit_node``/``apply_patch``.

The strongest invariant this package enforces: ONLY ``PatchPlanStatus.
PLANNED`` may result in a Gemini patch-generation call -- see
``app.services.agent.graph.plan_node``'s allow-list gate (``if
patch_plan_status == PLANNED: ... else: proposed_patches = []``). This is
deliberately an allow-list, not a denylist, so no status -- existing or
future -- can accidentally fall through to patch generation:

  - ``INSUFFICIENT_DIAGNOSIS`` (diagnosis was DIAGNOSIS_FAILED,
    INSUFFICIENT_EVIDENCE, absent, or retrieved_context was empty) NEVER
    results in a Gemini call, for planning OR for patch generation --
    ``plan_patches`` itself never calls Gemini in this case (the core
    no-fabricated-fix protection), and ``plan_node``'s gate independently
    ensures it regardless.
  - ``PLANNING_FAILED`` (planning's own process failed, or every proposed
    change failed deterministic validation) NEVER results in a patch-
    generation call either.
  - ``NOT_APPLICABLE`` (a genuine, evidence-grounded "no change needed"
    conclusion) is likewise never given a patch-generation call -- calling
    Gemini again after a validated conclusion that nothing needs to change
    would itself risk fabricating an unnecessary change.

Patch planning never calls reproduction, Docker, workspace creation, or
remote code-hosting functionality, and never affects finalize_node's
outcome classification or should_continue's retry routing -- it is purely
advisory input to plan_node's existing patch-generation prompt.

Public API: ``plan_patches``/``insufficient_diagnosis_plan`` (planner.py),
the ``PatchPlan``/``PatchPlanStatus``/``PlannedChange``/
``PatchPlanValidationResult`` models (models.py), and ``validate_patch_plan``
(patch_plan_validator.py).
"""

from .planner import insufficient_diagnosis_plan, plan_patches
from .patch_plan_validator import validate_patch_plan
from .models import PatchPlan, PatchPlanStatus, PatchPlanValidationResult, PlannedChange

__all__ = [
    "plan_patches",
    "insufficient_diagnosis_plan",
    "validate_patch_plan",
    "PatchPlan",
    "PatchPlanStatus",
    "PatchPlanValidationResult",
    "PlannedChange",
]
