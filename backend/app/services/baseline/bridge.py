"""Plan -> execution-input bridge (Phase 4B-2).

Converts an already-validated ``ReproductionPlan`` (Phase 4B-1) into a Phase
4A ``ReproductionInput`` -- and nothing else. This module never executes a
command, never calls ``BaselineExecutor``/``VerificationEngine.
execute_command``/``service.reproduce``, and never classifies an outcome;
it only decides whether a safe, unambiguous ``ReproductionInput`` can be
constructed at all.

Three-way outcome (see ``models.BridgeOutcome`` for the full contract):

1. ``plan.planning_failed=True``            -> PLANNING_FAILED (never executed)
2. ``plan.planning_failed=False`` and not
   ``plan.applicable``                      -> NOT_APPLICABLE (genuine verdict, never executed)
3. ``plan.planning_failed=False`` and
   ``plan.applicable`` and re-validation +
   real-workspace reconciliation succeed    -> EXECUTABLE (a ReproductionInput is returned)

Every "unsafe or ambiguous" situation the bridge itself discovers (a
re-validation failure, a working_dir that escapes the *real* filesystem
workspace, or the plan's stated ecosystem/image conflicting with what is
*actually* detected there) also collapses into PLANNING_FAILED -- never a
silent best-effort execution against the wrong ecosystem/image, and never a
verdict about the reported bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .executor import (
    BaselineExecutor,
    WorkspaceEscapeError,
    _safe_join_working_dir,
)
from .models import (
    BridgeOutcome,
    PlanBridgeResult,
    ReproductionExpectation,
    ReproductionInput,
    ReproductionPlan,
    RepositoryEvidence,
)
from .plan_validator import validate_plan


def _effective_working_dir(plan: ReproductionPlan) -> Optional[str]:
    """``working_dir`` is authoritative when given. Otherwise, a
    ``project_root`` other than "." (repo root) is used as the execution
    subdirectory -- ``project_root`` alone never means anything beyond
    that; it is not itself trusted as a safety-checked path until resolved
    below via the exact same mechanism Phase 4A's executor uses."""
    if plan.working_dir:
        return plan.working_dir
    if plan.project_root and plan.project_root != ".":
        return plan.project_root
    return None


def build_reproduction_input(
    plan: ReproductionPlan,
    evidence: RepositoryEvidence,
    workspace_path: str,
) -> PlanBridgeResult:
    """Attempt to convert ``plan`` into an executable ``ReproductionInput``.

    ``workspace_path`` is the real, already-isolated repository checkout
    this reproduction would run against -- supplied by the caller (e.g. the
    task's own ``WorkspaceManager``-provided directory); the bridge never
    invents, looks up, or derives it itself.

    ``evidence`` is the same ``RepositoryEvidence`` the plan was produced
    from, re-used here to re-run ``plan_validator.validate_plan`` -- the
    bridge is a new trust boundary and never assumes a plan that reaches it
    was actually produced by ``planner.plan_reproduction`` (which already
    validates), so it re-validates independently rather than duplicating
    those same rules.
    """
    # 1. planning_failed is never executed, and is never reinterpreted.
    if plan.planning_failed:
        return PlanBridgeResult(
            outcome=BridgeOutcome.PLANNING_FAILED,
            detail=plan.failure_reason or plan.reason,
        )

    # 2. A genuine "not applicable" verdict is preserved, never executed.
    if not plan.applicable:
        return PlanBridgeResult(outcome=BridgeOutcome.NOT_APPLICABLE, detail=plan.reason)

    # 3. Never trust that a plan reaching this function was already
    # validated by planner.plan_reproduction -- re-run the exact same
    # deterministic safety validator independently.
    validation = validate_plan(plan, evidence)
    if not validation.valid:
        return PlanBridgeResult(
            outcome=BridgeOutcome.PLANNING_FAILED,
            detail="Plan failed re-validation at the execution bridge: " + "; ".join(validation.errors),
        )

    base = Path(workspace_path).resolve()
    if not base.is_dir():
        return PlanBridgeResult(
            outcome=BridgeOutcome.PLANNING_FAILED,
            detail=f"Workspace directory does not exist: {base}",
        )

    working_dir = _effective_working_dir(plan)

    # Re-check workspace containment using Phase 4A's own real,
    # filesystem-aware (Path.parents-based) mechanism -- plan_validator's
    # check above is purely lexical (no real workspace existed at planning
    # time); this is the authoritative check, reused rather than duplicated.
    try:
        resolved_dir = _safe_join_working_dir(base, working_dir) if working_dir else base
    except WorkspaceEscapeError as e:
        return PlanBridgeResult(outcome=BridgeOutcome.PLANNING_FAILED, detail=str(e))

    # Real, authoritative ecosystem/image detection at the actual execution
    # directory -- reusing BaselineExecutor.detect_ecosystem (the same
    # detection the executor itself performs at run time) rather than
    # trusting RepositoryEvidence's possibly-stale snapshot or the plan's
    # own (LLM-reported) ecosystem/image fields.
    detection = BaselineExecutor().detect_ecosystem(resolved_dir)
    real_ecosystem = detection.ecosystem if detection and detection.adapter else None
    real_image = detection.adapter.docker_image if detection and detection.adapter else None

    if plan.ecosystem and real_ecosystem and plan.ecosystem != real_ecosystem:
        return PlanBridgeResult(
            outcome=BridgeOutcome.PLANNING_FAILED,
            detail=(
                f"Plan's stated ecosystem '{plan.ecosystem}' conflicts with the "
                f"ecosystem actually detected at '{resolved_dir}' ('{real_ecosystem}') -- "
                "refusing to execute against an ambiguous ecosystem."
            ),
        )
    if plan.ecosystem and not real_ecosystem:
        return PlanBridgeResult(
            outcome=BridgeOutcome.PLANNING_FAILED,
            detail=(
                f"Plan claims ecosystem '{plan.ecosystem}' but no ecosystem could "
                f"actually be detected at '{resolved_dir}' -- refusing to execute "
                "against an unverifiable ecosystem claim."
            ),
        )

    if plan.image and real_image and plan.image != real_image:
        return PlanBridgeResult(
            outcome=BridgeOutcome.PLANNING_FAILED,
            detail=(
                f"Plan's stated image '{plan.image}' conflicts with the image the "
                f"detected ecosystem would actually use ('{real_image}') -- refusing "
                "to execute against a mismatched image."
            ),
        )
    # plan.image with no real ecosystem detected at all is accepted as-is:
    # validate_plan already confirmed above that it is a member of Phase
    # 4A's own allowlist (_allowed_images), so there is nothing arbitrary
    # about it, and there is no authoritative real-ecosystem signal here to
    # conflict with.

    reproduction_input = ReproductionInput(
        workspace_path=workspace_path,
        commands=plan.commands,
        working_dir=working_dir,
        timeout_seconds=plan.timeout_seconds,
        image=plan.image,
        expectation=ReproductionExpectation(
            exit_code_semantics=plan.exit_code_semantics,
            reproduced_output_pattern=plan.reproduced_output_pattern,
            not_reproduced_output_pattern=plan.not_reproduced_output_pattern,
        ),
        task_context=plan.expected_observation,
    )
    return PlanBridgeResult(outcome=BridgeOutcome.EXECUTABLE, reproduction_input=reproduction_input, detail=plan.reason)
