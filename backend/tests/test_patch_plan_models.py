"""Unit tests for app.services.patch_plan.models.

Covers the PatchPlanStatus/PatchPlan/PlannedChange shapes and the
to_dict() serialization that AgentState["patch_plan"] stores.
"""

from app.services.patch_plan.models import (
    PatchPlan,
    PatchPlanStatus,
    PatchPlanValidationResult,
    PlannedChange,
)
from app.services.qa.models import CitationRef


def test_patch_plan_status_values_are_stable_strings():
    """AgentState stores patch_plan_status as a plain string (.value) --
    the exact four literals must never change without a deliberate,
    coordinated update everywhere they're compared."""
    assert PatchPlanStatus.PLANNED.value == "PLANNED"
    assert PatchPlanStatus.NOT_APPLICABLE.value == "NOT_APPLICABLE"
    assert PatchPlanStatus.INSUFFICIENT_DIAGNOSIS.value == "INSUFFICIENT_DIAGNOSIS"
    assert PatchPlanStatus.PLANNING_FAILED.value == "PLANNING_FAILED"


def test_patch_plan_defaults_are_safe_placeholders():
    plan = PatchPlan(status=PatchPlanStatus.INSUFFICIENT_DIAGNOSIS)
    assert plan.summary == ""
    assert plan.changes == []
    assert plan.diagnosis_alignment is None
    assert plan.confidence == "no_evidence"
    assert plan.failure_reason is None


def test_planned_change_only_allows_modify_or_create_type_hint():
    """Not a runtime enforcement (dataclasses don't validate Literal types),
    but documents the schema's intended allow-list -- see
    patch_plan_validator's own runtime enforcement."""
    change = PlannedChange(file_path="a.py", change_type="modify", description="x", rationale="y")
    assert change.change_type == "modify"
    change2 = PlannedChange(file_path="b.py", change_type="create", description="x", rationale="y")
    assert change2.change_type == "create"


def test_patch_plan_to_dict_round_trips_changes_and_citations():
    plan = PatchPlan(
        status=PatchPlanStatus.PLANNED,
        summary="Fix the operator.",
        changes=[
            PlannedChange(
                file_path="src/math_lib.py",
                change_type="modify",
                description="Swap - for +.",
                rationale="Matches the diagnosed cause.",
                citations=[CitationRef(file_path="src/math_lib.py", start_line=1, end_line=2, symbol_name="add")],
                symbols_affected=["add"],
            )
        ],
        diagnosis_alignment="Directly addresses the diagnosed operator bug.",
        confidence="inferred",
    )
    data = plan.to_dict()

    assert data["status"] == "PLANNED"
    assert data["summary"] == "Fix the operator."
    assert data["diagnosis_alignment"] == "Directly addresses the diagnosed operator bug."
    assert data["confidence"] == "inferred"
    assert data["failure_reason"] is None
    assert len(data["changes"]) == 1
    change = data["changes"][0]
    assert change["file_path"] == "src/math_lib.py"
    assert change["change_type"] == "modify"
    assert change["symbols_affected"] == ["add"]
    assert change["citations"] == [
        {"file_path": "src/math_lib.py", "start_line": 1, "end_line": 2, "symbol_name": "add"}
    ]


def test_patch_plan_failed_to_dict_has_empty_changes_and_failure_reason():
    plan = PatchPlan(
        status=PatchPlanStatus.PLANNING_FAILED,
        confidence="no_evidence",
        failure_reason="Gemini quota exceeded",
    )
    data = plan.to_dict()
    assert data["status"] == "PLANNING_FAILED"
    assert data["changes"] == []
    assert data["failure_reason"] == "Gemini quota exceeded"


def test_patch_plan_validation_result_defaults_to_no_notes():
    result = PatchPlanValidationResult(plan=PatchPlan(status=PatchPlanStatus.NOT_APPLICABLE))
    assert result.notes == []
