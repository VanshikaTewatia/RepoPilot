"""Unit tests for app.services.patch_plan.patch_plan_validator.validate_patch_plan.

Mirrors app.services.diagnosis.diagnosis_validator's own tests: the LLM's
structured output is never trusted directly -- hallucinated citations,
unsafe paths, and unsupported change types are dropped/rejected
deterministically, regardless of what Gemini itself claimed.
"""

from app.services.patch_plan.models import PatchPlan, PatchPlanStatus, PlannedChange
from app.services.patch_plan.patch_plan_validator import MAX_CHANGES, validate_patch_plan
from app.services.qa.models import CitationRef


def _planned(**overrides) -> PatchPlan:
    defaults = dict(
        status=PatchPlanStatus.PLANNED,
        summary="x",
        changes=[PlannedChange(file_path="a.py", change_type="modify", description="d", rationale="r")],
        confidence="inferred",
    )
    defaults.update(overrides)
    return PatchPlan(**defaults)


def test_non_planned_status_passes_through_unchanged():
    """NOT_APPLICABLE/INSUFFICIENT_DIAGNOSIS/PLANNING_FAILED have nothing
    to filter -- validate_patch_plan must never alter their status or
    failure_reason."""
    failed = PatchPlan(status=PatchPlanStatus.PLANNING_FAILED, failure_reason="quota exceeded")
    result = validate_patch_plan(failed, real_files=set(), diagnosis_citation_files=set())
    assert result.plan is failed
    assert result.notes == []


def test_valid_plan_is_preserved():
    plan = _planned(
        changes=[
            PlannedChange(
                file_path="a.py",
                change_type="modify",
                description="d",
                rationale="r",
                citations=[CitationRef(file_path="a.py", start_line=1, end_line=2)],
            )
        ]
    )
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files={"a.py"})
    assert result.plan.status == PatchPlanStatus.PLANNED
    assert len(result.plan.changes) == 1
    assert result.plan.changes[0].file_path == "a.py"


def test_absolute_file_path_is_dropped():
    plan = _planned(changes=[PlannedChange(file_path="/etc/passwd", change_type="modify", description="d", rationale="r")])
    result = validate_patch_plan(plan, real_files=set(), diagnosis_citation_files=set())
    assert result.plan.status == PatchPlanStatus.PLANNING_FAILED
    assert any("unsafe file_path" in note for note in result.notes)


def test_traversal_file_path_is_dropped():
    plan = _planned(changes=[PlannedChange(file_path="../../secrets.py", change_type="modify", description="d", rationale="r")])
    result = validate_patch_plan(plan, real_files=set(), diagnosis_citation_files=set())
    assert result.plan.status == PatchPlanStatus.PLANNING_FAILED
    assert any("unsafe file_path" in note for note in result.notes)


def test_windows_drive_absolute_path_is_dropped():
    plan = _planned(changes=[PlannedChange(file_path="C:\\Windows\\system.ini", change_type="modify", description="d", rationale="r")])
    result = validate_patch_plan(plan, real_files=set(), diagnosis_citation_files=set())
    assert result.plan.status == PatchPlanStatus.PLANNING_FAILED


def test_unsupported_change_type_is_dropped():
    plan = _planned(changes=[PlannedChange(file_path="a.py", change_type="delete", description="d", rationale="r")])
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files=set())
    assert result.plan.status == PatchPlanStatus.PLANNING_FAILED
    assert any("unsupported change_type" in note for note in result.notes)


def test_hallucinated_citation_is_filtered_out():
    plan = _planned(
        changes=[
            PlannedChange(
                file_path="a.py",
                change_type="modify",
                description="d",
                rationale="r",
                citations=[
                    CitationRef(file_path="a.py", start_line=1, end_line=2),
                    CitationRef(file_path="made_up_file.py", start_line=1, end_line=2),
                ],
            )
        ]
    )
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files=set())
    files = {c.file_path for c in result.plan.changes[0].citations}
    assert files == {"a.py"}
    assert any("hallucinated" in note for note in result.notes)


def test_excessive_changes_are_truncated():
    changes = [PlannedChange(file_path=f"f{i}.py", change_type="modify", description="d", rationale="r") for i in range(10)]
    plan = _planned(changes=changes)
    result = validate_patch_plan(plan, real_files={f"f{i}.py" for i in range(10)}, diagnosis_citation_files=set())
    assert len(result.plan.changes) == MAX_CHANGES
    assert any("truncated changes" in note for note in result.notes)


def test_all_changes_rejected_downgrades_to_planning_failed():
    """An applicable<->content inconsistency (PLANNED with zero surviving
    changes) must never be returned as-is -- see the module docstring."""
    plan = _planned(changes=[PlannedChange(file_path="/etc/passwd", change_type="modify", description="d", rationale="r")])
    result = validate_patch_plan(plan, real_files=set(), diagnosis_citation_files=set())
    assert result.plan.status == PatchPlanStatus.PLANNING_FAILED
    assert result.plan.changes == []
    assert result.plan.failure_reason is not None
    assert any("downgrading to PLANNING_FAILED" in note for note in result.notes)


def test_thin_evidence_direct_evidence_confidence_is_downgraded():
    plan = _planned(confidence="direct_evidence")
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files=set())
    assert result.plan.confidence == "inferred"


def test_sufficient_evidence_direct_evidence_confidence_is_preserved():
    plan = _planned(
        confidence="direct_evidence",
        changes=[
            PlannedChange(
                file_path="a.py",
                change_type="modify",
                description="d",
                rationale="r",
                citations=[
                    CitationRef(file_path="a.py", start_line=1, end_line=2),
                    CitationRef(file_path="a.py", start_line=5, end_line=6),
                ],
            )
        ],
    )
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files={"a.py"})
    assert result.plan.confidence == "direct_evidence"


def test_diagnosis_alignment_overlap_preserves_confidence():
    plan = _planned(
        confidence="direct_evidence",
        changes=[
            PlannedChange(
                file_path="a.py",
                change_type="modify",
                description="d",
                rationale="r",
                citations=[CitationRef(file_path="a.py", start_line=1, end_line=2), CitationRef(file_path="a.py", start_line=3, end_line=4)],
            )
        ],
    )
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files={"a.py"})
    assert not any("alignment unconfirmed" in note for note in result.notes)


def test_diagnosis_alignment_mismatch_downgrades_confidence_and_notes():
    plan = _planned(
        confidence="direct_evidence",
        changes=[
            PlannedChange(
                file_path="unrelated.py",
                change_type="modify",
                description="d",
                rationale="r",
                citations=[CitationRef(file_path="unrelated.py", start_line=1, end_line=2), CitationRef(file_path="unrelated.py", start_line=3, end_line=4)],
            )
        ],
    )
    result = validate_patch_plan(plan, real_files={"unrelated.py"}, diagnosis_citation_files={"diagnosed_file.py"})
    assert result.plan.confidence == "inferred"
    assert any("alignment unconfirmed" in note for note in result.notes)
    # a legitimate, evidence-grounded change is NOT rejected, only downgraded
    assert len(result.plan.changes) == 1


def test_long_text_fields_are_bounded():
    plan = _planned(
        summary="x" * 5000,
        diagnosis_alignment="y" * 5000,
        changes=[
            PlannedChange(file_path="a.py", change_type="modify", description="z" * 5000, rationale="w" * 5000)
        ],
    )
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files=set())
    assert len(result.plan.summary) < 5000
    assert len(result.plan.diagnosis_alignment) < 5000
    assert len(result.plan.changes[0].description) < 5000
    assert len(result.plan.changes[0].rationale) < 5000


def test_missing_diagnosis_alignment_is_left_none_not_fabricated():
    plan = _planned(diagnosis_alignment=None)
    result = validate_patch_plan(plan, real_files={"a.py"}, diagnosis_citation_files=set())
    assert result.plan.diagnosis_alignment is None
