"""Unit tests for app.services.diagnosis.models.

Covers the DiagnosisStatus/Diagnosis/RootCauseHypothesis shapes and the
to_dict() serialization that AgentState["diagnosis"] stores.
"""

from app.services.diagnosis.models import (
    Diagnosis,
    DiagnosisStatus,
    DiagnosisValidationResult,
    RootCauseHypothesis,
)
from app.services.qa.models import CitationRef


def test_diagnosis_status_values_are_stable_strings():
    """AgentState stores diagnosis_status as a plain string (.value) --
    the exact three literals must never change without a deliberate,
    coordinated update everywhere they're compared."""
    assert DiagnosisStatus.DIAGNOSED.value == "DIAGNOSED"
    assert DiagnosisStatus.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"
    assert DiagnosisStatus.DIAGNOSIS_FAILED.value == "DIAGNOSIS_FAILED"


def test_diagnosis_defaults_are_safe_placeholders():
    diagnosis = Diagnosis(status=DiagnosisStatus.INSUFFICIENT_EVIDENCE)
    assert diagnosis.summary == ""
    assert diagnosis.hypotheses == []
    assert diagnosis.confidence == "no_evidence"
    assert diagnosis.failure_reason is None


def test_diagnosis_to_dict_round_trips_hypotheses_and_citations():
    diagnosis = Diagnosis(
        status=DiagnosisStatus.DIAGNOSED,
        summary="The bug is in subtotal().",
        hypotheses=[
            RootCauseHypothesis(
                rank=1,
                description="subtotal() uses the wrong operator.",
                citations=[CitationRef(file_path="src/cart.py", start_line=1, end_line=5, symbol_name="subtotal")],
                suggested_fix_approach="Swap - for +.",
            )
        ],
        confidence="direct_evidence",
    )
    data = diagnosis.to_dict()

    assert data["status"] == "DIAGNOSED"
    assert data["summary"] == "The bug is in subtotal()."
    assert data["confidence"] == "direct_evidence"
    assert data["failure_reason"] is None
    assert len(data["hypotheses"]) == 1
    h = data["hypotheses"][0]
    assert h["rank"] == 1
    assert h["description"] == "subtotal() uses the wrong operator."
    assert h["suggested_fix_approach"] == "Swap - for +."
    assert h["citations"] == [
        {"file_path": "src/cart.py", "start_line": 1, "end_line": 5, "symbol_name": "subtotal"}
    ]


def test_diagnosis_failed_to_dict_has_empty_hypotheses_and_failure_reason():
    diagnosis = Diagnosis(
        status=DiagnosisStatus.DIAGNOSIS_FAILED,
        confidence="no_evidence",
        failure_reason="Gemini quota exceeded",
    )
    data = diagnosis.to_dict()
    assert data["status"] == "DIAGNOSIS_FAILED"
    assert data["hypotheses"] == []
    assert data["failure_reason"] == "Gemini quota exceeded"


def test_diagnosis_validation_result_defaults_to_no_notes():
    result = DiagnosisValidationResult(diagnosis=Diagnosis(status=DiagnosisStatus.INSUFFICIENT_EVIDENCE))
    assert result.notes == []
