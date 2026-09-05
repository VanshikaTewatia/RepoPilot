"""Unit tests for app.services.diagnosis.diagnosis_validator.validate_diagnosis.

Mirrors app.services.qa.answerer's own deterministic-override tests
(test_qa_answerer.py): the LLM's structured output is never trusted
directly -- hallucinated citations are dropped and a thin-evidence
"direct_evidence" confidence is downgraded, regardless of what Gemini
itself claimed.
"""

from app.services.diagnosis.diagnosis_validator import validate_diagnosis
from app.services.diagnosis.models import Diagnosis, DiagnosisStatus, RootCauseHypothesis
from app.services.qa.models import CitationRef


def _diagnosed(**overrides) -> Diagnosis:
    defaults = dict(
        status=DiagnosisStatus.DIAGNOSED,
        summary="x",
        hypotheses=[],
        confidence="inferred",
    )
    defaults.update(overrides)
    return Diagnosis(**defaults)


def test_non_diagnosed_status_passes_through_unchanged():
    """INSUFFICIENT_EVIDENCE/DIAGNOSIS_FAILED have nothing to filter --
    validate_diagnosis must never alter their status or failure_reason."""
    failed = Diagnosis(status=DiagnosisStatus.DIAGNOSIS_FAILED, failure_reason="quota exceeded")
    result = validate_diagnosis(failed, real_files=set(), total_evidence_pieces=0)
    assert result.diagnosis is failed
    assert result.notes == []


def test_hallucinated_citation_is_filtered_out():
    diagnosis = _diagnosed(
        hypotheses=[
            RootCauseHypothesis(
                rank=1,
                description="x",
                citations=[
                    CitationRef(file_path="src/cart.py", start_line=1, end_line=5),
                    CitationRef(file_path="src/made_up_file.py", start_line=1, end_line=2),
                ],
            )
        ]
    )
    result = validate_diagnosis(diagnosis, real_files={"src/cart.py"}, total_evidence_pieces=2)

    files = {c.file_path for c in result.diagnosis.hypotheses[0].citations}
    assert files == {"src/cart.py"}
    assert any("hallucinated" in note for note in result.notes)


def test_thin_evidence_direct_evidence_confidence_is_downgraded():
    diagnosis = _diagnosed(confidence="direct_evidence")
    result = validate_diagnosis(diagnosis, real_files=set(), total_evidence_pieces=1)
    assert result.diagnosis.confidence == "inferred"
    assert any("downgraded confidence" in note for note in result.notes)


def test_sufficient_evidence_direct_evidence_confidence_is_preserved():
    diagnosis = _diagnosed(confidence="direct_evidence")
    result = validate_diagnosis(diagnosis, real_files=set(), total_evidence_pieces=2)
    assert result.diagnosis.confidence == "direct_evidence"


def test_hypotheses_are_truncated_to_max_count():
    hypotheses = [RootCauseHypothesis(rank=i, description=f"h{i}") for i in range(1, 10)]
    diagnosis = _diagnosed(hypotheses=hypotheses)
    result = validate_diagnosis(diagnosis, real_files=set(), total_evidence_pieces=5)
    assert len(result.diagnosis.hypotheses) == 5
    assert any("truncated hypotheses" in note for note in result.notes)


def test_long_text_fields_are_bounded():
    diagnosis = _diagnosed(
        summary="x" * 5000,
        hypotheses=[RootCauseHypothesis(rank=1, description="y" * 5000, suggested_fix_approach="z" * 5000)],
    )
    result = validate_diagnosis(diagnosis, real_files=set(), total_evidence_pieces=5)
    assert len(result.diagnosis.summary) < 5000
    assert len(result.diagnosis.hypotheses[0].description) < 5000
    assert len(result.diagnosis.hypotheses[0].suggested_fix_approach) < 5000


def test_no_citations_needs_no_filtering_note():
    diagnosis = _diagnosed(hypotheses=[RootCauseHypothesis(rank=1, description="x", citations=[])])
    result = validate_diagnosis(diagnosis, real_files=set(), total_evidence_pieces=5)
    assert result.diagnosis.hypotheses[0].citations == []
    assert result.notes == []
