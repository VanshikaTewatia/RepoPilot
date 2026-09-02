"""Deep Codebase Q&A package.

question -> QuestionClass (classifier.py) -> Investigator (investigator.py)
-> Evidence -> structured answer (answerer.py), composed by service.py.

Builds on the existing verification (project/ecosystem detection) and RAG
(semantic retrieval) infrastructure throughout; see each module's docstring
for exactly what it reuses. Not yet wired into the API or frontend -- see
service.ask_codebase for the standalone entry point.
"""

from app.services.qa.answerer import generate_answer, no_evidence_answer
from app.services.qa.classifier import QuestionClass, classify_question
from app.services.qa.investigator import (
    VALID_DEPTHS,
    Evidence,
    FileInspection,
    InvestigationResult,
    SymbolMatch,
    investigate,
)
from app.services.qa.models import CitationRef, FlowStep, QAAnswer
from app.services.qa.service import QuestionValidationError, ask_codebase

__all__ = [
    "VALID_DEPTHS",
    "ask_codebase",
    "classify_question",
    "generate_answer",
    "no_evidence_answer",
    "investigate",
    "CitationRef",
    "Evidence",
    "FileInspection",
    "FlowStep",
    "InvestigationResult",
    "QAAnswer",
    "QuestionClass",
    "QuestionValidationError",
    "SymbolMatch",
]
