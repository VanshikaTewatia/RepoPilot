"""Deep Codebase Q&A investigation package.

Builds structured, evidence-based investigation on top of the existing
verification (project/ecosystem detection) and RAG (semantic retrieval)
infrastructure. See investigator.py for the depth-aware investigation
pipeline; question classification and answer synthesis are later phases
that will consume InvestigationResult/Evidence, not part of this package
yet.
"""

from app.services.qa.investigator import (
    VALID_DEPTHS,
    Evidence,
    FileInspection,
    InvestigationResult,
    SymbolMatch,
    investigate,
)

__all__ = [
    "VALID_DEPTHS",
    "Evidence",
    "FileInspection",
    "InvestigationResult",
    "SymbolMatch",
    "investigate",
]
