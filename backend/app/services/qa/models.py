"""Answer-facing structured models for Deep Codebase Q&A.

These are the shapes an answer is built into (Part 2 of the Deep Q&A
design) -- distinct from the investigation-facing shapes in
investigator.py (``Evidence``, ``SymbolMatch``, ``FileInspection``) and
from retriever.py's ``RetrievedChunk``. ``CitationRef`` intentionally does
not duplicate ``RetrievedChunk``: once an answer is synthesized, only the
provenance (file/line/symbol) is needed, not ``source_code`` or
``similarity_score``.

Pydantic models (rather than dataclasses, unlike the investigation-layer
types) because these are what a Gemini JSON completion gets parsed into and
validated against, and -- from Phase 3 onward -- what the API will likely
serialize directly.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ConfidenceLevel = Literal["direct_evidence", "inferred", "no_evidence"]


class CitationRef(BaseModel):
    """A concrete piece of evidence backing a claim in a QAAnswer.

    Preserves the existing repository-wide citation convention
    (``file:start-end``, see ``RetrievedChunk.citation`` /
    ``VerificationResult``) via the ``citation`` property below.
    """

    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class FlowStep(BaseModel):
    """One ordered step in a traced multi-file flow (e.g. a payment flow),
    used only for ``kind="flow"``/``"architecture"`` answers."""

    order: int
    description: str
    file_path: str
    citation: Optional[CitationRef] = None


class QAAnswer(BaseModel):
    """The final, structured Deep Codebase Q&A answer.

    ``summary`` is always present (and is the whole answer for a simple
    lookup/symbol question -- see the investigation-depth design).
    ``details``/``flow_trace`` are populated only for broader
    flow/architecture questions. ``corrected_premise`` is set whenever the
    question's own asserted technology didn't match the repository's real,
    evidence-derived facts. ``confidence`` is never "direct_evidence" for a
    claim the evidence doesn't actually support -- see answerer.py.
    """

    summary: str
    details: Optional[str] = None
    flow_trace: Optional[List[FlowStep]] = None
    evidence: List[CitationRef] = Field(default_factory=list)
    corrected_premise: Optional[str] = None
    confidence: ConfidenceLevel
    projects_considered: List[str] = Field(default_factory=list)
