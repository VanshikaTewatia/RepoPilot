"""Deep Codebase Q&A orchestration entry point.

question -> QuestionClass -> Investigator -> Evidence -> structured answer

Composes, without modifying, every previously-built piece: classify_question
(classifier.py), investigate (investigator.py, Phase 0/1), and
generate_answer/no_evidence_answer (answerer.py) -- no project-detection,
retrieval, or answer-synthesis logic is duplicated here.

This is a new, additive service. It does not touch and is not used by the
existing ``POST /api/rag/ask`` endpoint (app.api.v1.rag) or
``CodeRetriever.answer_question``'s plain single-shot path -- both continue
to work exactly as before.
"""

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.qa.answerer import generate_answer, no_evidence_answer
from app.services.qa.classifier import classify_question
from app.services.qa.investigator import investigate
from app.services.qa.models import QAAnswer
from app.services.rag.retriever import CodeRetriever


class QuestionValidationError(ValueError):
    """Raised when the question itself is invalid (missing/blank)."""


async def ask_codebase(
    question: str,
    workspace_dir: str,
    repository_id: int,
    db: Optional[AsyncSession] = None,
    retriever: Optional[CodeRetriever] = None,
) -> QAAnswer:
    """Answer a question about a repository with full, evidence-based
    investigation -- the Deep Codebase Q&A entry point.

    1. Validate the question.
    2. Classify it (kind/depth/subject_terms/user_asserted_tech) --
       classify_question() never raises; a classification failure degrades
       to a safe "targeted" depth rather than propagating.
    3-6. Investigate the repository at the classified depth -- project
       detection, multi-project relevance selection, and depth-aware
       RAG/search/read are all handled inside investigate() (Phase 0/1);
       not reimplemented here. The classifier's subject_terms/kind/
       likely_multi_file are passed through to inform search (never to
       override the deterministic project/framework facts investigate()
       gathers -- see investigator.py).
    7. No evidence -> a deterministic no-evidence QAAnswer, no Gemini call.
    8. Otherwise -> generate_answer() for a structured, cited answer.

    Deliberately does not wrap steps 2-4 in a blanket try/except: a genuine
    bug (as opposed to a Gemini/retrieval failure, which classify_question/
    investigate/generate_answer already handle internally and never raise
    for) should propagate as a real exception rather than being silently
    swallowed into a misleadingly generic answer.
    """
    if not question or not question.strip():
        raise QuestionValidationError("Question must not be empty.")

    question_class = await classify_question(question)

    result = await investigate(
        workspace_dir=workspace_dir,
        repository_id=repository_id,
        question=question,
        depth=question_class.depth,
        db=db,
        retriever=retriever,
        subject_terms=question_class.subject_terms,
        kind=question_class.kind,
        likely_multi_file=question_class.likely_multi_file,
    )

    if not result.has_evidence:
        return no_evidence_answer(result.investigated_projects)

    return await generate_answer(question, question_class, result.evidence)
