"""Structured answer synthesis for Deep Codebase Q&A.

Turns an already-gathered ``Evidence`` set (from investigator.py), the
question's ``QuestionClass`` (from classifier.py), and the question itself
into a ``QAAnswer`` (models.py) via a single Gemini call. Reuses the exact
Gemini client/call/test-mode conventions already used elsewhere (see
app.services.rag.retriever.answer_question and
app.services.agent.graph._generate_patches_with_gemini).

Never fabricates:
  - When there is no evidence at all, ``generate_answer`` returns the
    deterministic ``no_evidence_answer`` WITHOUT ever calling Gemini --
    asking an LLM to "explain" an empty evidence set risks exactly the
    invented answer this module exists to prevent.
  - Every citation Gemini returns is checked against the real evidence's
    own file paths; any citation for a file that was never actually
    inspected/retrieved/matched is dropped rather than trusted (see
    ``_filter_hallucinated_citations``).
  - ``corrected_premise`` is computed deterministically from the real,
    evidence-derived project facts (never left purely to the LLM
    remembering the instruction) -- see ``_compute_corrected_premise``.
  - A "direct_evidence" verdict is downgraded to "inferred" when the
    evidence gathered is too sparse (a single chunk/file/match) to
    directly support a confident, specific claim -- see
    ``_sanity_check_confidence``. Weak evidence must never be reported as
    if it were a confident architectural fact.
"""

from typing import List, Set

from google import genai

from app.core.config import settings
from app.core.logging import logger
from app.services.qa.classifier import QuestionClass
from app.services.qa.investigator import Evidence
from app.services.qa.json_utils import parse_json_object
from app.services.qa.models import CitationRef, FlowStep, QAAnswer

NO_EVIDENCE_SUMMARY = "There is no evidence of the requested component/feature in this repository."

_ANSWER_SYSTEM_INSTRUCTION = (
    "You are RepoPilot, an expert AI software engineer investigating a real "
    "repository to answer a user's question. Follow these rules strictly:\n"
    "1. Repository evidence is authoritative -- it describes what actually "
    "exists in this repository.\n"
    "2. The user's own terminology is a hypothesis, not ground truth -- "
    "they may name the wrong framework, component, or technology.\n"
    "3. Answer ONLY from the supplied Evidence below. Do not use outside "
    "knowledge of how other, similar projects are typically built.\n"
    "4. Do not invent files, symbols, frameworks, flows, or behavior that "
    "are not present in the Evidence.\n"
    "5. If the Evidence is insufficient to answer confidently, say so "
    "plainly rather than guessing.\n"
    "6. Every factual claim should be backed by evidence where possible -- "
    "cite a specific file and line range from the Evidence.\n"
    "7. If the question's asserted technology conflicts with the actual "
    "project facts given below, explain the correction in corrected_premise.\n"
    '8. If there is no relevant Evidence at all, set confidence to "no_evidence".\n'
    '9. When confidence is "no_evidence", the summary must be EXACTLY: '
    f'"{NO_EVIDENCE_SUMMARY}"\n'
    "\n"
    "Return ONLY a JSON object, no markdown fences, no commentary. Schema:\n"
    "{\n"
    '  "summary": "1-3 sentence direct answer",\n'
    '  "details": "deeper explanation, or null for a simple lookup",\n'
    '  "flow_trace": [{"order": 1, "description": "...", "file_path": "...", '
    '"citation": {"file_path": "...", "start_line": 1, "end_line": 10, '
    '"symbol_name": "..." or null}}] or null,\n'
    '  "evidence": [{"file_path": "...", "start_line": 1, "end_line": 10, '
    '"symbol_name": "..." or null}, ...],\n'
    '  "corrected_premise": "correction text, or null",\n'
    '  "confidence": "direct_evidence" | "inferred" | "no_evidence"\n'
    "}"
)


def no_evidence_answer(investigated_projects: List[str]) -> QAAnswer:
    """The deterministic answer for a no-evidence case -- never produced by
    asking Gemini to explain an empty evidence set. Both the orchestrator
    (before ever calling generate_answer) and generate_answer itself (as a
    defensive guard) return exactly this, so the required sentence is never
    left to an LLM's own phrasing."""
    return QAAnswer(
        summary=NO_EVIDENCE_SUMMARY,
        details=None,
        flow_trace=None,
        evidence=[],
        corrected_premise=None,
        confidence="no_evidence",
        projects_considered=list(investigated_projects),
    )


def _gemini_unavailable() -> bool:
    key = settings.gemini_api_key
    return not key or key.startswith("test") or key.startswith("mock")


def _format_project_facts(e: Evidence) -> str:
    return (
        f"Project root: {e.project_root}\n"
        f"Ecosystem: {e.ecosystem}\n"
        f"Languages: {', '.join(e.languages) or 'unknown'}\n"
        f"Frameworks: {', '.join(e.frameworks) or 'none detected'}\n"
        f"Build system: {e.build_system or 'unknown'}\n"
        f"Package manager: {e.package_manager or 'unknown'}\n"
        f"Test system: {e.test_system or 'unknown'}\n"
        f"Manifest evidence: {', '.join(e.project_evidence) or 'none'}"
    )


def _format_evidence(evidence_list: List[Evidence]) -> str:
    parts: List[str] = []
    for e in evidence_list:
        section = [f"## Project: {e.project_root}", _format_project_facts(e)]
        if e.chunks:
            section.append("### Retrieved code chunks:")
            for c in e.chunks:
                section.append(
                    f"- {c.citation} (symbol: {c.symbol_name or 'block'}, {c.symbol_type})\n"
                    f"```\n{c.source_code}\n```"
                )
        if e.files_inspected:
            section.append("### Files inspected:")
            for f in e.files_inspected:
                suffix = " (truncated)" if f.truncated else ""
                section.append(f"- {f.citation}{suffix}\n```\n{f.content}\n```")
        if e.symbol_matches:
            section.append("### Related search matches:")
            for m in e.symbol_matches:
                section.append(f"- {m.citation}: {m.content}")
        parts.append("\n".join(section))
    return "\n\n".join(parts)


def _build_answer_prompt(question: str, question_class: QuestionClass, evidence_list: List[Evidence]) -> str:
    return (
        f"User Question: {question}\n\n"
        f"Question classification: kind={question_class.kind}, depth={question_class.depth}\n"
        "User-asserted technology (the user's HYPOTHESIS ONLY -- verify against "
        f"the real project facts below): {', '.join(question_class.user_asserted_tech) or 'none stated'}\n\n"
        f"Repository Evidence (authoritative):\n{_format_evidence(evidence_list)}\n\n"
        "Produce the JSON answer now:"
    )


def _real_evidence_files(evidence_list: List[Evidence]) -> Set[str]:
    files: Set[str] = set()
    for e in evidence_list:
        files.update(c.file_path for c in e.chunks)
        files.update(f.file_path for f in e.files_inspected)
        files.update(m.file for m in e.symbol_matches)
    return files


def _filter_hallucinated_citations(
    citations: List[CitationRef], real_files: Set[str]
) -> List[CitationRef]:
    """Drop any citation for a file that was never actually gathered as
    evidence -- Gemini is instructed not to invent files, but this is the
    deterministic backstop for when it does anyway."""
    return [c for c in citations if c.file_path in real_files]


def _compute_corrected_premise(user_asserted_tech: List[str], evidence_list: List[Evidence]) -> str | None:
    """Deterministically check the question's asserted technology against
    the REAL, evidence-derived facts for every investigated project --
    never trust the LLM alone to remember this: a prompt instruction can be
    missed, this check cannot."""
    if not user_asserted_tech:
        return None

    real_facts: Set[str] = set()
    for e in evidence_list:
        real_facts.add(e.ecosystem.lower())
        real_facts.update(lang.lower() for lang in e.languages)
        real_facts.update(fw.lower() for fw in e.frameworks)

    unmatched = [t for t in user_asserted_tech if t.lower() not in real_facts]
    if not unmatched:
        return None

    actual_frameworks = sorted({fw for e in evidence_list for fw in e.frameworks})
    actual_desc = ", ".join(actual_frameworks) if actual_frameworks else ", ".join(
        sorted({e.ecosystem for e in evidence_list})
    )
    if not actual_desc:
        return None

    return (
        f"The question mentions {', '.join(unmatched)}, but this repository "
        f"actually uses {actual_desc}."
    )


def _sanity_check_confidence(confidence: str, evidence_list: List[Evidence]) -> str:
    """Never let a "direct_evidence" verdict stand on objectively thin
    evidence (e.g. a single chunk) -- downgrade to "inferred". A broad claim
    ("authentication uses JWT in localStorage") needs more than one isolated
    fragment to be reported as directly evidenced."""
    if confidence != "direct_evidence":
        return confidence
    total_pieces = sum(
        len(e.chunks) + len(e.files_inspected) + len(e.symbol_matches) for e in evidence_list
    )
    return "inferred" if total_pieces <= 1 else confidence


async def generate_answer(
    question: str,
    question_class: QuestionClass,
    evidence_list: List[Evidence],
) -> QAAnswer:
    """Synthesize a structured, cited QAAnswer from already-gathered Evidence.

    Never calls Gemini when there's no evidence to answer from (returns
    ``no_evidence_answer`` immediately). ``corrected_premise`` and
    ``confidence`` are always deterministically (re-)computed after
    parsing, regardless of what Gemini returned for them -- see module
    docstring.
    """
    investigated_projects = [e.project_root for e in evidence_list]

    if not any(e.has_evidence for e in evidence_list):
        return no_evidence_answer(investigated_projects)

    corrected_premise = _compute_corrected_premise(question_class.user_asserted_tech, evidence_list)
    real_files = _real_evidence_files(evidence_list)

    if _gemini_unavailable():
        answer = _mock_answer(evidence_list, investigated_projects)
    else:
        prompt = _build_answer_prompt(question, question_class, evidence_list)
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model_name,
                contents=prompt,
                config={"system_instruction": _ANSWER_SYSTEM_INSTRUCTION},
            )
            raw_text = response.text or ""
            data = parse_json_object(raw_text)
            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
            answer = _answer_from_data(data, investigated_projects)
        except Exception as e:  # noqa: BLE001 -- any failure (network, quota, malformed JSON) degrades safely
            logger.error(f"Error generating structured answer with Gemini: {e}")
            answer = QAAnswer(
                summary=(
                    "Relevant evidence was found, but the answer could not be generated "
                    f"due to an error: {e}"
                ),
                evidence=[],
                confidence="inferred",
                projects_considered=investigated_projects,
            )

    # Deterministic overrides -- never left solely to the LLM's own output.
    answer.evidence = _filter_hallucinated_citations(answer.evidence, real_files)
    answer.corrected_premise = corrected_premise
    answer.confidence = _sanity_check_confidence(answer.confidence, evidence_list)
    answer.projects_considered = investigated_projects
    return answer


def _parse_citation(data: dict) -> CitationRef:
    return CitationRef(
        file_path=str(data.get("file_path", "")),
        start_line=int(data.get("start_line", 1)),
        end_line=int(data.get("end_line", data.get("start_line", 1))),
        symbol_name=data.get("symbol_name"),
    )


def _answer_from_data(data: dict, investigated_projects: List[str]) -> QAAnswer:
    flow_trace = None
    if data.get("flow_trace"):
        flow_trace = [
            FlowStep(
                order=int(step.get("order", i + 1)),
                description=str(step.get("description", "")),
                file_path=str(step.get("file_path", "")),
                citation=_parse_citation(step["citation"]) if step.get("citation") else None,
            )
            for i, step in enumerate(data["flow_trace"])
        ]

    evidence = [_parse_citation(c) for c in (data.get("evidence") or [])]

    confidence = data.get("confidence")
    if confidence not in ("direct_evidence", "inferred", "no_evidence"):
        confidence = "inferred"

    return QAAnswer(
        summary=str(data.get("summary") or ""),
        details=data.get("details"),
        flow_trace=flow_trace,
        evidence=evidence,
        corrected_premise=data.get("corrected_premise"),
        confidence=confidence,
        projects_considered=investigated_projects,
    )


def _mock_answer(evidence_list: List[Evidence], investigated_projects: List[str]) -> QAAnswer:
    """Deterministic answer used in test/mock environments (no real Gemini
    key configured) -- mirrors retriever.answer_question's own "Simulated
    answer in test environment" convention rather than making a network
    call."""
    citations = [
        CitationRef(file_path=c.file_path, start_line=c.start_line, end_line=c.end_line, symbol_name=c.symbol_name)
        for e in evidence_list
        for c in e.chunks
    ]
    return QAAnswer(
        summary="Based on the gathered repository evidence, here is the explanation for this question.",
        evidence=citations,
        confidence="inferred",
        projects_considered=investigated_projects,
    )
