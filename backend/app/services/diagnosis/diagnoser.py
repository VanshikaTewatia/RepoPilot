"""Evidence-driven root-cause diagnosis synthesis (Phase 6A).

Turns already-gathered ``retrieved_context`` (and, on a retry, the current
``error_analysis``) into a structured ``Diagnosis`` via a single Gemini
call. Mirrors the exact Gemini-call/test-mode/deterministic-override
conventions already established by ``app.services.qa.answerer`` and
``app.services.agent.graph._generate_patches_with_gemini`` -- see this
module's functions below for the direct analogues.

Never fabricates:
  - No retrieved context at all -> ``INSUFFICIENT_EVIDENCE`` returned
    immediately, WITHOUT ever calling Gemini (mirrors
    ``qa.answerer.no_evidence_answer``).
  - Every citation Gemini returns is checked against the real retrieved
    file paths; anything else is dropped (see ``diagnosis_validator``).
  - A "direct_evidence" verdict is downgraded to "inferred" when the
    gathered evidence is objectively thin (see ``diagnosis_validator``).
  - Any Gemini/network/JSON failure produces ``DIAGNOSIS_FAILED`` with a
    ``failure_reason`` -- never silently reinterpreted as
    ``INSUFFICIENT_EVIDENCE`` (a process failure is not a verdict).

Deliberately has NO dependency on reproduction, Docker, workspace
creation, or GitHub functionality: diagnosis works purely from the
``retrieved_context``/``error_analysis`` values already in ``AgentState``,
exactly like ``_generate_patches_with_gemini`` already does.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from google import genai

from app.core.config import settings
from app.core.logging import logger
from app.services.qa.json_utils import parse_json_object
from app.services.qa.models import CitationRef

from .diagnosis_validator import validate_diagnosis
from .models import Diagnosis, DiagnosisStatus, RootCauseHypothesis

NO_EVIDENCE_SUMMARY = "There is no retrieved evidence to diagnose a root cause from."

_DIAGNOSIS_SYSTEM_INSTRUCTION = (
    "You are RepoPilot, an expert AI software engineer investigating a real "
    "repository to diagnose the ROOT CAUSE of a reported bug before a fix is "
    "attempted. Follow these rules strictly:\n"
    "1. The supplied Context Code Files are authoritative -- they are the "
    "actual current contents of files in this repository.\n"
    "2. Answer ONLY from the supplied context. Do not invent files, symbols, "
    "or behavior that are not present in it.\n"
    "3. If a previous attempt's test failure trace is supplied, use it as "
    "additional evidence about what is actually going wrong.\n"
    "4. Produce one or more ranked root-cause hypotheses (rank 1 = most "
    "likely). Each hypothesis must cite specific files/line ranges from the "
    "context that support it.\n"
    "5. If the evidence is insufficient to form any real hypothesis, return "
    'an empty "hypotheses" array and set confidence to "no_evidence" rather '
    "than guessing.\n"
    "6. Do not propose a patch or write code -- only diagnose the cause and, "
    'optionally, a brief "suggested_fix_approach" description per hypothesis.\n'
    "\n"
    "Return ONLY a JSON object, no markdown fences, no commentary. Schema:\n"
    "{\n"
    '  "summary": "1-3 sentence overall diagnosis summary",\n'
    '  "hypotheses": [{"rank": 1, "description": "...", "citations": '
    '[{"file_path": "...", "start_line": 1, "end_line": 10, "symbol_name": '
    '"..." or null}], "suggested_fix_approach": "..." or null}],\n'
    '  "confidence": "direct_evidence" | "inferred" | "no_evidence"\n'
    "}"
)


def _gemini_unavailable() -> bool:
    key = settings.gemini_api_key
    return not key or key.startswith("test") or key.startswith("mock")


def insufficient_evidence_diagnosis() -> Diagnosis:
    """The deterministic result for a no-evidence case -- never produced by
    asking Gemini to diagnose an empty context (mirrors
    ``qa.answerer.no_evidence_answer``)."""
    return Diagnosis(
        status=DiagnosisStatus.INSUFFICIENT_EVIDENCE,
        summary=NO_EVIDENCE_SUMMARY,
        hypotheses=[],
        confidence="no_evidence",
    )


def _format_retrieved_context(retrieved_context: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for item in retrieved_context:
        fpath = item.get("file_path", "")
        content = item.get("content", "")
        total_lines = item.get("total_lines", 0)
        parts.append(f"### File: {fpath} ({total_lines} lines total)\n```\n{content}\n```")
    return "\n\n".join(parts)


def _build_diagnosis_prompt(
    task_description: str,
    retrieved_context: List[Dict[str, Any]],
    error_analysis: Optional[str],
) -> str:
    prompt = (
        f"Task Description:\n{task_description}\n\n"
        f"Context Code Files:\n{_format_retrieved_context(retrieved_context)}\n\n"
    )
    if error_analysis:
        prompt += f"Previous Attempt Test Failure Trace:\n{error_analysis}\n\n"
    prompt += "Produce the JSON diagnosis now:"
    return prompt


def _real_evidence_files(retrieved_context: List[Dict[str, Any]]) -> Set[str]:
    return {item.get("file_path") for item in retrieved_context if item.get("file_path")}


def _parse_citation(data: dict) -> CitationRef:
    return CitationRef(
        file_path=str(data.get("file_path", "")),
        start_line=int(data.get("start_line", 1)),
        end_line=int(data.get("end_line", data.get("start_line", 1))),
        symbol_name=data.get("symbol_name"),
    )


def _diagnosis_from_data(data: dict) -> Diagnosis:
    confidence = data.get("confidence")
    if confidence not in ("direct_evidence", "inferred", "no_evidence"):
        confidence = "inferred"

    hypotheses: List[RootCauseHypothesis] = []
    for i, h in enumerate(data.get("hypotheses") or []):
        hypotheses.append(
            RootCauseHypothesis(
                rank=int(h.get("rank", i + 1)),
                description=str(h.get("description", "")),
                citations=[_parse_citation(c) for c in (h.get("citations") or [])],
                suggested_fix_approach=h.get("suggested_fix_approach"),
            )
        )

    return Diagnosis(
        status=DiagnosisStatus.DIAGNOSED,
        summary=str(data.get("summary") or ""),
        hypotheses=hypotheses,
        confidence=confidence,
    )


def _mock_diagnosis(retrieved_context: List[Dict[str, Any]]) -> Diagnosis:
    """Deterministic diagnosis used in test/dev environments (no real Gemini
    key configured) -- mirrors ``qa.answerer._mock_answer``'s convention of a
    deterministic placeholder rather than a network call."""
    first = retrieved_context[0]
    citation = CitationRef(file_path=first.get("file_path", ""), start_line=1, end_line=1)
    return Diagnosis(
        status=DiagnosisStatus.DIAGNOSED,
        summary="Based on the retrieved code, here is a candidate root-cause diagnosis.",
        hypotheses=[
            RootCauseHypothesis(
                rank=1,
                description="The retrieved code is the most likely location of the reported issue.",
                citations=[citation],
                suggested_fix_approach=None,
            )
        ],
        confidence="inferred",
    )


async def diagnose(
    task_description: str,
    retrieved_context: List[Dict[str, Any]],
    error_analysis: Optional[str] = None,
) -> Diagnosis:
    """Synthesize a structured, cited ``Diagnosis`` from already-gathered
    ``retrieved_context``.

    Never calls Gemini when there's no context to diagnose from (returns
    ``insufficient_evidence_diagnosis()`` immediately). Any Gemini/parsing
    failure returns ``DIAGNOSIS_FAILED`` with ``failure_reason`` set, never
    raised -- this function must never crash ``diagnose_node``. The result
    is always passed through ``validate_diagnosis`` before being returned,
    so callers never see an un-filtered hallucinated citation or an
    unjustified "direct_evidence" confidence.
    """
    if not retrieved_context:
        return insufficient_evidence_diagnosis()

    real_files = _real_evidence_files(retrieved_context)
    total_evidence_pieces = len(retrieved_context)

    if _gemini_unavailable():
        raw = _mock_diagnosis(retrieved_context)
    else:
        prompt = _build_diagnosis_prompt(task_description, retrieved_context, error_analysis)
        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model_name,
                contents=prompt,
                config={"system_instruction": _DIAGNOSIS_SYSTEM_INSTRUCTION},
            )
            raw_text = response.text or ""
            data = parse_json_object(raw_text)
            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
            raw = _diagnosis_from_data(data)
        except Exception as e:  # noqa: BLE001 -- any failure degrades to DIAGNOSIS_FAILED, never crashes
            logger.error(f"Error generating diagnosis with Gemini: {e}")
            return Diagnosis(
                status=DiagnosisStatus.DIAGNOSIS_FAILED,
                confidence="no_evidence",
                failure_reason=str(e),
            )

    result = validate_diagnosis(raw, real_files, total_evidence_pieces)
    return result.diagnosis
