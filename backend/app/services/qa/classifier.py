"""Question classification for Deep Codebase Q&A.

Maps a free-text question to a ``QuestionClass`` (kind + investigation
depth + search hints) via a single, lightweight Gemini structured-output
call -- reusing the exact Gemini client/call convention already used for
the agent's own structured completions (see
app.services.agent.graph._generate_patches_with_gemini): a plain
``genai.Client(...).models.generate_content(...)`` call with a JSON-only
system instruction, parsed tolerantly, and a real/test/mock key check
performed *before* ever attempting the call (never a network call with an
empty or placeholder key).

The user's own wording is never treated as fact. ``user_asserted_tech`` is
captured verbatim from the question for the *investigator*/*answerer* to
later verify (or correct) against real repository evidence -- this module
never validates it, never uses it to decide what actually exists.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from google import genai

from app.core.config import settings
from app.core.logging import logger
from app.services.qa.investigator import VALID_DEPTHS, _question_terms
from app.services.qa.json_utils import parse_json_object

QuestionKind = Literal["lookup", "symbol", "flow", "architecture", "existence_check"]
QuestionDepth = Literal["shallow", "targeted", "medium", "deep"]

# Never fall back to "shallow": an under-classified architecture/flow
# question silently getting a shallow (RAG-only) investigation is worse
# than the extra cost of a safer, deeper default. "targeted" is the
# smallest depth that still inspects real files and runs a search, not
# just RAG alone.
FALLBACK_DEPTH: QuestionDepth = "targeted"


class QuestionClass(BaseModel):
    """A question's classification: what kind of question it is, how deep
    an investigation it warrants, and what it's actually about.

    ``classification_failed``/``failure_reason`` are set only by the safe
    fallback path (Gemini unavailable, quota-limited, or returned output
    that couldn't be parsed into a valid classification) -- when True,
    ``kind`` is a placeholder, not a real classification; callers that care
    about the failure should check ``classification_failed``, not infer it
    from ``kind``.
    """

    kind: QuestionKind
    depth: QuestionDepth
    subject_terms: List[str] = Field(default_factory=list)
    user_asserted_tech: List[str] = Field(default_factory=list)
    likely_multi_file: bool = False
    classification_failed: bool = False
    failure_reason: Optional[str] = None


_CLASSIFIER_SYSTEM_INSTRUCTION = (
    "You are RepoPilot's question classifier. Given a user's question about "
    "a code repository, classify it. Return ONLY a JSON object, no markdown "
    "fences, no commentary. Schema:\n"
    '{"kind": "lookup"|"symbol"|"flow"|"architecture"|"existence_check", '
    '"depth": "shallow"|"targeted"|"medium"|"deep", '
    '"subject_terms": [string, ...], '
    '"user_asserted_tech": [string, ...], '
    '"likely_multi_file": boolean}\n'
    "\n"
    "kind guide:\n"
    '- lookup: a single concrete fact ("where is X calculated")\n'
    "- symbol: about one specific named function/class/component\n"
    '- flow: a process spanning multiple files/steps ("how does X flow work")\n'
    '- architecture: a broad design/system question ("explain how X works overall")\n'
    "- existence_check: asks whether something exists at all\n"
    "\n"
    "depth guide: lookup->shallow, symbol->targeted, flow->medium or deep, "
    "architecture->deep, existence_check->targeted or deep depending on how "
    "much investigation would be needed to answer confidently.\n"
    "\n"
    "user_asserted_tech: list ONLY the specific technology/framework/language "
    'names the question assumes or names (e.g. "React", "JWT"). This is the '
    "user's own hypothesis, not a verified fact about the repository -- "
    "extract it verbatim, do not validate or correct it.\n"
    "subject_terms: the concrete nouns/identifiers the question is actually "
    'about (e.g. "cart", "subtotal", "calculateSubtotal"), for search.'
)


def _build_classifier_prompt(question: str) -> str:
    return f'Classify this question:\n"{question}"\n\nReturn the JSON object now:'


def _gemini_unavailable() -> bool:
    """True when there's no real Gemini key to call -- mirrors
    _generate_patches_with_gemini's exact test/mock/empty-key check."""
    key = settings.gemini_api_key
    return not key or key.startswith("test") or key.startswith("mock")


# Fixed, curated alias -> canonical-name vocabulary used ONLY by
# _extract_asserted_tech_terms() below (the classification-failure
# fallback). Deliberately small and explicit -- never inferred from what
# the repository actually contains, only from the question's own wording
# -- so a classification failure still leaves _compute_corrected_premise
# (app.services.qa.answerer) something real to compare against.
_TECH_ALIASES: Dict[str, str] = {
    "react": "React", "reactjs": "React", "react.js": "React",
    "vue": "Vue", "vuejs": "Vue", "vue.js": "Vue",
    "angular": "Angular", "angularjs": "Angular",
    "next.js": "Next.js", "nextjs": "Next.js",
    "vite": "Vite",
    "svelte": "Svelte",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "python": "Python",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "java": "Java",
    "spring": "Spring", "springboot": "Spring", "spring boot": "Spring",
    "node.js": "Node.js", "nodejs": "Node.js", "node": "Node.js",
    "go": "Go", "golang": "Go",
    "rust": "Rust",
    ".net": ".NET", "dotnet": ".NET", "dot net": ".NET",
    "c#": "C#", "csharp": "C#", "c sharp": "C#",
    "flutter": "Flutter",
    "dart": "Dart",
}


def _alias_present(alias: str, text_lower: str) -> bool:
    """Whole-word/phrase boundary check for ``alias`` inside ``text_lower``
    (already lowercased). Boundaries are based on "is the neighboring
    character alphanumeric", not regex ``\\b``, so aliases containing
    punctuation (".net", "c#", "next.js") are handled correctly and "java"
    never matches inside "javascript"."""
    start = 0
    while True:
        idx = text_lower.find(alias, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not text_lower[idx - 1].isalnum()
        end = idx + len(alias)
        after_ok = end >= len(text_lower) or not text_lower[end].isalnum()
        if before_ok and after_ok:
            return True
        start = idx + 1


def _extract_asserted_tech_terms(question: str) -> List[str]:
    """Best-effort, deterministic technology/framework term extraction from
    a question's own text -- used ONLY when real (Gemini) classification
    has failed, as a substitute for what it would have extracted, so the
    deterministic corrected_premise check still has an explicit premise to
    compare against real repository evidence.

    Recognizes a small, fixed vocabulary (never inferred from what the
    repository actually contains -- only terms the question itself names),
    case-insensitively, with common aliases collapsed to one canonical name
    (e.g. "react"/"reactjs"/"react.js" -> "React", "node"/"nodejs"/
    "node.js" -> "Node.js", ".net"/"dotnet" -> ".NET").

    Deliberately simple: this is a fallback for when real classification
    already failed, not a replacement for it -- a handful of the
    vocabulary's short words (e.g. "Go", "Dart", "Rust") are also ordinary
    English words and can false-positive on an unrelated sentence. That
    trade-off is acceptable here specifically because Gemini's real
    language understanding is the primary path and only unavailable in
    this fallback.
    """
    if not question:
        return []
    text_lower = question.lower()
    found: List[str] = []
    for alias, canonical in _TECH_ALIASES.items():
        if canonical in found:
            continue
        if _alias_present(alias, text_lower):
            found.append(canonical)
    return found


def _fallback_question_class(question: str, reason: str) -> QuestionClass:
    return QuestionClass(
        kind="lookup",
        depth=FALLBACK_DEPTH,
        subject_terms=_question_terms(question or "", limit=5),
        user_asserted_tech=_extract_asserted_tech_terms(question or ""),
        likely_multi_file=False,
        classification_failed=True,
        failure_reason=reason,
    )


async def classify_question(question: str) -> QuestionClass:
    """Classify a question via a lightweight Gemini structured-output call.

    Falls back to a safe, explicit classification (``FALLBACK_DEPTH``,
    ``classification_failed=True``) whenever Gemini is unavailable, quota-
    limited, or returns output that can't be parsed into a valid
    ``QuestionClass`` -- never raises, and never silently under-classifies
    toward "shallow".
    """
    if not question or not question.strip():
        return _fallback_question_class(question, "Empty question")

    if _gemini_unavailable():
        return _fallback_question_class(question, "Gemini not configured (missing/test/mock API key)")

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=settings.gemini_model_name,
            contents=_build_classifier_prompt(question),
            config={"system_instruction": _CLASSIFIER_SYSTEM_INSTRUCTION},
        )
        raw_text = response.text or ""
        data = parse_json_object(raw_text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got {type(data).__name__}")

        return QuestionClass(
            kind=data["kind"],
            depth=data["depth"],
            subject_terms=[str(t) for t in (data.get("subject_terms") or [])],
            user_asserted_tech=[str(t) for t in (data.get("user_asserted_tech") or [])],
            likely_multi_file=bool(data.get("likely_multi_file", False)),
            classification_failed=False,
        )
    except Exception as e:  # noqa: BLE001 -- any failure (network, quota, malformed JSON, invalid schema) falls back safely
        logger.warning(f"Question classification failed, falling back to {FALLBACK_DEPTH} depth: {e}")
        return _fallback_question_class(question, str(e))
