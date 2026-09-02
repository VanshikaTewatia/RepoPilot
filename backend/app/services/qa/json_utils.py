"""Shared helper for parsing a JSON object out of a Gemini text completion.

Gemini is always asked to return raw JSON with no markdown fences, but that
isn't always honored -- this mirrors the same tolerant-parsing approach
already used by app.services.agent.graph.parse_and_validate_patches for the
agent's own structured Gemini calls (strip an accidental code fence, then
fall back to slicing between the outermost braces/brackets if a direct
``json.loads`` fails).
"""

import json
from typing import Any


def strip_code_fence(text: str) -> str:
    """Remove a wrapping ``` ... ``` markdown fence, if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_object(raw_text: str) -> Any:
    """Parse a JSON object (or array) from a Gemini completion.

    Tolerates an accidental markdown fence or stray leading/trailing
    commentary around the JSON payload. Raises ``json.JSONDecodeError`` (or
    ``ValueError`` if no object/array delimiters can be found at all) when
    the text genuinely isn't parseable -- callers are expected to treat that
    as "classification/answer generation failed" and fall back safely, never
    to guess at a partial result.
    """
    cleaned = strip_code_fence(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        obj_start, obj_end = cleaned.find("{"), cleaned.rfind("}")
        arr_start, arr_end = cleaned.find("["), cleaned.rfind("]")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            return json.loads(cleaned[obj_start : obj_end + 1])
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            return json.loads(cleaned[arr_start : arr_end + 1])
        raise
