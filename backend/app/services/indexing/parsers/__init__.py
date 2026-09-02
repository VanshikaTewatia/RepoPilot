"""Code parsers package."""

from typing import Dict, Type
from app.services.indexing.parsers.base import LanguageParser, ParsedSymbol
from app.services.indexing.parsers.javascript_parser import JavaScriptParser
from app.services.indexing.parsers.python_parser import PythonParser

# Registry mapping language identifier to parser class. Only languages with
# a real, dedicated structural parser are listed here -- every other
# language identifier (go, rust, java, ...) intentionally still falls back
# to PythonParser in get_parser_for_language() below, exactly as before;
# that's a pre-existing, separate limitation, not something this registry
# widens speculatively.
PARSER_REGISTRY: Dict[str, Type[LanguageParser]] = {
    "python": PythonParser,
    "javascript": JavaScriptParser,
    "javascriptreact": JavaScriptParser,
    "typescript": JavaScriptParser,
    "typescriptreact": JavaScriptParser,
}


def get_parser_for_language(language: str) -> LanguageParser:
    """Retrieve an initialized parser for the specified language."""
    parser_cls = PARSER_REGISTRY.get(language.lower())
    if parser_cls is JavaScriptParser:
        return JavaScriptParser(language=language.lower())
    if parser_cls:
        return parser_cls()
    # Default to PythonParser fallback or generic block parser
    return PythonParser()


__all__ = [
    "LanguageParser",
    "ParsedSymbol",
    "PythonParser",
    "JavaScriptParser",
    "get_parser_for_language",
]
