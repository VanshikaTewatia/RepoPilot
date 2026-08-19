"""Code parsers package."""

from typing import Dict, Type
from app.services.indexing.parsers.base import LanguageParser, ParsedSymbol
from app.services.indexing.parsers.python_parser import PythonParser

# Registry mapping language identifier to parser class
PARSER_REGISTRY: Dict[str, Type[LanguageParser]] = {
    "python": PythonParser,
}


def get_parser_for_language(language: str) -> LanguageParser:
    """Retrieve an initialized parser for the specified language."""
    parser_cls = PARSER_REGISTRY.get(language.lower())
    if parser_cls:
        return parser_cls()
    # Default to PythonParser fallback or generic block parser
    return PythonParser()


__all__ = [
    "LanguageParser",
    "ParsedSymbol",
    "PythonParser",
    "get_parser_for_language",
]
