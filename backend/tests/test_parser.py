"""Unit tests for syntax-aware Python parser."""

from app.services.indexing.parsers.python_parser import PythonParser

SAMPLE_PYTHON_CODE = """import os
import sys
from typing import List, Optional

class Calculator:
    \"\"\"A simple calculator class.\"\"\"

    def __init__(self, initial: int = 0):
        self.value = initial

    def add(self, x: int) -> int:
        \"\"\"Add x to value.\"\"\"
        self.value += x
        return self.value

def standalone_helper(msg: str) -> str:
    \"\"\"Format helper string.\"\"\"
    return f"Helper: {msg}"
"""


def test_python_parser_extracts_all_symbols():
    """Test parser extracts imports, classes, methods, functions, and line ranges."""
    parser = PythonParser()
    symbols = parser.parse(SAMPLE_PYTHON_CODE, file_path="calculator.py")

    assert len(symbols) >= 4

    symbol_names = [s.name for s in symbols]
    assert "imports" in symbol_names
    assert "Calculator" in symbol_names
    assert "standalone_helper" in symbol_names

    # Check method extraction
    method_symbols = [s for s in symbols if s.symbol_type == "method"]
    assert any("add" in m.name for m in method_symbols)

    # Check line numbers
    for s in symbols:
        assert s.start_line >= 1
        assert s.end_line >= s.start_line
        assert s.content_hash is not None
        assert len(s.content_hash) == 64


def test_python_parser_empty_input():
    """Test parsing empty or whitespace string returns empty list."""
    parser = PythonParser()
    assert parser.parse("") == []
    assert parser.parse("   \n\t  ") == []
