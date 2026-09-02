"""Unit tests for JavaScriptParser (app.services.indexing.parsers.javascript_parser)
and its registration in the parser registry.

This is the fix for the root cause behind Deep Q&A missing real
authentication components (see backend/tests/test_qa_investigator.py and
the Ecommerce_Frontend validation): before this parser existed, every
.js/.jsx/.ts/.tsx file silently routed through PythonParser, whose
ast.parse()-based extraction can't understand JS/TS syntax and (depending
on what tree-sitter's error-tolerant Python grammar happened to
misrecognize) could reduce a real component file to a single one-line
"imports" chunk, discarding the rest of its content.
"""

from app.services.indexing.parsers import PARSER_REGISTRY, get_parser_for_language
from app.services.indexing.parsers.javascript_parser import JavaScriptParser
from app.services.indexing.parsers.python_parser import PythonParser


def _names_and_types(symbols):
    return [(s.symbol_type, s.name) for s in symbols]


# ---------------------------------------------------------------------------
# Registry wiring: JS/JSX/TS/TSX never route through PythonParser
# ---------------------------------------------------------------------------
def test_registry_maps_js_ts_extensions_to_javascript_parser():
    for lang in ("javascript", "javascriptreact", "typescript", "typescriptreact"):
        parser = get_parser_for_language(lang)
        assert isinstance(parser, JavaScriptParser)
        assert not isinstance(parser, PythonParser)
        assert parser.language == lang


def test_existing_python_behavior_remains_intact():
    """Regression guard: Python parsing must be completely unaffected."""
    parser = get_parser_for_language("python")
    assert isinstance(parser, PythonParser)

    src = "def add(a, b):\n    return a + b\n"
    symbols = parser.parse(src, file_path="math.py")
    assert _names_and_types(symbols) == [("function", "add")]
    assert symbols[0].start_line == 1
    assert symbols[0].end_line == 2


def test_unknown_language_still_falls_back_to_python_parser():
    """A language with no dedicated parser (e.g. Go) keeps its pre-existing
    fallback behavior -- only JS/TS/JSX/TSX were fixed here, deliberately
    not every language."""
    parser = get_parser_for_language("go")
    assert isinstance(parser, PythonParser)


# ---------------------------------------------------------------------------
# JS / JSX / TS / TSX structural extraction
# ---------------------------------------------------------------------------
def test_js_function_declaration_and_class():
    src = (
        "import axios from 'axios';\n"
        "\n"
        "function fetchUser(id) {\n"
        "  return axios.get(`/users/${id}`);\n"
        "}\n"
        "\n"
        "class ApiClient {\n"
        "  get(url) { return axios.get(url); }\n"
        "}\n"
    )
    parser = JavaScriptParser(language="javascript")
    symbols = parser.parse(src, file_path="src/api.js")

    assert ("import", "imports") in _names_and_types(symbols)
    assert ("function", "fetchUser") in _names_and_types(symbols)
    assert ("class", "ApiClient") in _names_and_types(symbols)

    fn = next(s for s in symbols if s.name == "fetchUser")
    assert fn.start_line == 3
    assert fn.end_line == 5
    assert "axios.get" in fn.source_code


def test_jsx_react_component_detected_as_component_symbol():
    """A PascalCase const-arrow-function component (the dominant React
    authoring style) must be extracted as a full, named symbol -- not
    collapsed into an opaque whole-file block."""
    src = (
        "import React from 'react';\n"
        "\n"
        "const CartItem = ({ item, onRemove }) => {\n"
        "  const handleRemove = () => onRemove(item.id);\n"
        "  return (\n"
        "    <div className=\"cart-item\">\n"
        "      <span>{item.name}</span>\n"
        "      <button onClick={handleRemove}>Remove</button>\n"
        "    </div>\n"
        "  );\n"
        "};\n"
        "\n"
        "export default CartItem;\n"
    )
    parser = JavaScriptParser(language="javascriptreact")
    symbols = parser.parse(src, file_path="src/components/cart/CartItem.jsx")

    component = next(s for s in symbols if s.name == "CartItem")
    assert component.symbol_type == "component"
    assert component.start_line == 3
    assert component.end_line == 11
    assert "handleRemove" in component.source_code
    assert "cart-item" in component.source_code


def test_ts_typed_function_and_interface():
    src = (
        "export interface User {\n"
        "  id: string;\n"
        "  email: string;\n"
        "}\n"
        "\n"
        "export function getUser(id: string): User {\n"
        "  return { id, email: '' };\n"
        "}\n"
    )
    parser = JavaScriptParser(language="typescript")
    symbols = parser.parse(src, file_path="src/user.ts")

    assert ("type", "User") in _names_and_types(symbols)
    assert ("function", "getUser") in _names_and_types(symbols)


def test_tsx_typed_react_component_and_hook():
    src = (
        "import React, { useContext } from 'react';\n"
        "import { AuthContext } from './AuthContext';\n"
        "\n"
        "interface GreetingProps {\n"
        "  name: string;\n"
        "}\n"
        "\n"
        "export const Greeting: React.FC<GreetingProps> = ({ name }) => {\n"
        "  return <div>Hello {name}</div>;\n"
        "};\n"
        "\n"
        "export function useAuth() {\n"
        "  return useContext(AuthContext);\n"
        "}\n"
    )
    parser = JavaScriptParser(language="typescriptreact")
    symbols = parser.parse(src, file_path="src/Greeting.tsx")

    kinds = _names_and_types(symbols)
    assert ("type", "GreetingProps") in kinds
    assert ("component", "Greeting") in kinds
    assert ("function", "useAuth") in kinds


def test_line_ranges_are_accurate_across_multiple_symbols():
    src = (
        "import React from 'react';\n"  # line 1
        "\n"                              # line 2
        "function first() {\n"            # line 3
        "  return 1;\n"                   # line 4
        "}\n"                             # line 5
        "\n"                              # line 6
        "function second() {\n"           # line 7
        "  return 2;\n"                   # line 8
        "}\n"                             # line 9
    )
    parser = JavaScriptParser(language="javascript")
    symbols = parser.parse(src, file_path="src/two.js")

    first = next(s for s in symbols if s.name == "first")
    second = next(s for s in symbols if s.name == "second")
    assert (first.start_line, first.end_line) == (3, 5)
    assert (second.start_line, second.end_line) == (7, 9)


def test_content_hash_reflects_symbol_boundaries_not_whole_file():
    """Two different symbols in the same file must hash differently --
    proves chunk boundaries (and therefore embedding granularity) are real,
    not a single whole-file blob."""
    src = "function a() {\n  return 1;\n}\n\nfunction b() {\n  return 2;\n}\n"
    parser = JavaScriptParser(language="javascript")
    symbols = parser.parse(src, file_path="src/two.js")
    hashes = {s.content_hash for s in symbols if s.symbol_type == "function"}
    assert len(hashes) == 2


# ---------------------------------------------------------------------------
# Fallback behavior (never routes through PythonParser)
# ---------------------------------------------------------------------------
def test_empty_file_returns_no_symbols():
    parser = JavaScriptParser(language="javascript")
    assert parser.parse("", file_path="empty.js") == []
    assert parser.parse("   \n  \n", file_path="empty.js") == []


def test_file_with_no_structural_symbols_falls_back_to_whole_file_block():
    """A file that's valid JS but has nothing tree-sitter recognizes as a
    top-level declaration (e.g. bare statements) still gets SOME chunk, not
    silently nothing -- and it's tagged as a whole-file "module" block, not
    routed through PythonParser."""
    src = "console.log('side effect only, no declarations');\n"
    parser = JavaScriptParser(language="javascript")
    symbols = parser.parse(src, file_path="src/sideeffect.js")

    assert len(symbols) == 1
    assert symbols[0].symbol_type == "block"
    assert symbols[0].name == "module"
    assert "console.log" in symbols[0].source_code


def test_invalid_language_key_rejected():
    import pytest

    with pytest.raises(ValueError):
        JavaScriptParser(language="rust")
