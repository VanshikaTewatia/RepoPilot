"""JavaScript/JSX/TypeScript/TSX syntax parser using Tree-sitter.

Covers the four extensions the file-discovery layer maps to these language
identifiers (see app.services.indexing.file_discovery.EXTENSION_LANGUAGE_MAP):
"javascript" (.js), "javascriptreact" (.jsx), "typescript" (.ts), and
"typescriptreact" (.tsx). Never routes through PythonParser -- before this
module existed, get_parser_for_language() fell back to PythonParser for
every one of these, whose ast.parse()-based extraction can't understand JS/
TS syntax at all; depending on exactly what tree-sitter's error-tolerant
Python grammar happened to misrecognize, a real component file could index
as almost nothing (e.g. a single "import React from 'react';" chunk with
the rest of the file's content silently discarded) -- this is what caused
Deep Q&A to miss real authentication components entirely.

Extracts the dominant JS/TS/React authoring patterns as top-level symbols:
  - grouped imports (mirrors PythonParser's own "imports" block)
  - function declarations (`function Foo() {}`)
  - class declarations (`class Foo {}` / `class Foo extends Bar {}`)
  - `const/let/var Foo = (...) => {...}` and `= function () {...}`
    assignments -- the dominant React functional-component and hook style
  - TypeScript `interface`/`type` declarations
  - `export`/`export default` wrappers are unwrapped transparently so the
    underlying declaration is still extracted as its real symbol type
A PascalCase-named function/arrow-function is tagged symbol_type
"component" (a plain heuristic -- React's own naming convention -- not
framework detection) rather than "function", so component lookups can
filter on it later if useful.

Falls back to a single whole-file block (never PythonParser) when
tree-sitter bindings aren't available or a file has no structural symbols
tree-sitter recognizes.
"""

from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.services.indexing.parsers.base import LanguageParser, ParsedSymbol

_TREESITTER_JS_AVAILABLE = False
try:
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_typescript as tstypescript
    from tree_sitter import Language, Parser

    _JS_LANGUAGE = Language(tsjavascript.language())
    _TS_LANGUAGE = Language(tstypescript.language_typescript())
    _TSX_LANGUAGE = Language(tstypescript.language_tsx())
    _TREESITTER_JS_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised only when bindings are missing
    logger.debug(f"Tree-sitter JS/TS bindings not initialized ({e}); JS/TS files will index as whole-file blocks.")
    _JS_LANGUAGE = _TS_LANGUAGE = _TSX_LANGUAGE = None

# One grammar per language identifier -- JSX is natively part of the JS
# grammar (no separate "javascriptreact" grammar exists or is needed), but
# TSX needs its own grammar variant to disambiguate `<Foo>` generics from
# JSX syntax.
_GRAMMAR_BY_LANGUAGE: Dict[str, Any] = {
    "javascript": lambda: _JS_LANGUAGE,
    "javascriptreact": lambda: _JS_LANGUAGE,
    "typescript": lambda: _TS_LANGUAGE,
    "typescriptreact": lambda: _TSX_LANGUAGE,
}

_FUNCTION_VALUE_NODE_TYPES = ("arrow_function", "function_expression", "function")
_DECLARATION_LIST_NODE_TYPES = ("lexical_declaration", "variable_declaration")
_TYPE_DECLARATION_NODE_TYPES = ("interface_declaration", "type_alias_declaration")


class JavaScriptParser(LanguageParser):
    """Syntax-aware parser for JavaScript/JSX/TypeScript/TSX source code."""

    def __init__(self, language: str = "javascript"):
        if language not in _GRAMMAR_BY_LANGUAGE:
            raise ValueError(f"Unsupported language for JavaScriptParser: {language!r}")
        self._language_key = language

    @property
    def language(self) -> str:
        return self._language_key

    def parse(self, source_code: str, file_path: str = "") -> List[ParsedSymbol]:
        """Extract syntax-aware symbols from JS/JSX/TS/TSX source code."""
        if not source_code.strip():
            return []

        grammar = _GRAMMAR_BY_LANGUAGE[self._language_key]()
        if _TREESITTER_JS_AVAILABLE and grammar is not None:
            try:
                symbols = self._parse_treesitter(source_code, file_path, grammar)
                if symbols:
                    return symbols
            except Exception as e:
                logger.warning(f"Tree-sitter JS/TS parse failed for {file_path}: {e}. Falling back to whole-file block.")

        lines = source_code.splitlines(keepends=True)
        return [
            ParsedSymbol(
                name="raw_file",
                symbol_type="block",
                start_line=1,
                end_line=len(lines) or 1,
                source_code=source_code,
                metadata={"file": file_path, "parser_unavailable": not _TREESITTER_JS_AVAILABLE},
            )
        ]

    def _parse_treesitter(self, source_code: str, file_path: str, grammar) -> List[ParsedSymbol]:
        parser = Parser(grammar)
        source_bytes = source_code.encode("utf-8")
        tree = parser.parse(source_bytes)
        root_node = tree.root_node
        lines = source_code.splitlines(keepends=True)
        symbols: List[ParsedSymbol] = []

        def node_name(node) -> Optional[str]:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return None
            return source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")

        def node_text(node) -> str:
            start_l = node.start_point[0] + 1
            end_l = node.end_point[0] + 1
            return start_l, end_l, "".join(lines[start_l - 1 : end_l])

        def unwrap_export(node):
            """`export`/`export default` wraps the real declaration -- unwrap
            it so e.g. `export function Foo() {}` is still extracted as a
            function, not skipped because the top-level node is
            "export_statement"."""
            if node.type != "export_statement":
                return node
            for child in node.children:
                if child.type not in ("export", "default", ";"):
                    return child
            return node

        # Grouped imports, mirroring PythonParser's own "imports" block.
        import_nodes = [c for c in root_node.children if c.type == "import_statement"]
        if import_nodes:
            first_line = import_nodes[0].start_point[0] + 1
            last_line = import_nodes[-1].end_point[0] + 1
            import_code = "".join(lines[first_line - 1 : last_line])
            symbols.append(
                ParsedSymbol(
                    name="imports",
                    symbol_type="import",
                    start_line=first_line,
                    end_line=last_line,
                    source_code=import_code,
                    metadata={"count": len(import_nodes), "file": file_path},
                )
            )

        for child in root_node.children:
            node = unwrap_export(child)

            if node.type == "function_declaration":
                name = node_name(node) or "anonymous_function"
                start_l, end_l, code = node_text(node)
                symbol_type = "component" if name[:1].isupper() else "function"
                symbols.append(
                    ParsedSymbol(
                        name=name, symbol_type=symbol_type, start_line=start_l, end_line=end_l,
                        source_code=code, metadata={"file": file_path},
                    )
                )

            elif node.type == "class_declaration":
                name = node_name(node) or "AnonymousClass"
                start_l, end_l, code = node_text(node)
                symbols.append(
                    ParsedSymbol(
                        name=name, symbol_type="class", start_line=start_l, end_line=end_l,
                        source_code=code, metadata={"file": file_path},
                    )
                )

            elif node.type in _TYPE_DECLARATION_NODE_TYPES:
                name = node_name(node) or "AnonymousType"
                start_l, end_l, code = node_text(node)
                symbols.append(
                    ParsedSymbol(
                        name=name, symbol_type="type", start_line=start_l, end_line=end_l,
                        source_code=code, metadata={"file": file_path},
                    )
                )

            elif node.type in _DECLARATION_LIST_NODE_TYPES:
                # `const Foo = () => {...}` / `const Foo = function () {...}`
                # -- the whole declaration statement (not just the arrow
                # function expression) is the chunk, matching how the
                # symbol actually reads in the file.
                for declarator in node.children:
                    if declarator.type != "variable_declarator":
                        continue
                    value = declarator.child_by_field_name("value")
                    if value is None or value.type not in _FUNCTION_VALUE_NODE_TYPES:
                        continue
                    name = node_name(declarator) or "anonymous"
                    start_l, end_l, code = node_text(node)
                    symbol_type = "component" if name[:1].isupper() else "function"
                    symbols.append(
                        ParsedSymbol(
                            name=name, symbol_type=symbol_type, start_line=start_l, end_line=end_l,
                            source_code=code, metadata={"file": file_path},
                        )
                    )

        if not symbols:
            symbols.append(
                ParsedSymbol(
                    name="module",
                    symbol_type="block",
                    start_line=1,
                    end_line=len(lines) or 1,
                    source_code=source_code,
                    metadata={"file": file_path},
                )
            )

        return symbols
