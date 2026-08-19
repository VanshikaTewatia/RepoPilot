"""Python syntax parser using Tree-sitter with AST fallback."""

import ast
from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.services.indexing.parsers.base import LanguageParser, ParsedSymbol

# Attempt to load Tree-sitter Python language bindings
_TREESITTER_AVAILABLE = False
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    _parser = Parser(PY_LANGUAGE)
    _TREESITTER_AVAILABLE = True
except Exception as e:
    logger.debug(f"Tree-sitter native binding not initialized ({e}), using Python AST fallback.")


class PythonParser(LanguageParser):
    """Syntax-aware parser for Python source code."""

    @property
    def language(self) -> str:
        return "python"

    def parse(self, source_code: str, file_path: str = "") -> List[ParsedSymbol]:
        """Extract syntax symbols (functions, classes, methods, imports) from Python source code."""
        if not source_code.strip():
            return []

        if _TREESITTER_AVAILABLE:
            try:
                symbols = self._parse_treesitter(source_code, file_path)
                if symbols:
                    return symbols
            except Exception as e:
                logger.warning(f"Tree-sitter parse failed for {file_path}: {e}. Falling back to AST.")

        return self._parse_ast(source_code, file_path)

    def _parse_treesitter(self, source_code: str, file_path: str) -> List[ParsedSymbol]:
        """Parse using Tree-sitter CST."""
        source_bytes = source_code.encode("utf-8")
        tree = _parser.parse(source_bytes)
        root_node = tree.root_node

        symbols: List[ParsedSymbol] = []
        lines = source_code.splitlines(keepends=True)

        # Collect import blocks
        import_nodes = []
        for child in root_node.children:
            if child.type in ("import_statement", "import_from_statement"):
                import_nodes.append(child)

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
                    metadata={"count": len(import_nodes)},
                )
            )

        def walk_node(node: Any, parent_class: Optional[str] = None) -> None:
            for child in node.children:
                if child.type == "class_definition":
                    name_node = child.child_by_field_name("name")
                    class_name = (
                        source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                        if name_node
                        else "AnonymousClass"
                    )
                    start_l = child.start_point[0] + 1
                    end_l = child.end_point[0] + 1
                    chunk_code = "".join(lines[start_l - 1 : end_l])

                    symbols.append(
                        ParsedSymbol(
                            name=class_name,
                            symbol_type="class",
                            start_line=start_l,
                            end_line=end_l,
                            source_code=chunk_code,
                            metadata={"file": file_path},
                        )
                    )
                    # Walk inside class body to extract methods
                    body_node = child.child_by_field_name("body")
                    if body_node:
                        walk_node(body_node, parent_class=class_name)

                elif child.type == "function_definition":
                    name_node = child.child_by_field_name("name")
                    func_name = (
                        source_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8")
                        if name_node
                        else "anonymous_func"
                    )
                    start_l = child.start_point[0] + 1
                    end_l = child.end_point[0] + 1
                    chunk_code = "".join(lines[start_l - 1 : end_l])

                    sym_type = "method" if parent_class else "function"
                    display_name = f"{parent_class}.{func_name}" if parent_class else func_name

                    symbols.append(
                        ParsedSymbol(
                            name=display_name,
                            symbol_type=sym_type,
                            start_line=start_l,
                            end_line=end_l,
                            source_code=chunk_code,
                            parent_symbol=parent_class,
                            metadata={"file": file_path},
                        )
                    )

        walk_node(root_node)

        # If no structured symbols found, create a fallback module block
        if not symbols:
            symbols.append(
                ParsedSymbol(
                    name="module",
                    symbol_type="block",
                    start_line=1,
                    end_line=len(lines),
                    source_code=source_code,
                    metadata={"file": file_path},
                )
            )

        return symbols

    def _parse_ast(self, source_code: str, file_path: str) -> List[ParsedSymbol]:
        """Fallback parser using standard library ast module."""
        lines = source_code.splitlines(keepends=True)
        symbols: List[ParsedSymbol] = []

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.warning(f"AST syntax error in {file_path}: {e}")
            return [
                ParsedSymbol(
                    name="raw_file",
                    symbol_type="block",
                    start_line=1,
                    end_line=len(lines) if lines else 1,
                    source_code=source_code,
                    metadata={"file": file_path, "parse_error": str(e)},
                )
            ]

        # Extract import statements
        import_lines = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append((node.lineno, getattr(node, "end_lineno", node.lineno)))

        if import_lines:
            first_l = min(start for start, _ in import_lines)
            last_l = max(end for _, end in import_lines)
            import_code = "".join(lines[first_l - 1 : last_l])
            symbols.append(
                ParsedSymbol(
                    name="imports",
                    symbol_type="import",
                    start_line=first_l,
                    end_line=last_l,
                    source_code=import_code,
                    metadata={"count": len(import_lines)},
                )
            )

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", node.lineno)
                docstring = ast.get_docstring(node)
                class_code = "".join(lines[start_l - 1 : end_l])

                symbols.append(
                    ParsedSymbol(
                        name=node.name,
                        symbol_type="class",
                        start_line=start_l,
                        end_line=end_l,
                        source_code=class_code,
                        docstring=docstring,
                        metadata={"file": file_path},
                    )
                )

                # Methods
                for subnode in node.body:
                    if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start = subnode.lineno
                        m_end = getattr(subnode, "end_lineno", subnode.lineno)
                        m_code = "".join(lines[m_start - 1 : m_end])
                        symbols.append(
                            ParsedSymbol(
                                name=f"{node.name}.{subnode.name}",
                                symbol_type="method",
                                start_line=m_start,
                                end_line=m_end,
                                source_code=m_code,
                                docstring=ast.get_docstring(subnode),
                                parent_symbol=node.name,
                                metadata={"file": file_path},
                            )
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_l = node.lineno
                end_l = getattr(node, "end_lineno", node.lineno)
                docstring = ast.get_docstring(node)
                func_code = "".join(lines[start_l - 1 : end_l])

                symbols.append(
                    ParsedSymbol(
                        name=node.name,
                        symbol_type="function",
                        start_line=start_l,
                        end_line=end_l,
                        source_code=func_code,
                        docstring=docstring,
                        metadata={"file": file_path},
                    )
                )

        if not symbols:
            symbols.append(
                ParsedSymbol(
                    name="module",
                    symbol_type="block",
                    start_line=1,
                    end_line=len(lines) if lines else 1,
                    source_code=source_code,
                    metadata={"file": file_path},
                )
            )

        return symbols
