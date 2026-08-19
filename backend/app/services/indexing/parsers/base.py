"""Base interfaces and data structures for syntax-aware code parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional


@dataclass
class ParsedSymbol:
    """Represents a syntax unit extracted from source code."""

    name: str
    symbol_type: str  # function, class, method, import, block
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    source_code: str
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Calculate SHA-256 hash of the normalized source code."""
        return hashlib.sha256(self.source_code.strip().encode("utf-8")).hexdigest()

    @property
    def line_count(self) -> int:
        """Total lines in this symbol."""
        return self.end_line - self.start_line + 1


class LanguageParser(ABC):
    """Abstract interface for language-specific AST parsers."""

    @property
    @abstractmethod
    def language(self) -> str:
        """Identifier for the programming language."""
        pass

    @abstractmethod
    def parse(self, source_code: str, file_path: str = "") -> List[ParsedSymbol]:
        """Parse source code text and return syntax-aware symbols."""
        pass
