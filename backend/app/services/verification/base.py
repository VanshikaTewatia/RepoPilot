"""Shared types for the generic project verification engine.

Every ecosystem-specific adapter implements ``VerificationAdapter`` and every
verification run -- regardless of which adapter handled it -- is normalized
into a ``VerificationResult`` so callers (the agent graph, the GitHub
approval flow, the API layer) never need ecosystem-specific branching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional


@dataclass
class VerificationResult:
    """Normalized outcome of a verification run, regardless of ecosystem.

    ``available`` is False only when no supported ecosystem could be
    detected, or a detected ecosystem had no runnable verification command
    (e.g. a Node project with neither a "test" nor a "build" script).
    Verification is never silently reported as passed in either case.
    """

    ecosystem: str
    success: bool
    exit_code: int
    output: str
    passed: int
    failed: int
    duration: float
    available: bool = True
    manifests_found: List[str] = field(default_factory=list)
    detail: Optional[str] = None
    command: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize to the plain dict shape existing callers already expect
        (AgentState.test_results / Task.test_output), with ecosystem metadata
        added as extra keys so nothing downstream needs to change."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "output": self.output,
            "passed": self.passed,
            "failed": self.failed,
            "duration": self.duration,
            "ecosystem": self.ecosystem,
            "available": self.available,
            "manifests_found": self.manifests_found,
            "detail": self.detail,
            "command": self.command,
        }


class VerificationAdapter(ABC):
    """Common interface every ecosystem-specific verification adapter implements.

    Adding a new ecosystem means adding one adapter class and registering it
    in ``detector.ADAPTER_PRECEDENCE`` -- no if/else chains grow elsewhere.
    """

    ecosystem: ClassVar[str]
    # Literal manifest filenames this adapter looks for at the workspace root.
    # Adapters that need glob/recursive matching (e.g. .NET's *.csproj) override
    # find_manifests()/detect() instead of relying on this list alone.
    manifest_files: ClassVar[List[str]] = []

    @classmethod
    def find_manifests(cls, workspace: Path) -> List[str]:
        """Return which of this adapter's manifest filenames exist in the workspace."""
        return [name for name in cls.manifest_files if (workspace / name).is_file()]

    @classmethod
    def detect(cls, workspace: Path) -> bool:
        """True if this ecosystem applies to the given workspace."""
        return bool(cls.find_manifests(workspace))

    @abstractmethod
    def install_command(self, workspace: Path) -> Optional[List[str]]:
        """Best-effort dependency install argv, or None if nothing to install.

        A failing install must never prevent the test command from running
        (mirrors the existing Python sandbox's non-fatal install semantics).
        """

    @abstractmethod
    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> Optional[List[str]]:
        """Verification command argv, or None if no runnable command exists.

        Returning None means "verification unavailable for this project as
        currently configured" -- callers must not fall back to guessing.
        """

    @abstractmethod
    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        """Extract {"passed": int, "failed": int} from raw command output."""

    def unavailable_reason(self, workspace: Path) -> Optional[str]:
        """Human-readable reason test_command() returned None. Override when relevant."""
        return None
