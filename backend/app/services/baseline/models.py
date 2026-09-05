"""Typed models for standalone baseline bug-reproduction (Phase 4A).

These types intentionally carry no dependency on AgentState, the LangGraph
agent graph, Task/task-outcome models, or any API route -- see the package
docstring in ``app.services.baseline.__init__`` for why. Phase 4B is
responsible for wiring this into that surrounding machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class BaselineStatus(str, Enum):
    """Outcome of attempting to reproduce a reported bug.

    UNABLE_TO_REPRODUCE must never be interpreted as NOT_REPRODUCED by any
    caller -- they mean fundamentally different things: the former says
    nothing at all about whether the bug exists, the latter is a positive
    (if inconclusive) observation that the described behavior didn't occur
    under the supplied procedure.
    """

    REPRODUCED = "REPRODUCED"
    NOT_REPRODUCED = "NOT_REPRODUCED"
    UNABLE_TO_REPRODUCE = "UNABLE_TO_REPRODUCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ExitCodeSemantics(str, Enum):
    """How the *last* reproduction command's exit code should be read.

    Phase 4A never hard-codes "exit 0 = no bug" or "exit nonzero = bug" --
    either can be the correct signal depending on what the reported bug
    actually is (a crash that SHOULD exit nonzero vs. a silently-wrong-output
    bug where the process still exits 0). The caller must say which applies,
    via ``ReproductionExpectation``; IGNORE means the exit code alone never
    determines the result (only the output patterns, if any, do).
    """

    ZERO_IS_REPRODUCED = "zero_is_reproduced"
    NONZERO_IS_REPRODUCED = "nonzero_is_reproduced"
    IGNORE = "ignore"


@dataclass
class ReproductionExpectation:
    """Explicit, caller-supplied success condition for one reproduction
    procedure. Never inferred by an LLM in Phase 4A -- classification is a
    deterministic function of these fields plus the executed facts.

    ``reproduced_output_pattern`` / ``not_reproduced_output_pattern`` are
    regular expressions checked (in that order) against the last executed
    command's combined output, independent of its exit code -- this is what
    lets a successful (exit 0) command that explicitly prints an observed
    incorrect result still count as REPRODUCED, and a failing (nonzero)
    command that's merely an unrelated setup hiccup still count as
    NOT_REPRODUCED when nothing in the output actually demonstrates the
    reported bug.
    """

    exit_code_semantics: ExitCodeSemantics = ExitCodeSemantics.IGNORE
    reproduced_output_pattern: Optional[str] = None
    not_reproduced_output_pattern: Optional[str] = None


@dataclass
class ReproductionInput:
    """Explicit specification of how to attempt reproducing a reported bug.

    ``commands`` is a list of argv lists executed in order inside the same
    sandbox used for verification. An empty/omitted list means "no
    executable reproduction procedure was supplied" -- Phase 4A never
    invents one (e.g. it will not assume ``pytest``/``npm test`` are a
    reproduction of a specific reported bug just because they exist).

    Only environment/execution failures (missing toolchain, timeout, an
    unusable sandbox) abort the sequence early; a command that simply exits
    nonzero for an ordinary reason does not, since intermediate setup steps
    failing in an unsurprising way is not itself evidence about the bug.
    Classification is always based on the *last* command actually reached.
    """

    workspace_path: str
    commands: List[List[str]] = field(default_factory=list)
    working_dir: Optional[str] = None
    timeout_seconds: Optional[int] = None
    image: Optional[str] = None
    expectation: ReproductionExpectation = field(default_factory=ReproductionExpectation)
    task_context: Optional[str] = None


@dataclass
class CommandObservation:
    """Factual, non-interpretive record of one executed command.

    Produced by the executor layer only -- it never decides REPRODUCED vs.
    NOT_REPRODUCED; that is the classifier's job, working from these facts.
    """

    command: str
    exit_code: Optional[int]
    output: str
    toolchain_missing: Optional[str]
    timed_out: bool
    duration: float


@dataclass
class BaselineResult:
    """Normalized outcome of a baseline reproduction attempt.

    Mirrors the spirit of ``VerificationResult`` (see
    ``app.services.verification.base``): a single typed shape every caller
    can consume regardless of what happened internally, with bounded
    evidence rather than raw internal exception tracebacks.
    """

    status: BaselineStatus
    detail: str
    commands: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    ecosystem: Optional[str] = None
    manifests_found: List[str] = field(default_factory=list)
    evidence: Dict[str, str] = field(default_factory=dict)
    observations: List[CommandObservation] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to a plain, JSON-friendly dict."""
        return {
            "status": self.status.value,
            "detail": self.detail,
            "commands": self.commands,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "ecosystem": self.ecosystem,
            "manifests_found": self.manifests_found,
            "evidence": self.evidence,
        }
