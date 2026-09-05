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


# ===========================================================================
# Phase 4B-1: reproduction planning (models only -- see planner.py /
# plan_validator.py). Nothing below this line is executed by Phase 4A; a
# ReproductionPlan is a *proposal*, never itself run, classified, or treated
# as evidence that a bug exists. Phase 4B-2 is responsible for turning an
# already-validated ReproductionPlan into a Phase 4A ReproductionInput.
# ===========================================================================


class ReproductionType(str, Enum):
    """What *kind* of evidence would demonstrate the reported bug -- chosen
    by the planner only when repository evidence actually supports it, never
    guessed to avoid returning NOT_APPLICABLE."""

    COMMAND_FAILURE = "command_failure"
    TEST_FAILURE = "test_failure"
    RUNTIME_BEHAVIOR = "runtime_behavior"
    STATIC_OBSERVATION = "static_observation"
    BUILD_FAILURE = "build_failure"
    API_BEHAVIOR = "api_behavior"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class KnownCommand:
    """A command already known to exist in the repository (a package.json
    script, a Makefile target, an existing test invocation, ...) -- never
    invented. ``source_file`` is the repo-relative manifest/file that proves
    this command is real, and is what plan_validator checks evidence_refs
    against."""

    command: List[str]
    description: str
    source_file: str


@dataclass
class EvidenceReference:
    """One piece of repository evidence (a file, optionally a line range)
    that a plan may cite in ``evidence_refs``."""

    file_path: str
    description: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    @property
    def citation(self) -> str:
        if self.line_start is not None:
            end = self.line_end if self.line_end is not None else self.line_start
            return f"{self.file_path}:{self.line_start}-{end}"
        return self.file_path


@dataclass
class RepositoryEvidence:
    """Repository investigation evidence available to the planner.

    Treated as authoritative over the user's own report -- see
    planner.py's system instruction. Assembled by whatever calls the
    planner (a future Phase 4B-2 integration point, or a test); this
    package makes no assumption about where it came from and never queries
    a live repository or database itself.
    """

    detected_projects: List = field(default_factory=list)  # List[ProjectInfo]
    known_commands: List[KnownCommand] = field(default_factory=list)
    investigation_findings: str = ""
    evidence_references: List[EvidenceReference] = field(default_factory=list)

    def known_evidence_paths(self) -> set:
        """Every repo-relative file path this evidence actually vouches
        for -- used by plan_validator to reject a citation to a path that
        was never part of the investigation."""
        paths = {ref.file_path for ref in self.evidence_references}
        paths |= {kc.source_file for kc in self.known_commands}
        return paths


@dataclass
class ReproductionPlan:
    """A proposed, not-yet-validated-for-safety procedure for determining
    whether a reported bug can be reproduced.

    This is a PROPOSAL only: it never claims the bug exists, is never
    executed by this package, and must pass ``plan_validator.validate_plan``
    before any future caller may act on it. ``exit_code_semantics`` /
    ``reproduced_output_pattern`` / ``not_reproduced_output_pattern`` reuse
    the exact Phase 4A semantics (see ``ExitCodeSemantics`` /
    ``ReproductionExpectation``) so a validated plan maps onto a Phase 4A
    ``ReproductionInput`` without any semantic translation.

    ``planning_failed`` / ``failure_reason`` mirror the exact pattern already
    used by ``app.services.qa.classifier.QuestionClass.classification_failed``
    / ``failure_reason`` for the identical problem: distinguishing a genuine
    result from a safe placeholder returned only because something upstream
    (Gemini, JSON parsing, enum construction, safety validation) failed.
    ``applicable=False`` / ``reproduction_type=NOT_APPLICABLE`` alone are
    NEVER sufficient to conclude "no reproduction exists" -- a caller MUST
    check ``planning_failed`` first:

    - ``planning_failed=False`` and not applicable: planning completed and
      genuinely found no evidence-backed executable procedure. This is a
      real NOT_APPLICABLE verdict.
    - ``planning_failed=True``: planning itself did not complete (a
      transient Gemini/network/API failure, malformed JSON, an unsupported
      enum value, or a proposed plan that failed safety validation).
      ``applicable``/``reproduction_type`` are structural placeholders only
      -- this is NOT evidence that reproduction is inapplicable, and a
      future caller (Phase 4B-2) should map it to something like
      ``BaselineStatus.UNABLE_TO_REPRODUCE``, never silently to
      NOT_APPLICABLE/NOT_REPRODUCED.
    """

    applicable: bool
    reason: str
    reproduction_type: ReproductionType
    commands: List[List[str]] = field(default_factory=list)
    working_dir: Optional[str] = None
    expected_observation: Optional[str] = None
    exit_code_semantics: ExitCodeSemantics = ExitCodeSemantics.IGNORE
    reproduced_output_pattern: Optional[str] = None
    not_reproduced_output_pattern: Optional[str] = None
    confidence: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)
    project_root: Optional[str] = None
    ecosystem: Optional[str] = None
    image: Optional[str] = None
    timeout_seconds: Optional[int] = None
    planning_failed: bool = False
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "applicable": self.applicable,
            "reason": self.reason,
            "reproduction_type": self.reproduction_type.value,
            "commands": self.commands,
            "working_dir": self.working_dir,
            "expected_observation": self.expected_observation,
            "exit_code_semantics": self.exit_code_semantics.value,
            "reproduced_output_pattern": self.reproduced_output_pattern,
            "not_reproduced_output_pattern": self.not_reproduced_output_pattern,
            "confidence": self.confidence,
            "evidence_refs": self.evidence_refs,
            "project_root": self.project_root,
            "ecosystem": self.ecosystem,
            "image": self.image,
            "timeout_seconds": self.timeout_seconds,
            "planning_failed": self.planning_failed,
            "failure_reason": self.failure_reason,
        }


@dataclass
class PlanValidationResult:
    """Outcome of running a ``ReproductionPlan`` through the deterministic
    validator. ``valid=False`` means the plan must never be executed as-is
    -- the planner downgrades it to a NOT_APPLICABLE plan instead of
    returning it."""

    valid: bool
    errors: List[str] = field(default_factory=list)
