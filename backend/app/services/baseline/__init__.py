"""Standalone baseline bug-reproduction infrastructure (Phase 4A).

Before RepoPilot edits a repository for a reported bug, it should eventually
be able to independently establish whether the reported behavior can be
reproduced against the pre-fix checkout. This package is that standalone
capability -- executor, deterministic classifier, and typed result/input
models -- reusing the existing verification sandbox
(``app.services.verification.engine.VerificationEngine.execute_command``)
rather than a second Docker runner.

PHASE 4A SCOPE: this package is intentionally NOT wired into the LangGraph
agent graph (``app.services.agent.graph``), ``AgentState``, task outcomes,
the API layer, or any frontend behavior. Nothing here changes any existing
task's result. That integration is Phase 4B, once this is validated
independently.

Call ``reproduce()`` directly (e.g. from a test, or a future Phase 4B call
site) with a ``ReproductionInput``:

    from app.services.baseline import (
        ReproductionInput, ReproductionExpectation, ExitCodeSemantics, reproduce,
    )

    result = reproduce(ReproductionInput(
        workspace_path="/path/to/checkout",
        commands=[["python", "repro.py"]],
        expectation=ReproductionExpectation(
            exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
        ),
    ))
    result.status  # BaselineStatus.REPRODUCED / NOT_REPRODUCED / ...
"""

from .classifier import classify
from .executor import BaselineExecutor, WorkspaceEscapeError, bound_output
from .models import (
    BaselineResult,
    BaselineStatus,
    CommandObservation,
    ExitCodeSemantics,
    ReproductionExpectation,
    ReproductionInput,
)
from .service import reproduce

__all__ = [
    "BaselineExecutor",
    "BaselineResult",
    "BaselineStatus",
    "CommandObservation",
    "ExitCodeSemantics",
    "ReproductionExpectation",
    "ReproductionInput",
    "WorkspaceEscapeError",
    "bound_output",
    "classify",
    "reproduce",
]
