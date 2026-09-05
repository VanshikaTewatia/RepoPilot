"""Standalone baseline-reproduction service entry point (Phase 4A).

``reproduce()`` is the one function a future integration point (Phase 4B)
would call. It is deliberately not called from anywhere else yet -- see the
package docstring in ``app.services.baseline.__init__``.
"""

from __future__ import annotations

from typing import Optional

from .classifier import classify
from .executor import BaselineExecutor, WorkspaceEscapeError
from .models import BaselineResult, ReproductionInput


def reproduce(
    repro: ReproductionInput,
    executor: Optional[BaselineExecutor] = None,
) -> BaselineResult:
    """Attempt to establish whether ``repro`` reproduces its reported bug.

    No commands supplied -> NOT_APPLICABLE, without touching the sandbox at
    all. Otherwise, commands are executed in order (see
    ``BaselineExecutor.run``) and the result is classified deterministically
    (see ``classify``) from the factual observations, never from natural
    -language inference.
    """
    if not repro.commands:
        return classify(repro, [])

    exec_ = executor or BaselineExecutor()
    observations = exec_.run(repro)
    result = classify(repro, observations)

    # Best-effort ecosystem evidence only -- resolve_workspace() raises
    # WorkspaceEscapeError for exactly the same working_dir that run() above
    # already turned into a clean UNABLE_TO_REPRODUCE result, so this must
    # never let that (expected, already-handled) error escape a second time.
    try:
        workspace = exec_.resolve_workspace(repro)
    except WorkspaceEscapeError:
        return result

    detection = exec_.detect_ecosystem(workspace)
    if detection is not None and detection.ecosystem:
        result.ecosystem = detection.ecosystem
        result.manifests_found = detection.manifests_found

    return result
