"""Factual command execution for baseline reproduction -- no classification.

Reuses ``VerificationEngine.execute_command`` (the one small, additive
method added to ``app.services.verification.engine`` for this package)
rather than a second Docker/sandbox runner, so every existing safety
property -- sandboxed workspace, resource limits, network policy,
toolchain-specific images, timeout handling, cleanup -- is inherited
unchanged. This module only sequences commands and bounds their output; it
never decides whether the reported bug was reproduced.

Every early-exit case below (workspace escape, malformed command, a
disallowed image) is represented the same way as an ordinary environment
failure: a single ``CommandObservation`` whose ``toolchain_missing`` field
holds one of the sentinels below, with a clear, ready-to-display message
already written into ``output``. ``classifier.classify`` recognizes these
sentinels and always maps them to ``UNABLE_TO_REPRODUCE`` -- never
NOT_REPRODUCED -- without needing to duplicate their explanations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from app.core.config import settings
from app.services.verification.detector import ADAPTER_PRECEDENCE, DetectionResult, ProjectDetector
from app.services.verification.engine import VerificationEngine

from .models import CommandObservation, ReproductionInput

# Keeps a single reproduction's evidence bounded regardless of how chatty the
# underlying command is, so a huge command output can never blow up memory
# or a future API payload built from BaselineResult. Classification itself
# must NOT use this bounded copy -- see classifier.py, which receives the
# full, untruncated CommandObservation.output and only bounds what it copies
# into the returned BaselineResult.
MAX_OUTPUT_CHARS = 8000

# Sentinels distinguishing *why* a reproduction could not run at all from a
# real (possibly missing) toolchain name -- never surfaced as a real tool.
WORKSPACE_MISSING = "__workspace_missing__"
WORKSPACE_ESCAPED = "__workspace_escaped__"
MALFORMED_COMMAND = "__malformed_command__"
IMAGE_NOT_ALLOWED = "__image_not_allowed__"

ENVIRONMENT_FAILURE_SENTINELS = frozenset(
    {WORKSPACE_MISSING, WORKSPACE_ESCAPED, MALFORMED_COMMAND, IMAGE_NOT_ALLOWED}
)


def bound_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate ``text`` to at most ``limit`` characters, noting how much
    was cut so evidence is never silently incomplete."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... [truncated, {omitted} more characters]"


def _allowed_images() -> frozenset:
    """Docker images this package is allowed to run a reproduction in.

    Deliberately not open to arbitrary caller-supplied strings: the set is
    exactly the images already declared by the codebase's own verification
    adapters (``ADAPTER_PRECEDENCE``) plus the default sandbox image, the
    same trust boundary every other ecosystem's verification already
    operates under. An explicit ``ReproductionInput.image`` must be a member
    of this set or it is rejected outright, never silently substituted or
    executed anyway.
    """
    images = {adapter_cls.docker_image for adapter_cls in ADAPTER_PRECEDENCE}
    images.add(settings.docker_sandbox_image)
    return frozenset(images)


class WorkspaceEscapeError(ValueError):
    """Raised when a reproduction's working_dir would resolve outside its
    workspace_path -- an expected, specifically-handled input error, not a
    programming bug."""


def _safe_join_working_dir(base: Path, working_dir: str) -> Path:
    """Resolve ``working_dir`` as a subdirectory of ``base``, raising
    ``WorkspaceEscapeError`` if it would resolve outside ``base``.

    Containment is checked path-component-wise via ``Path.parents``, never
    a raw string-prefix comparison -- mirrors the existing safe-path pattern
    in ``app.services.agent.tools._resolve_safe_path`` (and
    ``WorkspaceManager``'s own cleanup containment check), which exists
    precisely because a naive ``str(resolved).startswith(str(base))`` check
    would wrongly admit a sibling directory that merely shares ``base``'s
    name as a text prefix (e.g. base ``.../task_123`` would wrongly accept
    ``.../task_123_other`` even though that is not a child of ``base`` at
    all -- ``Path.parents`` never makes that mistake since it compares real
    path components).

    An absolute ``working_dir`` is rejected outright before any resolution
    is attempted (pathlib's ``/`` operator otherwise silently discards
    ``base`` entirely when the right-hand side is itself absolute, which
    would let an absolute ``working_dir`` redirect execution to any
    directory the process can read/write -- confirmed empirically during
    the Phase 4A review).
    """
    candidate = Path(working_dir)
    if candidate.is_absolute():
        raise WorkspaceEscapeError(
            f"working_dir must be a relative path inside the workspace; "
            f"got an absolute path: '{working_dir}'"
        )

    resolved = (base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise WorkspaceEscapeError(
            f"working_dir '{working_dir}' resolves outside the workspace and was rejected."
        )
    return resolved


def _is_well_formed_command(command: List[str]) -> bool:
    return bool(command) and all(isinstance(part, str) and part != "" for part in command)


class BaselineExecutor:
    """Resolves a reproduction's workspace and runs its commands in order."""

    def __init__(self, engine: Optional[VerificationEngine] = None):
        self._engine = engine or VerificationEngine()

    def resolve_workspace(self, repro: ReproductionInput) -> Path:
        """Resolve ``repro``'s effective workspace root.

        Raises ``WorkspaceEscapeError`` if ``working_dir`` would resolve
        outside ``workspace_path`` -- callers that can act on that (i.e.
        ``run()``) must catch it explicitly; this method never silently
        substitutes a different, "safe" directory instead.
        """
        base = Path(repro.workspace_path).resolve()
        if not repro.working_dir:
            return base
        return _safe_join_working_dir(base, repro.working_dir)

    def detect_ecosystem(self, workspace: Path) -> Optional[DetectionResult]:
        """Best-effort ecosystem detection for evidence only -- never gates
        whether a reproduction can run, and a detection failure is silently
        treated as "no ecosystem detected" rather than surfaced as an error."""
        try:
            if not workspace.is_dir():
                return None
            return ProjectDetector.detect(workspace)
        except Exception:
            return None

    def run(self, repro: ReproductionInput) -> List[CommandObservation]:
        """Execute every command in ``repro.commands`` in order.

        Stops early only on an environment/execution failure (workspace
        escape, malformed command, disallowed image, missing toolchain, or
        timeout) -- a command that simply exits nonzero for an ordinary
        reason does not abort the sequence, since an intermediate setup step
        failing unsurprisingly is not evidence about the bug itself. Always
        returns at least one observation when ``repro.commands`` is
        non-empty.
        """
        try:
            workspace = self.resolve_workspace(repro)
        except WorkspaceEscapeError as e:
            return [
                CommandObservation(
                    command="",
                    exit_code=None,
                    output=str(e),
                    toolchain_missing=WORKSPACE_ESCAPED,
                    timed_out=False,
                    duration=0.0,
                )
            ]

        if not workspace.is_dir():
            return [
                CommandObservation(
                    command="",
                    exit_code=None,
                    output=f"Workspace directory does not exist: {workspace}",
                    toolchain_missing=WORKSPACE_MISSING,
                    timed_out=False,
                    duration=0.0,
                )
            ]

        image = repro.image
        if image is not None:
            if image not in _allowed_images():
                return [
                    CommandObservation(
                        command="",
                        exit_code=None,
                        output=(
                            f"Requested image '{image}' is not in the set of "
                            "allowed sandbox images and was rejected before "
                            "execution. Only the codebase's own known "
                            "ecosystem/toolchain images may be used."
                        ),
                        toolchain_missing=IMAGE_NOT_ALLOWED,
                        timed_out=False,
                        duration=0.0,
                    )
                ]
        else:
            detection = self.detect_ecosystem(workspace)
            if detection is not None and detection.adapter is not None:
                image = detection.adapter.docker_image

        observations: List[CommandObservation] = []
        for command in repro.commands:
            if not _is_well_formed_command(command):
                observations.append(
                    CommandObservation(
                        command=" ".join(str(part) for part in command) if command else "",
                        exit_code=None,
                        output=(
                            "Reproduction command is empty or malformed -- each "
                            "command must be a non-empty list of non-empty strings."
                        ),
                        toolchain_missing=MALFORMED_COMMAND,
                        timed_out=False,
                        duration=0.0,
                    )
                )
                break

            start = time.time()
            result = self._engine.execute_command(
                workspace,
                command,
                image=image,
                timeout=repro.timeout_seconds,
            )
            duration = round(time.time() - start, 2)
            exit_code = result["exit_code"]
            observations.append(
                CommandObservation(
                    command=" ".join(command),
                    exit_code=exit_code,
                    # Deliberately NOT bounded here -- classify() needs the
                    # complete output to find evidence that may occur past
                    # MAX_OUTPUT_CHARS; only the returned BaselineResult is
                    # bounded, by classifier.py, after classification.
                    output=result["output"],
                    toolchain_missing=result["toolchain_missing"],
                    timed_out=exit_code == 124,
                    duration=duration,
                )
            )
            if result["toolchain_missing"] or exit_code == 124:
                break

        return observations
