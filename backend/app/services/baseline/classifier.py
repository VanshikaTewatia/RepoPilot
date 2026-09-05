"""Deterministic classification of reproduction observations into a
``BaselineStatus``.

No Gemini/LLM calls, no natural-language inference. Every decision is a
plain function of: whether commands were supplied at all, whether execution
completed reliably, and the caller-supplied ``ReproductionExpectation``
matched against the *last* command's factual result. Environment/execution
failures are always classified before the expectation is ever consulted, so
UNABLE_TO_REPRODUCE can never be reinterpreted as NOT_REPRODUCED.

Pattern matching is performed against the *complete, untruncated* output of
the last command -- ``CommandObservation.output`` is never bounded before it
reaches this module (see executor.py) precisely so a reproduction marker
occurring after the evidence-size bound can still be found. Bounding is
applied here, once, only to what is copied into the returned
``BaselineResult`` -- so returned evidence stays small while classification
itself never misses anything.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .executor import ENVIRONMENT_FAILURE_SENTINELS, bound_output
from .models import (
    BaselineResult,
    BaselineStatus,
    CommandObservation,
    ExitCodeSemantics,
    ReproductionInput,
)


def _total_duration(observations: List[CommandObservation]) -> float:
    return round(sum(o.duration for o in observations), 2)


def _commands_list(observations: List[CommandObservation]) -> List[str]:
    return [o.command for o in observations if o.command]


def _bounded_observations(observations: List[CommandObservation]) -> List[CommandObservation]:
    """Copy of ``observations`` with each ``.output`` bounded -- never
    attach the full, untruncated output to the returned ``BaselineResult``."""
    return [
        CommandObservation(
            command=o.command,
            exit_code=o.exit_code,
            output=bound_output(o.output),
            toolchain_missing=o.toolchain_missing,
            timed_out=o.timed_out,
            duration=o.duration,
        )
        for o in observations
    ]


def _safe_search(pattern: Optional[str], text: str) -> Optional[bool]:
    """Return whether ``pattern`` matches ``text``, or ``None`` if
    ``pattern`` is not a valid regular expression -- never raises."""
    if not pattern:
        return False
    try:
        return re.search(pattern, text) is not None
    except re.error:
        return None


def classify(repro: ReproductionInput, observations: List[CommandObservation]) -> BaselineResult:
    """Turn factual ``observations`` (from ``BaselineExecutor.run``, carrying
    full, untruncated output) into a single ``BaselineResult`` (with bounded
    evidence)."""

    if not repro.commands:
        return BaselineResult(
            status=BaselineStatus.NOT_APPLICABLE,
            detail="No executable reproduction procedure was supplied for this task.",
        )

    if not observations:
        # Defensive only -- BaselineExecutor.run always returns at least one
        # observation whenever repro.commands is non-empty.
        return BaselineResult(
            status=BaselineStatus.UNABLE_TO_REPRODUCE,
            detail="Reproduction did not execute.",
            commands=[" ".join(c) for c in repro.commands],
        )

    last = observations[-1]
    duration = _total_duration(observations)
    commands = _commands_list(observations)
    bounded_obs = _bounded_observations(observations)

    # --- Environment/input failures: classified before anything else, and
    # never downgraded into NOT_REPRODUCED regardless of what the
    # expectation says. Covers workspace escape, a missing workspace, a
    # malformed command, and a disallowed image (see executor.py), all of
    # which already carry a ready-to-display explanation in `.output`. ---
    if last.toolchain_missing in ENVIRONMENT_FAILURE_SENTINELS:
        return BaselineResult(
            status=BaselineStatus.UNABLE_TO_REPRODUCE,
            detail=last.output,
            commands=commands,
            duration=duration,
            observations=bounded_obs,
        )

    if last.toolchain_missing:
        return BaselineResult(
            status=BaselineStatus.UNABLE_TO_REPRODUCE,
            detail=(
                f"Required tool '{last.toolchain_missing}' is not available in this "
                "environment, so the reported behavior could not be reliably "
                "reproduced. This is not evidence that the reported bug does or "
                "does not exist."
            ),
            commands=commands,
            exit_code=last.exit_code,
            stdout=bound_output(last.output),
            duration=duration,
            evidence={"toolchain_missing": last.toolchain_missing},
            observations=bounded_obs,
        )

    if last.timed_out:
        return BaselineResult(
            status=BaselineStatus.UNABLE_TO_REPRODUCE,
            detail=(
                "Reproduction timed out before a result could be determined. "
                "This is not evidence that the reported bug does or does not exist."
            ),
            commands=commands,
            exit_code=last.exit_code,
            stdout=bound_output(last.output),
            duration=duration,
            observations=bounded_obs,
        )

    # --- Deterministic classification against the explicit expectation,
    # evaluated against the FULL, untruncated output. ---
    expectation = repro.expectation
    output = last.output

    reproduced_match = _safe_search(expectation.reproduced_output_pattern, output)
    not_reproduced_match = _safe_search(expectation.not_reproduced_output_pattern, output)

    if reproduced_match is None or not_reproduced_match is None:
        bad_pattern = (
            expectation.reproduced_output_pattern
            if reproduced_match is None
            else expectation.not_reproduced_output_pattern
        )
        return BaselineResult(
            status=BaselineStatus.UNABLE_TO_REPRODUCE,
            detail=(
                f"The configured reproduction pattern '{bad_pattern}' is not a "
                "valid regular expression, so the result could not be reliably "
                "determined. This is not evidence that the reported bug does or "
                "does not exist -- fix the pattern and retry."
            ),
            commands=commands,
            exit_code=last.exit_code,
            stdout=bound_output(output),
            duration=duration,
            observations=bounded_obs,
        )

    if reproduced_match:
        return BaselineResult(
            status=BaselineStatus.REPRODUCED,
            detail=(
                "The reproduction procedure executed, and the configured "
                "reproduction-evidence pattern was observed in its output."
            ),
            commands=commands,
            exit_code=last.exit_code,
            stdout=bound_output(output),
            duration=duration,
            evidence={"matched_pattern": expectation.reproduced_output_pattern},
            observations=bounded_obs,
        )

    if not_reproduced_match:
        return BaselineResult(
            status=BaselineStatus.NOT_REPRODUCED,
            detail=(
                "The reproduction procedure executed successfully, and the "
                "configured not-reproduced pattern was observed in its output."
            ),
            commands=commands,
            exit_code=last.exit_code,
            stdout=bound_output(output),
            duration=duration,
            evidence={"matched_pattern": expectation.not_reproduced_output_pattern},
            observations=bounded_obs,
        )

    if expectation.exit_code_semantics == ExitCodeSemantics.ZERO_IS_REPRODUCED:
        reproduced = last.exit_code == 0
        return BaselineResult(
            status=BaselineStatus.REPRODUCED if reproduced else BaselineStatus.NOT_REPRODUCED,
            detail=(
                "The reproduction procedure exited 0, which is configured as the "
                "reported bug's expected (reproduced) condition."
                if reproduced
                else "The reproduction procedure executed successfully with a "
                "nonzero exit, which is not the configured reproduced condition."
            ),
            commands=commands,
            exit_code=last.exit_code,
            stdout=bound_output(output),
            duration=duration,
            observations=bounded_obs,
        )

    if expectation.exit_code_semantics == ExitCodeSemantics.NONZERO_IS_REPRODUCED:
        reproduced = last.exit_code != 0
        return BaselineResult(
            status=BaselineStatus.REPRODUCED if reproduced else BaselineStatus.NOT_REPRODUCED,
            detail=(
                "The reproduction procedure exited nonzero, which is configured "
                "as the reported bug's expected (reproduced) condition."
                if reproduced
                else "The reproduction procedure executed successfully and exited "
                "0, which is not the configured reproduced condition."
            ),
            commands=commands,
            exit_code=last.exit_code,
            stdout=bound_output(output),
            duration=duration,
            observations=bounded_obs,
        )

    # exit_code_semantics is IGNORE and no output pattern matched either way:
    # execution completed reliably but nothing configured or observed
    # demonstrates the reported bug.
    return BaselineResult(
        status=BaselineStatus.NOT_REPRODUCED,
        detail=(
            "The reproduction procedure executed successfully, but no expected-bug "
            "condition (output pattern or exit-code rule) was configured or observed."
        ),
        commands=commands,
        exit_code=last.exit_code,
        stdout=bound_output(output),
        duration=duration,
        observations=bounded_obs,
    )
