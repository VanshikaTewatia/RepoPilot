"""Deterministic safety validation for a planner-proposed ``ReproductionPlan``.

The LLM's structured output is never trusted directly -- every field is
re-checked here against explicit, conservative rules before a plan is
considered usable. This is deliberately NOT a general shell-security parser
(the task explicitly asks for "explicit conservative allow/deny rules
appropriate to RepoPilot's existing command representation", not a complete
one) -- it validates the same argv-array shape Phase 4A already executes,
and denies the small set of patterns that would let a plan escape that
shape (an invoked shell interpreter, a workspace-escaping working_dir, a
disallowed Docker image, ...).

When any rule fails, the plan is rejected outright -- see
``planner.plan_reproduction``, which downgrades a rejected plan to a
NOT_APPLICABLE result rather than ever returning something unsafe.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import List

from .executor import _allowed_images
from .models import PlanValidationResult, ReproductionPlan, ReproductionType

# --- Bounds -----------------------------------------------------------------
MAX_PLAN_COMMANDS = 5
MAX_ARGV_LENGTH = 20
MAX_ARGV_TOKEN_CHARS = 300
MAX_TEXT_FIELD_CHARS = 2000
MAX_EVIDENCE_REFS = 10
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 300
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# --- Conservative destructive/escape-hatch denylist --------------------------
# Never a general shell parser: a fixed, explicit set of executables and
# git subcommands a reproduction plan must never use, because none of them
# can ever be "observing" a reported bug -- only mutating the repository,
# fetching/installing arbitrary code, or handing the sandbox a full shell
# (which would defeat the argv-array execution model Phase 4A relies on).
SHELL_INTERPRETERS = {
    "sh", "bash", "zsh", "ksh", "dash",
    "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe",
}

DENIED_STANDALONE_EXECUTABLES = {
    "rm", "del", "rmdir", "rd", "format", "dd", "mkfs",
    "shutdown", "reboot", "poweroff", "halt",
    "curl", "wget",
    "sudo", "su",
} | SHELL_INTERPRETERS

# Executable -> subcommands that mutate history/remotes, install/fetch
# arbitrary code, or otherwise go beyond "observe the current checkout".
DENIED_SUBCOMMANDS_BY_EXECUTABLE = {
    "git": {"push", "commit", "reset", "clean", "branch", "checkout", "rebase", "merge", "tag", "am", "apply"},
    "pip": {"install", "uninstall"},
    "pip3": {"install", "uninstall"},
    "npm": {"install", "i", "ci", "uninstall", "publish"},
    "yarn": {"install", "add", "remove", "publish"},
    "pnpm": {"install", "add", "remove", "publish"},
    "gem": {"install", "uninstall"},
    "cargo": {"install"},
    "go": {"install", "get"},
    "apt": {"install", "remove"},
    "apt-get": {"install", "remove"},
    "yum": {"install", "remove"},
    "choco": {"install"},
    "brew": {"install"},
}

# A stray shell-pipeline/operator character embedded in a single argv token
# is never needed for a real command (Phase 4A always execs argv directly,
# never via a shell) and is a strong signal the plan tried to smuggle a
# pipeline/injection through a single "command" string.
_SUSPICIOUS_ARGV_CHARS = set(";|&`$<>(){}")


def _is_destructive_command(command: List[str]) -> bool:
    if not command:
        return False
    executable = Path(command[0]).name.lower()
    if executable in DENIED_STANDALONE_EXECUTABLES:
        return True
    denied_subcommands = DENIED_SUBCOMMANDS_BY_EXECUTABLE.get(executable)
    if denied_subcommands and any(part.lower() in denied_subcommands for part in command[1:]):
        return True
    for part in command:
        if any(ch in _SUSPICIOUS_ARGV_CHARS for ch in part):
            return True
    return False



# Deliberately NOT `Path(...).is_absolute()`: that's platform-native, and on
# Windows "/etc" is NOT considered absolute (no drive letter) even though it
# unambiguously is one on the POSIX systems these commands actually run on
# (inside a Linux container). A plan's working_dir is always a POSIX-style,
# repository-relative path regardless of which OS happens to run this
# validator, so absoluteness is checked explicitly and OS-independently: a
# leading "/" or "\", or a Windows drive letter followed by ":" and a
# separator.
_ABSOLUTE_PATH_PATTERN = re.compile(r"^(?:[a-zA-Z]:[\\/]|[\\/])")


def _working_dir_escapes(working_dir: str) -> bool:
    """Lexical (no filesystem access) rejection of an absolute or
    traversal-containing working_dir at plan time. This is deliberately
    *in addition to*, not a replacement for, the real, filesystem-aware
    ``Path.parents`` containment check Phase 4A's executor performs at
    execution time (``executor._safe_join_working_dir``) -- a plan has no
    real workspace to resolve against yet."""
    if _ABSOLUTE_PATH_PATTERN.match(working_dir):
        return True
    normalized = working_dir.replace("\\", "/")
    return ".." in PurePosixPath(normalized).parts


def _regex_is_valid(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return True
    except re.error:
        return False


def _citation_file_path(citation: str) -> str:
    """Strip an optional trailing ``:line`` / ``:start-end`` suffix from a
    citation string, matching the ``file:start-end`` convention already used
    elsewhere in this codebase (e.g. app.services.qa.investigator)."""
    if ":" not in citation:
        return citation
    head, _, tail = citation.rpartition(":")
    if tail.replace("-", "").isdigit():
        return head
    return citation


def validate_plan(plan: ReproductionPlan, evidence) -> PlanValidationResult:
    """Deterministically validate ``plan`` against ``evidence``.

    ``evidence`` is a ``RepositoryEvidence`` (or ``None``, in which case
    evidence_refs can never be validated against anything and any non-empty
    evidence_refs is rejected).
    """
    errors: List[str] = []

    # Rule 12/13/14: applicability <-> commands/expected_observation coherence.
    if plan.applicable:
        if not plan.commands:
            errors.append("applicable plan must have at least one command")
        if not plan.expected_observation or not plan.expected_observation.strip():
            errors.append("applicable plan must have a non-empty expected_observation")
    else:
        if plan.commands:
            errors.append("a NOT_APPLICABLE plan must not contain any executable commands")
        if plan.reproduction_type != ReproductionType.NOT_APPLICABLE:
            errors.append("a plan with applicable=False must use reproduction_type=not_applicable")

    # Rule 11: reproduction_type is a supported enum value (structurally
    # guaranteed once constructed as ReproductionType, but re-checked for
    # defense-in-depth against a caller constructing the dataclass directly).
    if not isinstance(plan.reproduction_type, ReproductionType):
        errors.append(f"unsupported reproduction_type: {plan.reproduction_type!r}")

    # Rule 10: exit_code_semantics is a supported enum value (same rationale).
    from .models import ExitCodeSemantics  # local import: avoid a module-level cycle risk

    if not isinstance(plan.exit_code_semantics, ExitCodeSemantics):
        errors.append(f"unsupported exit_code_semantics: {plan.exit_code_semantics!r}")

    # Rule 3: command count is bounded.
    if len(plan.commands) > MAX_PLAN_COMMANDS:
        errors.append(f"too many commands ({len(plan.commands)} > {MAX_PLAN_COMMANDS})")

    # Rules 1/2/4/15: each command is well-formed, bounded, and non-destructive.
    for command in plan.commands:
        if not command or not all(isinstance(part, str) and part != "" for part in command):
            errors.append(f"command is empty or contains a non-empty-string violation: {command!r}")
            continue
        if len(command) > MAX_ARGV_LENGTH:
            errors.append(f"command has too many arguments ({len(command)} > {MAX_ARGV_LENGTH}): {command!r}")
        if any(len(part) > MAX_ARGV_TOKEN_CHARS for part in command):
            errors.append(f"command argument exceeds {MAX_ARGV_TOKEN_CHARS} characters: {command!r}")
        if _is_destructive_command(command):
            errors.append(f"command is denied as potentially destructive/unsafe: {command!r}")

    # Rule 5/6: working_dir is relative and does not traverse outside the
    # repository (lexically -- see _working_dir_escapes's docstring).
    if plan.working_dir:
        if _working_dir_escapes(plan.working_dir):
            errors.append(f"working_dir must be a relative, non-traversing path: {plan.working_dir!r}")

    # Rule 7: timeout is within safe bounds, when given at all.
    if plan.timeout_seconds is not None:
        if not (MIN_TIMEOUT_SECONDS <= plan.timeout_seconds <= MAX_TIMEOUT_SECONDS):
            errors.append(
                f"timeout_seconds {plan.timeout_seconds} is outside the safe range "
                f"[{MIN_TIMEOUT_SECONDS}, {MAX_TIMEOUT_SECONDS}]"
            )

    # Rule 8: an explicit image override must be one Phase 4A already knows.
    if plan.image is not None and plan.image not in _allowed_images():
        errors.append(f"image {plan.image!r} is not in the Phase 4A allowed image set")

    # Rule 9: regex patterns compile.
    if plan.reproduced_output_pattern and not _regex_is_valid(plan.reproduced_output_pattern):
        errors.append(f"reproduced_output_pattern is not a valid regular expression: {plan.reproduced_output_pattern!r}")
    if plan.not_reproduced_output_pattern and not _regex_is_valid(plan.not_reproduced_output_pattern):
        errors.append(
            f"not_reproduced_output_pattern is not a valid regular expression: {plan.not_reproduced_output_pattern!r}"
        )

    # Bounded text fields.
    for field_name, value in (
        ("reason", plan.reason),
        ("expected_observation", plan.expected_observation),
    ):
        if value and len(value) > MAX_TEXT_FIELD_CHARS:
            errors.append(f"{field_name} exceeds {MAX_TEXT_FIELD_CHARS} characters")

    # Confidence is bounded.
    if not (MIN_CONFIDENCE <= plan.confidence <= MAX_CONFIDENCE):
        errors.append(f"confidence {plan.confidence} is outside [{MIN_CONFIDENCE}, {MAX_CONFIDENCE}]")

    # Rule: evidence_refs bounded and each must point to known evidence.
    if len(plan.evidence_refs) > MAX_EVIDENCE_REFS:
        errors.append(f"too many evidence_refs ({len(plan.evidence_refs)} > {MAX_EVIDENCE_REFS})")

    if plan.evidence_refs:
        known_paths = evidence.known_evidence_paths() if evidence is not None else set()
        for ref in plan.evidence_refs:
            if _citation_file_path(ref) not in known_paths:
                errors.append(f"evidence_ref does not correspond to known investigation evidence: {ref!r}")

    return PlanValidationResult(valid=not errors, errors=errors)
