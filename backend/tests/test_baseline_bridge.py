"""Unit tests for the plan -> execution-input bridge (Phase 4B-2):
app.services.baseline.bridge.build_reproduction_input.

No Gemini calls, no real Docker/sandbox execution -- these tests exercise
only the bridge's own logic (re-validation, real-filesystem workspace
containment, ecosystem/image reconciliation), using real temporary
directories with real manifest files so ProjectDetector's detection is
genuine, not mocked.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.services.baseline import (
    BridgeOutcome,
    EvidenceReference,
    ExitCodeSemantics,
    KnownCommand,
    ReproductionPlan,
    ReproductionType,
    RepositoryEvidence,
    build_reproduction_input,
)
from app.services.baseline.executor import BaselineExecutor
from app.services.verification.engine import VerificationEngine


def _write(root: Path, rel_path: str, content: str = "") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _node_workspace() -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
    _write(root, "package-lock.json")
    return tmp


def _applicable_plan(**overrides) -> ReproductionPlan:
    defaults = dict(
        applicable=True,
        reason="tests/test_cart.py fails today because of this bug",
        reproduction_type=ReproductionType.TEST_FAILURE,
        commands=[["npm", "test"]],
        expected_observation="The existing cart subtotal test fails.",
        exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
        evidence_refs=["package.json"],
        ecosystem="node",
        timeout_seconds=60,
    )
    defaults.update(overrides)
    return ReproductionPlan(**defaults)


def _evidence() -> RepositoryEvidence:
    return RepositoryEvidence(
        known_commands=[KnownCommand(command=["npm", "test"], description="runs jest", source_file="package.json")],
        evidence_references=[EvidenceReference(file_path="package.json")],
    )


# ===========================================================================
# 1 & 7 & 8. Valid applicable plan -> correct ReproductionInput, with argv,
# working_dir, expected_observation/patterns/timeout all preserved exactly.
# ===========================================================================
def test_valid_plan_produces_correct_reproduction_input():
    with _node_workspace() as tmp:
        root = Path(tmp)
        plan = _applicable_plan(
            reproduced_output_pattern=r"FAIL.*test_subtotal",
            not_reproduced_output_pattern=r"PASS.*test_subtotal",
        )
        result = build_reproduction_input(plan, _evidence(), workspace_path=str(root))

    assert result.outcome == BridgeOutcome.EXECUTABLE
    ri = result.reproduction_input
    assert ri is not None
    assert ri.workspace_path == str(root)
    assert ri.commands == [["npm", "test"]]  # argv preserved exactly, no coercion
    assert ri.working_dir is None  # plan had none, project_root defaults to None too
    assert ri.timeout_seconds == 60
    assert ri.task_context == "The existing cart subtotal test fails."
    assert ri.expectation.exit_code_semantics == ExitCodeSemantics.NONZERO_IS_REPRODUCED
    assert ri.expectation.reproduced_output_pattern == r"FAIL.*test_subtotal"
    assert ri.expectation.not_reproduced_output_pattern == r"PASS.*test_subtotal"


# ===========================================================================
# 2. planning_failed=True -> no executable input, regardless of what else
# the plan contains.
# ===========================================================================
def test_planning_failed_plan_produces_no_executable_input():
    with _node_workspace() as tmp:
        plan = _applicable_plan(planning_failed=True, failure_reason="Gemini timed out")
        result = build_reproduction_input(plan, _evidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert result.reproduction_input is None
    assert "Gemini timed out" in result.detail


# ===========================================================================
# 3. Genuine NOT_APPLICABLE -> no executable input, preserved as-is.
# ===========================================================================
def test_genuine_not_applicable_plan_produces_no_executable_input():
    with _node_workspace() as tmp:
        plan = ReproductionPlan(
            applicable=False,
            reason="No repository evidence supports any reproduction.",
            reproduction_type=ReproductionType.NOT_APPLICABLE,
            planning_failed=False,
        )
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.NOT_APPLICABLE
    assert result.reproduction_input is None


# ===========================================================================
# 4. working_dir remains repository-relative in the produced input.
# ===========================================================================
def test_working_dir_stays_repository_relative():
    with _node_workspace() as tmp:
        root = Path(tmp)
        _write(root, "backend/package.json", json.dumps({"scripts": {"test": "jest"}}))
        plan = _applicable_plan(working_dir="backend", ecosystem="node", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=str(root))

    assert result.outcome == BridgeOutcome.EXECUTABLE
    assert result.reproduction_input.working_dir == "backend"
    assert not Path(result.reproduction_input.working_dir).is_absolute()


# ===========================================================================
# 5. Absolute working_dir rejected (caught by validate_plan's re-check).
# ===========================================================================
def test_absolute_working_dir_is_rejected():
    with _node_workspace() as tmp:
        plan = _applicable_plan(working_dir="/etc", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert result.reproduction_input is None


# ===========================================================================
# 6. Traversal working_dir rejected (caught by validate_plan's re-check).
# ===========================================================================
def test_traversal_working_dir_is_rejected():
    with _node_workspace() as tmp:
        plan = _applicable_plan(working_dir="../../etc", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert result.reproduction_input is None


# ===========================================================================
# 11 (extra, defense-in-depth). A workspace-escape smuggled in via
# project_root -- a field plan_validator does NOT check at all -- must
# still be caught by the bridge's OWN real, filesystem-aware containment
# check (_safe_join_working_dir), proving the two safety layers are
# independent, not just plan_validator's lexical check alone.
# ===========================================================================
def test_project_root_escape_is_caught_by_bridges_own_real_containment_check():
    with tempfile.TemporaryDirectory() as parent_dir:
        parent = Path(parent_dir)
        workspace = parent / "task_123"
        sibling = parent / "task_123_other"
        workspace.mkdir()
        sibling.mkdir()
        (sibling / "secret.txt").write_text("must never be reachable", encoding="utf-8")

        plan = _applicable_plan(working_dir=None, project_root="../task_123_other", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=str(workspace))

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert result.reproduction_input is None


# ===========================================================================
# 9. An arbitrary Docker image cannot bypass the Phase 4A allowlist, even
# if a plan reaches the bridge without ever having gone through the
# planner's own validate_plan call (defense-in-depth re-validation).
# ===========================================================================
def test_disallowed_image_cannot_bypass_the_allowlist():
    with _node_workspace() as tmp:
        plan = _applicable_plan(image="some-attacker-image:latest", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert result.reproduction_input is None


# ===========================================================================
# 10. Ecosystem/image conflicting with real detection is handled
# deterministically -- rejected, never silently executed against the wrong
# ecosystem/image.
# ===========================================================================
def test_ecosystem_conflicting_with_real_detection_is_rejected():
    with _node_workspace() as tmp:  # real detection: node
        plan = _applicable_plan(ecosystem="python", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert "ecosystem" in result.detail.lower()
    assert result.reproduction_input is None


def test_image_conflicting_with_real_detected_ecosystem_image_is_rejected():
    with _node_workspace() as tmp:  # real detection: node -> node:20-slim
        plan = _applicable_plan(image="golang:1.22-alpine", ecosystem=None, evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert "image" in result.detail.lower()
    assert result.reproduction_input is None


def test_ecosystem_claim_with_no_real_detection_at_all_is_rejected_as_unverifiable():
    with tempfile.TemporaryDirectory() as tmp:  # no manifest at all -> nothing detected
        plan = _applicable_plan(ecosystem="node", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert result.reproduction_input is None


def test_matching_ecosystem_and_image_are_accepted():
    with _node_workspace() as tmp:
        plan = _applicable_plan(ecosystem="node", image="node:20-slim", evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.EXECUTABLE
    assert result.reproduction_input.image == "node:20-slim"


def test_no_ecosystem_or_image_claim_is_accepted_and_left_for_executor_to_auto_detect():
    with _node_workspace() as tmp:
        plan = _applicable_plan(ecosystem=None, image=None, evidence_refs=[])
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.EXECUTABLE
    assert result.reproduction_input.image is None  # executor auto-detects at run time


# ===========================================================================
# 11. Isolated workspace path cannot escape the expected workspace: a
# nonexistent workspace_path itself is refused.
# ===========================================================================
def test_nonexistent_workspace_path_is_rejected():
    plan = _applicable_plan(evidence_refs=[])
    result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path="/definitely/does/not/exist/anywhere")

    assert result.outcome == BridgeOutcome.PLANNING_FAILED
    assert result.reproduction_input is None


# ===========================================================================
# 12. Multi-command plans follow Phase 4A's actual supported semantics
# (sequential execution, order preserved) -- NOT rejected, since Phase 4A's
# executor/classifier already safely support running several commands in
# order and classifying based on the last one reached.
# ===========================================================================
def test_multi_command_plan_preserves_full_ordered_sequence():
    with _node_workspace() as tmp:
        plan = _applicable_plan(
            commands=[["npm", "run", "build"], ["npm", "test"]],
            evidence_refs=[],
        )
        result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.EXECUTABLE
    assert result.reproduction_input.commands == [["npm", "run", "build"], ["npm", "test"]]


# ===========================================================================
# 13. The bridge never invokes the sandbox/executor itself.
# ===========================================================================
def test_bridge_never_invokes_the_sandbox():
    with _node_workspace() as tmp:
        plan = _applicable_plan(evidence_refs=[])
        with patch.object(VerificationEngine, "execute_command") as mock_execute, patch.object(
            BaselineExecutor, "run"
        ) as mock_run:
            result = build_reproduction_input(plan, RepositoryEvidence(), workspace_path=tmp)

    assert result.outcome == BridgeOutcome.EXECUTABLE
    mock_execute.assert_not_called()
    mock_run.assert_not_called()
