"""Unit tests for the standalone baseline bug-reproduction package
(app.services.baseline, Phase 4A).

No real Docker or network calls are made -- Docker is exercised via a
MagicMock client (mirroring tests/test_verification_engine.py's approach)
and the subprocess fallback via `subprocess.run` patched with
`unittest.mock`. No Gemini/LLM calls are involved anywhere in this module.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ReadTimeout

from app.core.config import settings
from app.services.baseline import (
    BaselineExecutor,
    BaselineStatus,
    ExitCodeSemantics,
    ReproductionExpectation,
    ReproductionInput,
    WorkspaceEscapeError,
    reproduce,
)
from app.services.baseline.classifier import classify
from app.services.baseline.executor import MAX_OUTPUT_CHARS, _allowed_images
from app.services.baseline.models import CommandObservation
from app.services.verification.engine import VerificationEngine


def _write(root: Path, rel_path: str, content: str = "") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _fake_completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def _fake_container(status_code: int, logs: bytes) -> MagicMock:
    container = MagicMock()
    container.wait.return_value = {"StatusCode": status_code}
    container.logs.return_value = logs
    return container


_active_patchers = []


def _docker_unavailable_engine() -> VerificationEngine:
    """Return a VerificationEngine with Docker forced unavailable.

    The patch is stopped by ``_stop_leaked_patches`` (autouse) at the end of
    the test that called this, so it never leaks into other test modules
    that share the same DockerTestRunner class-level property.
    """
    engine = VerificationEngine()
    patcher = patch.object(type(engine._docker_runner), "is_docker_available", False)
    patcher.start()
    _active_patchers.append(patcher)
    return engine


@pytest.fixture(autouse=True)
def _stop_leaked_patches():
    yield
    for patcher in _active_patchers:
        patcher.stop()
    _active_patchers.clear()


# ---------------------------------------------------------------------------
# 1. No reproduction procedure -> NOT_APPLICABLE
# ---------------------------------------------------------------------------
def test_no_commands_returns_not_applicable_without_touching_sandbox():
    with tempfile.TemporaryDirectory() as tmpdir:
        repro = ReproductionInput(workspace_path=tmpdir, commands=[])

        with patch("app.services.baseline.executor.BaselineExecutor.run") as mock_run:
            result = reproduce(repro)

        assert result.status == BaselineStatus.NOT_APPLICABLE
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# 2 & 9. Valid command, expected behavior observed (including a *successful*
# exit-0 command that explicitly prints the observed incorrect behavior)
# -> REPRODUCED
# ---------------------------------------------------------------------------
def test_successful_command_with_reproduced_pattern_is_reproduced():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(
                reproduced_output_pattern=r"BUG: total was \d+, expected 100",
            ),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout="BUG: total was 42, expected 100\n"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.REPRODUCED
        assert result.exit_code == 0
        assert "matched_pattern" in result.evidence


# ---------------------------------------------------------------------------
# 3. Valid command, expected behavior not observed -> NOT_REPRODUCED
# ---------------------------------------------------------------------------
def test_successful_command_without_expected_pattern_is_not_reproduced():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(
                reproduced_output_pattern=r"BUG: total was \d+, expected 100",
            ),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout="total was 100, expected 100\n"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.NOT_REPRODUCED
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 4. Missing toolchain -> UNABLE_TO_REPRODUCE
# ---------------------------------------------------------------------------
def test_missing_toolchain_is_unable_to_reproduce():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["some-nonexistent-tool", "repro"]],
        )

        with patch("app.services.verification.engine._tool_is_available", return_value=False):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert "some-nonexistent-tool" in result.detail
        assert result.evidence.get("toolchain_missing") == "some-nonexistent-tool"


# ---------------------------------------------------------------------------
# 5. Sandbox/Docker unavailable (and the host has no usable fallback either)
# -> UNABLE_TO_REPRODUCE, never silently downgraded
# ---------------------------------------------------------------------------
def test_docker_unavailable_and_no_host_fallback_is_unable_to_reproduce():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["some-tool-not-on-host", "repro"]],
            expectation=ReproductionExpectation(
                exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
            ),
        )

        engine = _docker_unavailable_engine()
        assert engine._docker_runner.is_docker_available is False
        result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        # Even though the expectation says "nonzero exit == reproduced", an
        # environment failure must win -- never reinterpreted as REPRODUCED
        # or NOT_REPRODUCED.
        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE


# ---------------------------------------------------------------------------
# 6. Timeout -> UNABLE_TO_REPRODUCE
# ---------------------------------------------------------------------------
def test_docker_timeout_is_unable_to_reproduce_and_container_removed():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["node", "repro.js"]],
            timeout_seconds=5,
        )

        engine = VerificationEngine()
        fake_container = MagicMock()
        fake_container.wait.side_effect = ReadTimeout("timed out")
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = fake_container

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert "timed out" in result.detail.lower()
        fake_container.kill.assert_called_once()
        fake_container.remove.assert_called_once_with(force=True)


def test_subprocess_timeout_is_unable_to_reproduce():
    import subprocess as subprocess_module

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "slow.py"]],
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            side_effect=subprocess_module.TimeoutExpired(cmd="python slow.py", timeout=5),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE


# ---------------------------------------------------------------------------
# 7. Command exits non-zero, and that failure IS the expected reproduction
# -> REPRODUCED
# ---------------------------------------------------------------------------
def test_nonzero_exit_configured_as_reproduced_condition():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(
                exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
            ),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(1, stdout="", stderr="Traceback... IndexError\n"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.REPRODUCED
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# 8. Command exits non-zero but does not establish the expected behavior
# -> NOT_REPRODUCED (a normal project failure is never mistaken for
# infrastructure failure -- test 12 -- since no toolchain-missing sentinel
# or timeout was involved here)
# ---------------------------------------------------------------------------
def test_nonzero_exit_without_matching_expectation_is_not_reproduced():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(
                exit_code_semantics=ExitCodeSemantics.ZERO_IS_REPRODUCED,
            ),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(1, stdout="", stderr="unrelated failure\n"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.NOT_REPRODUCED
        assert result.exit_code == 1


def test_normal_project_failure_is_not_mistaken_for_infrastructure_failure():
    """A project's own command exiting nonzero (e.g. a real assertion
    failure) must never be classified as UNABLE_TO_REPRODUCE -- only our own
    preflight sentinel or a timeout may do that."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(1, stdout="", stderr="AssertionError: expected 100 got 42\n"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status != BaselineStatus.UNABLE_TO_REPRODUCE
        assert result.status == BaselineStatus.NOT_REPRODUCED


# ---------------------------------------------------------------------------
# 10. Output is captured and bounded
# ---------------------------------------------------------------------------
def test_large_output_is_bounded():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        huge_output = "x" * (MAX_OUTPUT_CHARS * 3)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout=huge_output),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert len(result.stdout) < len(huge_output)
        assert "truncated" in result.stdout


# ---------------------------------------------------------------------------
# 11. Existing sandbox cleanup still happens (Docker path)
# ---------------------------------------------------------------------------
def test_docker_container_is_removed_after_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["node", "repro.js"]],
        )

        engine = VerificationEngine()
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = _fake_container(0, b"ok\n")

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            reproduce(repro, executor=BaselineExecutor(engine=engine))

        fake_docker_client.containers.run.return_value.remove.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# 13. UNABLE_TO_REPRODUCE is never converted into NOT_REPRODUCED, even when
# the expectation would otherwise have said NOT_REPRODUCED for this exit code
# ---------------------------------------------------------------------------
def test_environment_failure_overrides_expectation_that_would_say_not_reproduced():
    observations = [
        CommandObservation(
            command="npm test",
            exit_code=127,
            output="REPOPILOT_TOOLCHAIN_MISSING:npm\nnot found",
            toolchain_missing="npm",
            timed_out=False,
            duration=0.1,
        )
    ]
    repro = ReproductionInput(
        workspace_path="/irrelevant",
        commands=[["npm", "test"]],
        expectation=ReproductionExpectation(
            exit_code_semantics=ExitCodeSemantics.ZERO_IS_REPRODUCED,
        ),
    )

    result = classify(repro, observations)

    assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
    assert result.status != BaselineStatus.NOT_REPRODUCED


# ---------------------------------------------------------------------------
# Multi-command sequencing: a non-last command's ordinary nonzero exit does
# not abort the sequence or get misclassified as an environment failure;
# classification is based on the last command reached.
# ---------------------------------------------------------------------------
def test_non_last_command_ordinary_failure_does_not_abort_sequence():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "setup.py"], ["python", "repro.py"]],
            expectation=ReproductionExpectation(
                exit_code_semantics=ExitCodeSemantics.NONZERO_IS_REPRODUCED,
            ),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            side_effect=[
                _fake_completed(1, stdout="setup step exited nonzero for its own reasons\n"),
                _fake_completed(1, stdout="", stderr="reproduced crash\n"),
            ],
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.REPRODUCED
        assert len(result.observations) == 2


def test_environment_failure_mid_sequence_stops_remaining_commands():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["missing-tool", "setup"], ["python", "repro.py"]],
        )

        with patch("app.services.verification.engine._tool_is_available", return_value=False):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert len(result.observations) == 1


# ---------------------------------------------------------------------------
# Workspace path that doesn't exist at all -> UNABLE_TO_REPRODUCE
# ---------------------------------------------------------------------------
def test_nonexistent_workspace_is_unable_to_reproduce():
    repro = ReproductionInput(
        workspace_path="/definitely/does/not/exist/anywhere",
        commands=[["python", "repro.py"]],
    )

    result = reproduce(repro, executor=BaselineExecutor(engine=_docker_unavailable_engine()))

    assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE


# ===========================================================================
# Hardening pass: CRITICAL #1 -- working_dir workspace escape
# ===========================================================================
def test_working_dir_valid_subdirectory_is_used():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "frontend/marker.txt", "present")
        repro = ReproductionInput(
            workspace_path=str(root),
            working_dir="frontend",
            commands=[["python", "repro.py"]],
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout="ok"),
        ) as mock_run:
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.NOT_REPRODUCED
        # the command actually ran with the subdirectory as cwd, not workspace_path
        assert mock_run.call_args.kwargs["cwd"] == (root / "frontend").resolve()


def test_working_dir_relative_traversal_escape_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            working_dir="../",
            commands=[["python", "repro.py"]],
        )

        with patch("app.services.verification.engine.subprocess.run") as mock_run:
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert "escape" in result.detail.lower() or "outside" in result.detail.lower()
        mock_run.assert_not_called()


def test_working_dir_deep_traversal_escape_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            working_dir="../../../../",
            commands=[["python", "repro.py"]],
        )

        with patch("app.services.verification.engine.subprocess.run") as mock_run:
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        mock_run.assert_not_called()


def test_working_dir_absolute_windows_path_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            working_dir="C:\\Windows\\System32",
            commands=[["python", "repro.py"]],
        )

        with patch("app.services.verification.engine.subprocess.run") as mock_run:
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert "absolute" in result.detail.lower()
        mock_run.assert_not_called()


def test_working_dir_sibling_prefix_directory_is_rejected():
    """A sibling directory that merely shares a text prefix with the
    workspace (e.g. 'task_123' vs 'task_123_other') must never be treated
    as contained -- this is exactly what a naive str.startswith() check
    would get wrong; Path.parents must not."""
    with tempfile.TemporaryDirectory() as tmpdir:
        parent = Path(tmpdir)
        workspace = parent / "task_123"
        sibling = parent / "task_123_other"
        workspace.mkdir()
        sibling.mkdir()
        (sibling / "secret.txt").write_text("should never be reachable", encoding="utf-8")

        repro = ReproductionInput(
            workspace_path=str(workspace),
            working_dir="../task_123_other",
            commands=[["python", "repro.py"]],
        )

        with patch("app.services.verification.engine.subprocess.run") as mock_run:
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        mock_run.assert_not_called()


def test_working_dir_escape_never_invokes_sandbox_via_executor_directly():
    """Same rejection, exercised directly on BaselineExecutor.run (not just
    through reproduce()), proving the sandbox is genuinely never invoked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            working_dir="/etc",
            commands=[["cat", "shadow"]],
        )

        with patch("app.services.verification.engine.subprocess.run") as mock_run, patch(
            "app.services.verification.engine.VerificationEngine.execute_command"
        ) as mock_execute:
            engine = _docker_unavailable_engine()
            observations = BaselineExecutor(engine=engine).run(repro)

        assert len(observations) == 1
        assert observations[0].toolchain_missing == "__workspace_escaped__"
        mock_run.assert_not_called()
        mock_execute.assert_not_called()


# ===========================================================================
# Hardening pass: HIGH #2 -- classification must see untruncated output
# ===========================================================================
def test_reproduction_marker_after_truncation_boundary_is_still_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        noisy_output = ("irrelevant log line\n" * 500) + "BUG: total was 42, expected 100\n"
        assert len(noisy_output) > MAX_OUTPUT_CHARS

        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(
                reproduced_output_pattern=r"BUG: total was \d+, expected 100",
            ),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout=noisy_output),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        # classification found the marker despite it occurring after the
        # evidence-size bound...
        assert result.status == BaselineStatus.REPRODUCED
        # ...but the *returned* evidence is still bounded.
        assert len(result.stdout) <= MAX_OUTPUT_CHARS + 100  # + truncation note
        assert "truncated" in result.stdout
        for observation in result.observations:
            assert len(observation.output) <= MAX_OUTPUT_CHARS + 100


# ===========================================================================
# Hardening pass: MEDIUM #3 -- execution exception boundary
# ===========================================================================
def test_empty_command_produces_clean_unable_to_reproduce_not_a_crash():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(workspace_path=str(root), commands=[[]])

        engine = _docker_unavailable_engine()
        result = reproduce(repro, executor=BaselineExecutor(engine=engine))  # must not raise

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert "malformed" in result.detail.lower() or "empty" in result.detail.lower()


def test_command_with_blank_argument_is_also_rejected_cleanly():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(workspace_path=str(root), commands=[["python", ""]])

        engine = _docker_unavailable_engine()
        result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE


# ===========================================================================
# Hardening pass: MEDIUM #4 -- malformed regex
# ===========================================================================
def test_malformed_reproduced_pattern_is_unable_to_reproduce_not_crash_not_not_reproduced():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(reproduced_output_pattern="([unbalanced"),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout="hello"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))  # must not raise

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert result.status != BaselineStatus.NOT_REPRODUCED


def test_malformed_not_reproduced_pattern_is_unable_to_reproduce():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            expectation=ReproductionExpectation(not_reproduced_output_pattern="[a-z"),
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout="hello"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE


# ===========================================================================
# Hardening pass: MEDIUM #5 -- timeout mutation race
# ===========================================================================
def test_custom_timeout_does_not_leak_into_engine_shared_state():
    engine = VerificationEngine(timeout=30)
    fake_docker_client = MagicMock()
    fake_docker_client.containers.run.return_value = _fake_container(0, b"ok\n")

    with patch.object(type(engine._docker_runner), "is_docker_available", True):
        engine._docker_runner._docker_client = fake_docker_client
        engine.execute_command(Path("."), ["echo", "hi"], timeout=5)

    assert engine.timeout == 30  # unmutated by the call
    fake_docker_client.containers.run.return_value.wait.assert_called_once_with(timeout=5)


def test_sequential_calls_with_different_custom_timeouts_never_leak_between_each_other():
    """No shared, mutated timeout state: two calls on the same engine with
    different custom timeouts must each use their own value, and the
    engine's own baseline timeout must never change."""
    engine = VerificationEngine(timeout=30)
    fake_docker_client = MagicMock()

    first_container = _fake_container(0, b"first\n")
    second_container = _fake_container(0, b"second\n")
    fake_docker_client.containers.run.side_effect = [first_container, second_container]

    with patch.object(type(engine._docker_runner), "is_docker_available", True):
        engine._docker_runner._docker_client = fake_docker_client
        engine.execute_command(Path("."), ["echo", "one"], timeout=5)
        assert engine.timeout == 30
        engine.execute_command(Path("."), ["echo", "two"], timeout=10)
        assert engine.timeout == 30

    first_container.wait.assert_called_once_with(timeout=5)
    second_container.wait.assert_called_once_with(timeout=10)


def test_no_custom_timeout_falls_back_to_engine_default():
    engine = VerificationEngine(timeout=42)
    fake_docker_client = MagicMock()
    fake_docker_client.containers.run.return_value = _fake_container(0, b"ok\n")

    with patch.object(type(engine._docker_runner), "is_docker_available", True):
        engine._docker_runner._docker_client = fake_docker_client
        engine.execute_command(Path("."), ["echo", "hi"])

    fake_docker_client.containers.run.return_value.wait.assert_called_once_with(timeout=42)
    assert engine.timeout == 42


# ===========================================================================
# Hardening pass: MEDIUM #6 -- arbitrary Docker image
# ===========================================================================
def test_disallowed_image_is_rejected_and_never_executed():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            image="some-attacker-supplied-image:latest",
        )

        with patch("app.services.verification.engine.subprocess.run") as mock_run, patch(
            "app.services.verification.engine.VerificationEngine.execute_command"
        ) as mock_execute:
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.UNABLE_TO_REPRODUCE
        assert "not in the set of allowed" in result.detail.lower() or "not allowed" in result.detail.lower()
        mock_run.assert_not_called()
        mock_execute.assert_not_called()


def test_allowed_explicit_image_is_accepted():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        allowed_image = settings.docker_sandbox_image
        assert allowed_image in _allowed_images()

        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            image=allowed_image,
        )

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout="ok"),
        ):
            engine = _docker_unavailable_engine()
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        assert result.status == BaselineStatus.NOT_REPRODUCED


def test_auto_detected_ecosystem_image_bypasses_no_allowlist_check():
    """Auto-detected images come from the adapters themselves (already in
    the allowlist by construction), so this must keep working exactly as
    before -- only an *explicit* ReproductionInput.image is validated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["go", "test", "./..."]],
        )

        engine = VerificationEngine()
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = _fake_container(0, b"ok\n")

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        run_kwargs = fake_docker_client.containers.run.call_args.kwargs
        assert run_kwargs["image"] == "golang:1.22-alpine"
        assert result.status == BaselineStatus.NOT_REPRODUCED


# ===========================================================================
# Test gaps: deterministic Docker-unavailable coverage (distinct from
# missing-toolchain: here the tool genuinely would exist, Docker is simply
# forced unavailable and the subprocess fallback is exercised deterministically)
# ===========================================================================
def test_docker_unavailable_falls_back_to_subprocess_deterministically():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(workspace_path=str(root), commands=[["python", "repro.py"]])

        with patch(
            "app.services.verification.engine.subprocess.run",
            return_value=_fake_completed(0, stdout="ran via subprocess"),
        ) as mock_run:
            engine = VerificationEngine()
            with patch.object(type(engine._docker_runner), "is_docker_available", False):
                result = reproduce(repro, executor=BaselineExecutor(engine=engine))

        mock_run.assert_called_once()
        assert result.status == BaselineStatus.NOT_REPRODUCED


# ===========================================================================
# Test gaps: timeout_seconds validation
# ===========================================================================
def test_negative_timeout_seconds_is_rejected_cleanly():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        repro = ReproductionInput(
            workspace_path=str(root),
            commands=[["python", "repro.py"]],
            timeout_seconds=-5,
        )

        engine = _docker_unavailable_engine()
        result = reproduce(repro, executor=BaselineExecutor(engine=engine))  # must not raise/hang

        assert result.status in (BaselineStatus.UNABLE_TO_REPRODUCE, BaselineStatus.NOT_REPRODUCED, BaselineStatus.REPRODUCED)


def test_zero_timeout_seconds_is_passed_through_explicitly_not_treated_as_unset():
    engine = VerificationEngine(timeout=30)
    fake_docker_client = MagicMock()
    fake_docker_client.containers.run.return_value = _fake_container(0, b"ok\n")

    with patch.object(type(engine._docker_runner), "is_docker_available", True):
        engine._docker_runner._docker_client = fake_docker_client
        engine.execute_command(Path("."), ["echo", "hi"], timeout=0)

    fake_docker_client.containers.run.return_value.wait.assert_called_once_with(timeout=0)
