"""Unit tests for Docker and subprocess sandbox execution."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, ContainerError
from requests.exceptions import ReadTimeout

import app.services.sandbox.docker_runner as docker_runner_module
from app.services.sandbox.docker_runner import DockerTestRunner, _detect_dependency_install_args


@pytest.fixture(autouse=True)
def _reset_global_docker_disabled():
    """_GLOBAL_DOCKER_DISABLED is a module-level cache shared across every
    DockerTestRunner instance -- never let one test's simulated Docker
    outage leak into another test in this file."""
    docker_runner_module._GLOBAL_DOCKER_DISABLED = False
    yield
    docker_runner_module._GLOBAL_DOCKER_DISABLED = False


def _fake_completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def _fake_container(status_code: int = 0, logs: bytes = b"1 passed in 0.01s") -> MagicMock:
    """Stands in for the docker-py ``Container`` object returned by
    ``containers.run(detach=True, ...)``."""
    container = MagicMock()
    container.wait.return_value = {"StatusCode": status_code}
    container.logs.return_value = logs
    return container


def test_sandbox_pytest_output_parser():
    """Test parsing pytest summary strings."""
    runner = DockerTestRunner()

    sample_output = "======= 5 passed, 2 failed in 0.45s ======="
    parsed = runner._parse_pytest_output(sample_output)
    assert parsed["passed"] == 5
    assert parsed["failed"] == 2

    passing_output = "======= 10 passed in 1.20s ======="
    parsed_pass = runner._parse_pytest_output(passing_output)
    assert parsed_pass["passed"] == 10
    assert parsed_pass["failed"] == 0


def test_sandbox_run_passing_test():
    """Test executing a test file in temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        test_file = workspace / "test_math.py"
        test_file.write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

        runner = DockerTestRunner()
        result = runner.run_tests(workspace)

        assert result["success"] is True
        assert result["passed"] == 1
        assert result["failed"] == 0


def test_sandbox_run_failing_test():
    """Test executing a failing test in temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        test_file = workspace / "test_fail.py"
        test_file.write_text("def test_bad(): assert 1 == 2\n", encoding="utf-8")

        runner = DockerTestRunner()
        result = runner.run_tests(workspace)

        assert result["success"] is False
        assert result["failed"] >= 1


# -------------------------------------------------------------------------
# Dependency-manifest detection
# -------------------------------------------------------------------------
def test_detect_dependency_install_args_requirements_txt():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "requirements.txt").write_text("somepkg==1.0\n", encoding="utf-8")
        assert _detect_dependency_install_args(workspace) == ["-r", "requirements.txt"]


def test_detect_dependency_install_args_pyproject_toml():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        assert _detect_dependency_install_args(workspace) == ["."]


def test_detect_dependency_install_args_setup_py():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "setup.py").write_text("from setuptools import setup\nsetup()\n", encoding="utf-8")
        assert _detect_dependency_install_args(workspace) == ["."]


def test_detect_dependency_install_args_none_when_no_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert _detect_dependency_install_args(Path(tmpdir)) is None


# -------------------------------------------------------------------------
# Subprocess fallback: dependency installation
# -------------------------------------------------------------------------
def test_subprocess_installs_requirements_before_running_pytest():
    """A requirements.txt in the workspace triggers an isolated pip install
    (pip is called with --target, never mutating RepoPilot's own env) before
    pytest runs, and the install log is folded into the returned output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "requirements.txt").write_text("somepkg==1.0\n", encoding="utf-8")
        (workspace / "test_math.py").write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "pip" in cmd:
                return _fake_completed(0, stdout="Successfully installed somepkg\n")
            return _fake_completed(0, stdout="1 passed in 0.01s\n")

        runner = DockerTestRunner()
        runner._docker_checked = True
        runner._docker_available = False  # force the subprocess path under test
        with patch("app.services.sandbox.docker_runner.subprocess.run", side_effect=fake_run):
            result = runner.run_tests(workspace)

        assert len(calls) == 2
        install_cmd, pytest_cmd = calls
        assert "pip" in install_cmd and "install" in install_cmd
        assert "--target" in install_cmd
        assert "-r" in install_cmd and "requirements.txt" in install_cmd
        assert "pytest" in pytest_cmd
        assert "Successfully installed somepkg" in result["output"]
        assert result["success"] is True


def test_subprocess_install_failure_does_not_block_pytest():
    """A failed (e.g. no-network) dependency install must not prevent pytest
    from running -- the install failure is surfaced, not fatal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "requirements.txt").write_text("somepkg==1.0\n", encoding="utf-8")
        (workspace / "test_math.py").write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

        def fake_run(cmd, **kwargs):
            if "pip" in cmd:
                return _fake_completed(1, stderr="ERROR: No internet connection\n")
            return _fake_completed(0, stdout="1 passed in 0.01s\n")

        runner = DockerTestRunner()
        runner._docker_checked = True
        runner._docker_available = False  # force the subprocess path under test
        with patch("app.services.sandbox.docker_runner.subprocess.run", side_effect=fake_run):
            result = runner.run_tests(workspace)

        assert "No internet connection" in result["output"]
        assert result["success"] is True


def test_subprocess_skips_install_step_when_no_manifest_present():
    """No requirements.txt/pyproject.toml/setup.py -> exactly one subprocess
    call (pytest only), matching the pre-existing behavior byte-for-byte."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "test_math.py").write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="1 passed in 0.01s\n")

        runner = DockerTestRunner()
        runner._docker_checked = True
        runner._docker_available = False  # force the subprocess path under test
        with patch("app.services.sandbox.docker_runner.subprocess.run", side_effect=fake_run):
            runner.run_tests(workspace)

        assert len(calls) == 1


def test_subprocess_install_target_dir_is_removed_after_run():
    """The isolated --target directory is temporary and always cleaned up."""
    created_dirs = []
    real_mkdtemp = tempfile.mkdtemp

    def spy_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "requirements.txt").write_text("somepkg==1.0\n", encoding="utf-8")
        (workspace / "test_math.py").write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

        runner = DockerTestRunner()
        runner._docker_checked = True
        runner._docker_available = False  # force the subprocess path under test
        with patch(
            "app.services.sandbox.docker_runner.subprocess.run",
            side_effect=lambda cmd, **kwargs: _fake_completed(0, stdout="1 passed\n"),
        ), patch("app.services.sandbox.docker_runner.tempfile.mkdtemp", side_effect=spy_mkdtemp):
            runner.run_tests(workspace)

    assert len(created_dirs) == 1
    assert not Path(created_dirs[0]).exists()


# -------------------------------------------------------------------------
# Docker path: safe command construction (no shell injection via test_path)
# -------------------------------------------------------------------------
def test_docker_command_installs_dependencies_and_passes_test_target_safely():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "requirements.txt").write_text("somepkg==1.0\n", encoding="utf-8")

        runner = DockerTestRunner()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = _fake_container(0, b"1 passed in 0.01s")
        runner._docker_client = mock_client

        malicious_test_path = "tests/test_x.py; rm -rf /"
        result = runner._run_in_docker(workspace, malicious_test_path, time.time())

        command = mock_client.containers.run.call_args.kwargs["command"]
        assert command[0:2] == ["sh", "-c"]
        assert "pip install" in command[2]
        assert "requirements.txt" in command[2]
        # The (untrusted) test target is a distinct argv element passed via
        # "$@", never interpolated into the shell script string itself.
        assert malicious_test_path not in command[2]
        assert command[-1] == malicious_test_path
        assert result["success"] is True
        # detach=True is required so a real timeout (Container.wait) can be
        # enforced; the old blocking, non-timeoutable run() must not return.
        assert mock_client.containers.run.call_args.kwargs["detach"] is True
        # The container is explicitly removed rather than relying on
        # run()'s remove=True/auto_remove (incompatible with detach=True).
        mock_client.containers.run.return_value.remove.assert_called_once_with(force=True)


def test_docker_command_skips_install_when_no_manifest_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        runner = DockerTestRunner()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = _fake_container()
        runner._docker_client = mock_client

        runner._run_in_docker(workspace, None, time.time())

        command = mock_client.containers.run.call_args.kwargs["command"]
        assert command == ["pytest", "-v", "-o", "testpaths=.", "."]


# ---------------------------------------------------------------------------
# ContainerError (a failing test, NOT a broken Docker daemon) must never
# disable Docker sandboxing -- Phase 3C hardening fix #1.
# ---------------------------------------------------------------------------
def test_container_error_reports_normal_failure_without_disabling_docker():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        runner = DockerTestRunner()
        mock_client = MagicMock()
        mock_client.containers.run.side_effect = ContainerError(
            container="fake-container-id",
            exit_status=1,
            command="pytest",
            image="python:3.11-slim",
            stderr=b"1 failed in 0.02s",
        )
        runner._docker_client = mock_client
        # Simulate an already-successful availability probe so we can prove
        # it survives untouched (assertion (c) below).
        runner._docker_checked = True
        runner._docker_available = True

        result = runner._run_in_docker(workspace, None, time.time())

        # (a) reported as a normal failed-test result, not a runner error
        assert result["success"] is False
        assert result["failed"] == 1
        assert result["exit_code"] == 1

        # (b) the process-wide flag must remain untouched
        assert docker_runner_module._GLOBAL_DOCKER_DISABLED is False

        # (c) Docker is still considered available for this runner
        assert runner.is_docker_available is True

        # (d) subsequent execution still goes through Docker, not the
        # subprocess fallback -- reconfigure the same mock to succeed and
        # confirm the Docker path (not subprocess) produced the result.
        mock_client.containers.run.side_effect = None
        mock_client.containers.run.return_value = _fake_container(0, b"1 passed in 0.01s")
        second = runner._run_in_docker(workspace, None, time.time())
        assert second["success"] is True
        assert second["passed"] == 1


def test_genuine_docker_api_error_disables_docker_and_falls_back_to_subprocess():
    """A real daemon-level failure (unreachable daemon, image not pulled,
    connection dropped, ...) is the one case that should still disable
    Docker sandboxing and degrade to the subprocess fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        (workspace / "test_math.py").write_text("def test_add(): assert 1 + 1 == 2\n", encoding="utf-8")

        runner = DockerTestRunner()
        mock_client = MagicMock()
        mock_client.containers.run.side_effect = APIError("Cannot connect to the Docker daemon")
        runner._docker_client = mock_client

        with patch(
            "app.services.sandbox.docker_runner.subprocess.run",
            side_effect=lambda cmd, **kwargs: _fake_completed(0, stdout="1 passed in 0.01s\n"),
        ):
            result = runner._run_in_docker(workspace, None, time.time())

        assert docker_runner_module._GLOBAL_DOCKER_DISABLED is True
        assert runner._docker_available is False
        # The subprocess fallback still ran and produced a real result.
        assert result["success"] is True


def test_docker_execution_timeout_kills_container_and_reports_timeout():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        runner = DockerTestRunner(timeout=5)
        mock_client = MagicMock()
        fake_container = MagicMock()
        fake_container.wait.side_effect = ReadTimeout("timed out waiting for container")
        mock_client.containers.run.return_value = fake_container
        runner._docker_client = mock_client

        result = runner._run_in_docker(workspace, None, time.time())

        assert result["success"] is False
        assert result["exit_code"] == 124
        assert "timed out after 5 seconds" in result["output"].lower()
        fake_container.kill.assert_called_once()
        fake_container.remove.assert_called_once_with(force=True)
        # A timeout is not a Docker-daemon failure.
        assert docker_runner_module._GLOBAL_DOCKER_DISABLED is False
