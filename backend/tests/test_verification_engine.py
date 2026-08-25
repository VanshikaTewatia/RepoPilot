"""Unit tests for VerificationEngine: ecosystem-aware execution wiring.

No real Docker or network calls are made -- Docker is exercised via a
MagicMock client (never invoked here since the CI/test environment has no
Docker daemon, so `is_docker_available` is naturally False) and subprocess
execution is exercised via `subprocess.run` patched with `unittest.mock`.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


# ---------------------------------------------------------------------------
# Unknown ecosystem: never pretend verification passed
# ---------------------------------------------------------------------------
def test_verify_unknown_ecosystem_reports_unavailable_without_running_anything():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "README.md", "no recognizable project here\n")

        calls = []
        with patch(
            "app.services.verification.engine.subprocess.run",
            side_effect=lambda *a, **k: calls.append(a) or _fake_completed(0),
        ):
            result = VerificationEngine().verify(root)

        assert result["success"] is False
        assert result["available"] is False
        assert result["ecosystem"] == "unknown"
        assert "requirements.txt" in result["output"] or "package.json" in result["output"]
        assert calls == []  # nothing was ever executed


def test_verify_nonexistent_workspace_reports_failure():
    result = VerificationEngine().verify("/definitely/not/a/real/path/xyz")
    assert result["success"] is False
    assert result["available"] is False


# ---------------------------------------------------------------------------
# Node: install + test script selection, executed via subprocess fallback
# ---------------------------------------------------------------------------
def test_verify_node_project_runs_test_script_via_subprocess():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[0] == "npm" and "ci" in cmd:
                return _fake_completed(0, stdout="added 42 packages\n")
            return _fake_completed(0, stdout="Tests: 0 failed, 3 passed, 3 total\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["success"] is True
        assert result["passed"] == 3
        assert result["failed"] == 0
        assert len(calls) == 2
        assert calls[0] == ["npm", "ci"]
        assert calls[1] == ["npm", "test"]


def test_verify_node_project_falls_back_to_build_script():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"build": "next build"}}))

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="Compiled successfully\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["success"] is True
        assert calls[-1] == ["npm", "run", "build"]


def test_verify_node_project_with_no_test_or_build_script_is_unavailable():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"start": "node index.js"}}))

        calls = []
        with patch(
            "app.services.verification.engine.subprocess.run",
            side_effect=lambda *a, **k: calls.append(a) or _fake_completed(0),
        ):
            result = VerificationEngine().verify(root)

        assert result["ecosystem"] == "node"
        assert result["available"] is False
        assert result["success"] is False
        assert calls == []


def test_verify_node_project_never_runs_pytest():
    """Regression test for the reported bug: a Next.js repo must never be
    verified with pytest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "next lint"}}))

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="0 failed, 1 passed\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            engine.verify(root)

        assert all("pytest" not in cmd for cmd in calls)


# ---------------------------------------------------------------------------
# Install failure must not block the test command (mirrors Python's semantics)
# ---------------------------------------------------------------------------
def test_verify_node_install_failure_does_not_block_test_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))

        def fake_run(cmd, **kwargs):
            if "install" in cmd:
                return _fake_completed(1, stderr="ERROR: no network\n")
            return _fake_completed(0, stdout="0 failed, 2 passed\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert "no network" in result["output"]
        assert result["success"] is True
        assert result["passed"] == 2


# ---------------------------------------------------------------------------
# Go / Rust / Java / .NET: at least one full round trip each through the engine
# ---------------------------------------------------------------------------
def test_verify_go_project_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[:2] == ["go", "mod"]:
                return _fake_completed(0)
            return _fake_completed(1, stdout="--- PASS: TestA\n--- FAIL: TestB\nFAIL")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert result["ecosystem"] == "go"
        assert result["success"] is False
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert calls[-1] == ["go", "test", "./...", "-v"]


def test_verify_rust_project_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "Cargo.toml", "[package]\nname='app'\n")

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["cargo", "fetch"]:
                return _fake_completed(0)
            return _fake_completed(0, stdout="test result: ok. 4 passed; 0 failed; 0 ignored\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert result["ecosystem"] == "rust"
        assert result["success"] is True
        assert result["passed"] == 4


def test_verify_java_maven_project_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pom.xml", "<project></project>")

        def fake_run(cmd, **kwargs):
            return _fake_completed(0, stdout="Tests run: 5, Failures: 0, Errors: 0, Skipped: 0\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert result["ecosystem"] == "java-maven"
        assert result["success"] is True
        assert result["passed"] == 5


def test_verify_dotnet_project_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "App.sln", "solution\n")

        def fake_run(cmd, **kwargs):
            if cmd == ["dotnet", "restore"]:
                return _fake_completed(0)
            return _fake_completed(1, stdout="Passed!  - Failed: 1, Passed: 4, Skipped: 0, Total: 5\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert result["ecosystem"] == "dotnet"
        assert result["success"] is False
        assert result["passed"] == 4
        assert result["failed"] == 1


# ---------------------------------------------------------------------------
# Python still routes through the existing, unmodified DockerTestRunner
# ---------------------------------------------------------------------------
def test_verify_python_project_delegates_to_existing_sandbox_and_preserves_behavior():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        result = VerificationEngine().verify(root)

        assert result["ecosystem"] == "python"
        assert result["success"] is True
        assert result["passed"] == 1
        assert result["failed"] == 0


# ---------------------------------------------------------------------------
# tools.run_tests wiring
# ---------------------------------------------------------------------------
def test_tools_run_tests_delegates_to_verification_engine():
    from app.services.agent import tools

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_engine = MagicMock()
        fake_engine.verify.return_value = {"success": True, "ecosystem": "node"}

        with patch("app.services.agent.tools.VerificationEngine", return_value=fake_engine):
            result = tools.run_tests(tmpdir, test_path="some/target")

        fake_engine.verify.assert_called_once_with(workspace_path=tmpdir, test_path="some/target")
        assert result == {"success": True, "ecosystem": "node"}
