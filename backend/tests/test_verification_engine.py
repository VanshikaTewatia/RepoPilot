"""Unit tests for VerificationEngine: ecosystem-aware execution wiring.

No real Docker or network calls are made -- Docker is exercised via a
MagicMock client (never invoked here since the CI/test environment has no
Docker daemon, so `is_docker_available` is naturally False) and subprocess
execution is exercised via `subprocess.run` patched with `unittest.mock`.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ReadTimeout

from app.core.config import settings
from app.services.verification.adapters.node_adapter import NodeAdapter
from app.services.verification.engine import VerificationEngine, _INSTALL_TIMEOUT_SECONDS


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
    """Fake docker-py ``Container`` as returned by ``containers.run(detach=True, ...)``:
    exposes ``.wait()``, ``.logs()``, ``.kill()``, and ``.remove()`` the way
    ``_execute_in_docker`` now drives them directly (instead of relying on
    the synchronous ``containers.run()`` helper, which has no timeout)."""
    container = MagicMock()
    container.wait.return_value = {"StatusCode": status_code}
    container.logs.return_value = logs
    return container


def _tool_available():
    """Patch the engine's host-side toolchain preflight check to pass.

    Whether e.g. real `go` or `cargo` happen to be installed on the machine
    running these tests is irrelevant to what's being verified here (the
    engine's command-selection and result-parsing logic) -- only the
    dedicated "toolchain missing" tests below exercise the preflight itself,
    by patching this the other way.
    """
    return patch("app.services.verification.engine._tool_is_available", return_value=True)


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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
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
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "dotnet"
        assert result["success"] is False
        assert result["passed"] == 4
        assert result["failed"] == 1


# ---------------------------------------------------------------------------
# Phase 7: Python now routes through the same generic _run_adapter path
# every other ecosystem already uses (no more DockerTestRunner.run_tests()
# special case in VerificationEngine.verify()).
# ---------------------------------------------------------------------------
def test_verify_python_project_delegates_to_existing_sandbox_and_preserves_behavior():
    """A manifest-less Python project (detected via PythonAdapter's
    test_*.py-glob fallback, same as before Phase 7) has no install step at
    all -- adapter.install_command() returns None -- so this exercises pure
    pytest execution with zero pip/network involvement, and must keep
    passing exactly as it did through the old delegation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        engine = VerificationEngine()
        # Force the subprocess path so this test is deterministic regardless
        # of whether a real Docker daemon happens to be running/reachable in
        # the host environment (the sandbox's own real Docker path is
        # covered separately in test_sandbox.py).
        engine._docker_runner._docker_checked = True
        engine._docker_runner._docker_available = False
        result = engine.verify(root)

        assert result["ecosystem"] == "python"
        assert result["available"] is True
        assert result["success"] is True
        assert result["passed"] == 1
        assert result["failed"] == 0


def test_verify_python_project_uses_python_adapter_docker_image():
    """PythonAdapter.docker_image (the pre-baked repopilot-sandbox-python
    image carrying pytest) must actually be what _run_adapter passes to
    Docker for a Python project -- not settings.docker_sandbox_image, and
    not a hardcoded literal in the engine."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", True), \
             patch.object(engine, "_execute_in_docker", return_value=("2 passed", 0)) as mock_exec, \
             _tool_available():
            result = engine.verify(root)

        assert result["success"] is True
        from app.services.verification.adapters.python_adapter import PythonAdapter
        assert mock_exec.call_args.args[1] == PythonAdapter.docker_image
        assert mock_exec.call_args.args[1] != "python:3.11-slim"


def test_verify_python_project_with_dependencies_installs_isolated_and_mounts_readonly():
    """A dependency-bearing Python project: the isolated ('outside the
    container') pip-install step must run BEFORE the container, its result
    mounted read-only with PYTHONPATH set, and the container itself must
    receive no install step (install_argv=None) of its own -- i.e. no
    network is needed inside the network_mode="none" container at all."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\ndependencies=['requests']\n")
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", True), \
             patch(
                 "app.services.verification.engine.subprocess.run",
                 return_value=_fake_completed(0, stdout="Successfully installed x\n"),
             ) as mock_run, \
             patch.object(engine, "_execute_in_docker", return_value=("2 passed", 0)) as mock_exec:
            result = engine.verify(root)

        assert result["success"] is True
        assert result["available"] is True

        # The isolated install actually ran, via python -m pip --target.
        install_cmd = mock_run.call_args.args[0]
        assert "--target" in install_cmd
        assert install_cmd[0] != "pip"  # sys.executable, never a bare "pip"

        # The container call received install_argv=None (no in-container
        # install/network step) plus the read-only mount + PYTHONPATH.
        _, _, install_argv, _test_argv = mock_exec.call_args.args
        assert install_argv is None
        kwargs = mock_exec.call_args.kwargs
        assert kwargs["extra_env"] == {"PYTHONPATH": "/repopilot-deps"}
        (mount_path,) = kwargs["extra_volumes"].keys()
        assert kwargs["extra_volumes"][mount_path]["mode"] == "ro"


def test_verify_python_project_without_dependencies_skips_install_and_runs_pytest_for_real():
    """Regression test (Phase 6 real-repo end-to-end validation): a
    pyproject.toml with no declared [project].dependencies -- the exact
    shape of the real disposable GitHub fixture used for that validation
    (name/version plus only a [build-system] table, flat two-file layout,
    no [tool.setuptools] package/module config) -- must never trigger
    `pip install .`. Before the fix, this repo's own pyproject.toml alone
    was treated as sufficient reason to install it as a package, and the
    build failed with setuptools' "Multiple top-level modules discovered
    in a flat-layout" safety check -- deterministically, regardless of
    whether the reported bug was actually fixed, which meant verification
    could never succeed against it.

    Uses the real subprocess.run (wrapped, not replaced) so this proves
    pytest genuinely executed and produced a real result, not merely that
    no exception was raised.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(
            root,
            "pyproject.toml",
            "[project]\n"
            "name = 'calc-fixture'\n"
            "version = '0.1.0'\n"
            "\n"
            "[build-system]\n"
            "requires = ['setuptools']\n"
            "build-backend = 'setuptools.build_meta'\n",
        )
        _write(root, "calculator.py", "def calculate_total(price, quantity):\n    return price + quantity\n")
        _write(
            root,
            "test_calculator.py",
            "from calculator import calculate_total\n\n\ndef test_calculate_total():\n    assert calculate_total(10, 3) == 13\n",
        )

        engine = VerificationEngine()
        engine._docker_runner._docker_checked = True
        engine._docker_runner._docker_available = False

        with patch("app.services.verification.engine.subprocess.run", wraps=subprocess.run) as spy_run:
            result = engine.verify(root)

        assert result["ecosystem"] == "python"
        assert result["available"] is True
        assert result["success"] is True
        assert result["passed"] == 1
        assert result["failed"] == 0

        # No isolated pip-install was ever attempted -- only the real
        # pytest invocation itself.
        install_calls = [c for c in spy_run.call_args_list if "--target" in c.args[0]]
        assert install_calls == []
        assert spy_run.call_count == 1


def test_verify_python_project_isolated_install_failure_is_unable_to_verify_not_failed():
    """A failed isolated dependency install must classify as available=False
    (an environment/setup failure) -- never as a generic test failure --
    and the network-isolated container must never even be started."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "requirements.txt", "some-package-that-does-not-resolve==999.999\n")
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", True), \
             patch(
                 "app.services.verification.engine.subprocess.run",
                 return_value=_fake_completed(1, stderr="ERROR: Could not find a version\n"),
             ), \
             patch.object(engine, "_execute_in_docker") as mock_exec:
            result = engine.verify(root)

        mock_exec.assert_not_called()
        assert result["available"] is False
        assert result["success"] is False
        assert "not evidence that the reported issue does or does not exist" in result["detail"]
        assert "Could not find a version" in result["output"]


def test_verify_python_project_genuine_test_failure_is_still_available_true():
    """A real pytest failure (no install involved -- manifest-less project)
    must remain a genuine, retryable test failure: available=True,
    success=False -- proving Phase 7's classification fix distinguishes a
    real failure from an environment failure rather than conflating them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 3\n")

        engine = VerificationEngine()
        engine._docker_runner._docker_checked = True
        engine._docker_runner._docker_available = False
        result = engine.verify(root)

        assert result["ecosystem"] == "python"
        assert result["available"] is True
        assert result["success"] is False
        assert result["failed"] == 1


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


# ---------------------------------------------------------------------------
# verify_repository: task-aware, multi-project verification
# ---------------------------------------------------------------------------
def test_verify_repository_single_project_delegates_exactly_like_verify():
    """A single-ecosystem repository must behave byte-for-byte like plain
    verify() -- no wrapping, no behavior change for existing callers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        engine = VerificationEngine()
        engine._docker_runner._docker_checked = True
        engine._docker_runner._docker_available = False
        result = engine.verify_repository(root, task_description="fix math", keyword_matches=[])

        assert result["ecosystem"] == "python"
        assert result["success"] is True
        assert result["project_root"] == "."


def test_verify_repository_selects_only_the_relevant_project_in_a_monorepo():
    """Spring backend + React frontend monorepo: a task about the React
    product card must verify only the frontend project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "backend/pom.xml", "<project></project>")
        _write(root, "frontend/package.json", json.dumps({
            "dependencies": {"react": "^18.0.0"},
            "scripts": {"test": "jest"},
        }))

        def fake_run(cmd, **kwargs):
            return _fake_completed(0, stdout="Tests: 0 failed, 2 passed, 2 total\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify_repository(
                root,
                task_description="Fix the React product card",
                keyword_matches=[{"file": "frontend/src/ProductCard.jsx"}],
            )

        assert result["project_root"] == "frontend"
        assert result["ecosystem"] == "node"
        assert result["success"] is True
        detected_roots = sorted(p["root"] for p in result["detected_projects"])
        assert detected_roots == ["backend", "frontend"]


def test_verify_repository_verifies_all_relevant_projects_when_ambiguous():
    """When the task doesn't clearly point at one project, verify every
    detected project rather than silently guessing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "backend/pom.xml", "<project></project>")
        _write(root, "frontend/package.json", json.dumps({"scripts": {"test": "jest"}}))

        def fake_run(cmd, **kwargs):
            if cmd[:1] == ["mvn"]:
                return _fake_completed(0, stdout="Tests run: 2, Failures: 0, Errors: 0, Skipped: 0\n")
            return _fake_completed(0, stdout="Tests: 0 failed, 1 passed, 1 total\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify_repository(root, task_description="fix the login bug", keyword_matches=[])

        assert result["success"] is True
        verified_roots = sorted(r["project_root"] for r in result["project_results"])
        assert verified_roots == ["backend", "frontend"]
        assert result["passed"] == 3


# ---------------------------------------------------------------------------
# Node: pnpm / yarn package manager selection (lockfile-driven)
# ---------------------------------------------------------------------------
def test_verify_node_project_with_pnpm_lockfile_uses_pnpm():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "vitest run"}}))
        _write(root, "pnpm-lock.yaml")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="0 failed, 5 passed\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["success"] is True
        assert result["passed"] == 5
        assert calls[0] == ["pnpm", "install"]
        assert calls[1] == ["pnpm", "test"]


def test_verify_node_project_with_yarn_lockfile_uses_yarn():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "yarn.lock")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="Tests: 0 failed, 4 passed, 4 total\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["success"] is True
        assert result["passed"] == 4
        assert calls[0] == ["yarn", "install"]
        assert calls[1] == ["yarn", "test"]


# ---------------------------------------------------------------------------
# Missing toolchain: reported precisely, and nothing is ever actually run
# ---------------------------------------------------------------------------
def test_verify_node_project_reports_unable_to_verify_when_npm_missing():
    """The exact bug report this fix addresses: a real Node/React repo must
    never be silently treated as passed/failed when npm isn't available in
    the execution environment -- it must be reported UNABLE_TO_VERIFY, and
    the test command must never actually be invoked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "react-scripts test"}}))
        _write(root, "package-lock.json")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0)

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             patch("app.services.verification.engine._tool_is_available", return_value=False):
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["success"] is False
        assert result["available"] is False
        assert result["exit_code"] == 127
        assert "npm" in result["detail"]
        assert calls == []  # the test command was never actually executed


def test_verify_node_project_own_script_exit_127_is_not_misclassified_as_missing_npm():
    """Regression test for Task #15's actual root cause: npm is present and
    the install step succeeds, but the project's OWN test script (e.g.
    react-scripts, never actually installed for some unrelated reason) exits
    127 when the shell can't find it. This must be reported as a genuine
    test failure -- available for verification, just failing -- never as
    "npm is missing", since npm ran fine the whole time."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "react-scripts test"}}))
        _write(root, "package-lock.json")

        def fake_run(cmd, **kwargs):
            if "ci" in cmd:
                return _fake_completed(0, stdout="added 10 packages\n")
            return _fake_completed(127, stderr="sh: 1: react-scripts: not found\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["exit_code"] == 127
        assert result["available"] is True
        assert result["success"] is False
        assert "npm" not in (result["detail"] or "")
        assert "react-scripts" in result["output"]


def test_verify_go_project_reports_unable_to_verify_when_go_missing():
    """A second ecosystem exercising the same preflight mechanism, proving
    it's generic rather than Node-specific."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0)

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             patch("app.services.verification.engine._tool_is_available", return_value=False):
            result = engine.verify(root)

        assert result["ecosystem"] == "go"
        assert result["available"] is False
        assert result["success"] is False
        assert result["exit_code"] == 127
        assert "go" in result["detail"]
        assert calls == []


# ---------------------------------------------------------------------------
# Java: mvnw / gradlew wrapper preference
# ---------------------------------------------------------------------------
def test_verify_java_maven_project_prefers_mvnw_wrapper_when_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pom.xml", "<project></project>")
        mvnw = root / "mvnw"
        mvnw.write_text("#!/bin/sh\nexec mvn \"$@\"\n", encoding="utf-8")
        mvnw.chmod(0o644)  # deliberately non-executable, as some checkouts leave it

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="Tests run: 3, Failures: 0, Errors: 0, Skipped: 0\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "java-maven"
        assert result["success"] is True
        assert result["passed"] == 3
        assert calls[0][0] == "./mvnw"
        # the engine restores the wrapper's executable bit rather than
        # leaving it unusable
        assert os.access(mvnw, os.X_OK)


def test_verify_java_maven_project_uses_mvn_when_no_wrapper_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pom.xml", "<project></project>")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="Tests run: 1, Failures: 0, Errors: 0, Skipped: 0\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["success"] is True
        assert calls[0][0] == "mvn"


def test_verify_java_gradle_project_prefers_gradlew_wrapper_when_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "build.gradle", "plugins { id 'java' }\n")
        gradlew = root / "gradlew"
        gradlew.write_text("#!/bin/sh\nexec gradle \"$@\"\n", encoding="utf-8")
        gradlew.chmod(0o644)

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="BUILD SUCCESSFUL\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "java-gradle"
        assert result["success"] is True
        assert calls[0][0] == "./gradlew"
        assert os.access(gradlew, os.X_OK)


def test_verify_java_gradle_project_uses_gradle_when_no_wrapper_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "build.gradle.kts", "plugins { java }\n")

        def fake_run(cmd, **kwargs):
            return _fake_completed(1, stdout="BUILD FAILED\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "java-gradle"
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Flutter / Dart
# ---------------------------------------------------------------------------
def test_verify_flutter_project_runs_flutter_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pubspec.yaml", "name: app\ndependencies:\n  flutter:\n    sdk: flutter\n")
        _write(root, "test/widget_test.dart", "void main() {}\n")

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _fake_completed(0, stdout="00:03 +5: All tests passed!\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "flutter"
        assert result["success"] is True
        assert result["passed"] == 5
        assert calls[-1] == ["flutter", "test"]


def test_verify_dart_project_runs_dart_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pubspec.yaml", "name: pkg\ndependencies:\n  path: ^1.8.0\n")
        _write(root, "test/pkg_test.dart", "void main() {}\n")

        def fake_run(cmd, **kwargs):
            return _fake_completed(1, stdout="00:02 +2 -1: some tests failed.\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run), \
             _tool_available():
            result = engine.verify(root)

        assert result["ecosystem"] == "dart"
        assert result["success"] is False
        assert result["passed"] == 2
        assert result["failed"] == 1


# ---------------------------------------------------------------------------
# Docker path: per-ecosystem image selection (the actual root-cause fix)
# ---------------------------------------------------------------------------
def test_verify_node_project_in_docker_uses_node_image_not_shared_python_image():
    """Integration-style: Node repository -> Node adapter -> correct package
    manager -> verification command -> successful result, executed through
    the Docker path with the toolchain available in the (mocked) containers.

    This is the regression test for the reported bug: previously every
    ecosystem ran in the single shared `python:3.11-slim` image, so a Node
    repo's `npm test` failed with "npm: not found" even though Docker itself
    was available and the correct command was selected. The fix is that the
    adapter's own image (node:20-slim) is what gets run, not the Python
    sandbox's image.

    Phase 8 (Task #31): Node verification now makes TWO containers.run()
    calls -- the isolated, network-enabled dependency-preparation container
    first, then the network-isolated verification container that actually
    runs `npm test` -- so this asserts against each container distinctly
    rather than a single shared mock.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        engine = VerificationEngine()
        fake_docker_client = MagicMock()
        prep_container = _fake_container(0, b"added 6 packages in 1s\n")
        test_container = _fake_container(
            0, b"PASS  src/App.test.js\nTests: 0 failed, 6 passed, 6 total\n"
        )
        fake_docker_client.containers.run.side_effect = [prep_container, test_container]

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["success"] is True
        assert result["passed"] == 6
        assert result["available"] is True

        assert fake_docker_client.containers.run.call_count == 2
        prep_kwargs, test_kwargs = (
            call.kwargs for call in fake_docker_client.containers.run.call_args_list
        )

        # Dependency-preparation container: same Node image, but network
        # ENABLED (network_mode omitted) -- the one deliberate exception.
        assert prep_kwargs["image"] == "node:20-slim"
        assert "network_mode" not in prep_kwargs
        prep_container.wait.assert_called_once_with(timeout=_INSTALL_TIMEOUT_SECONDS)
        prep_container.remove.assert_called_once_with(force=True)

        # Verification container: still the adapter's own image, never the
        # shared Python sandbox image, and stays network-isolated.
        assert test_kwargs["image"] == "node:20-slim"
        assert test_kwargs["image"] != settings.docker_sandbox_image
        assert test_kwargs["network_mode"] == engine.network_mode
        # detach=True is required so the engine can apply its own
        # wait(timeout=...) instead of the unbounded synchronous helper.
        assert test_kwargs["detach"] is True
        command = test_kwargs["command"]
        assert command[-2:] == ["npm", "test"]
        # the preflight check for npm's presence is embedded in the script
        # run before test, not skipped
        assert "command -v npm" in command[2]
        # No install step embedded in the verification container's script --
        # dependencies were already prepared by the isolated container above.
        assert "npm ci" not in command[2]
        assert "npm install" not in command[2]

        test_container.wait.assert_called_once_with(timeout=engine.timeout)
        test_container.remove.assert_called_once_with(force=True)


def test_verify_go_project_in_docker_uses_go_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")

        engine = VerificationEngine()
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = _fake_container(
            0, b"--- PASS: TestA\nok  \texample.com/app\t0.004s\n"
        )

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "go"
        assert result["success"] is True
        run_kwargs = fake_docker_client.containers.run.call_args.kwargs
        assert run_kwargs["image"] == "golang:1.22-alpine"


def test_verify_docker_reports_missing_tool_via_preflight_sentinel():
    """A container that actually ran but hit the preflight's "not found"
    exit must be reported as UNABLE_TO_VERIFY, not as a passed/failed test
    result -- and only because OUR preflight sentinel fired, not because of
    a bare exit code 127 (see the Task #15 regression tests above)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "Cargo.toml", "[package]\nname='app'\n")

        engine = VerificationEngine()
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = _fake_container(
            127, b"REPOPILOT_TOOLCHAIN_MISSING:cargo\n"
        )

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "rust"
        assert result["available"] is False
        assert result["success"] is False
        assert "cargo" in result["detail"]


# ---------------------------------------------------------------------------
# Docker path: hard execution timeout (settings.sandbox_timeout_seconds)
# ---------------------------------------------------------------------------
def test_verify_docker_execution_timeout_is_enforced_and_container_removed():
    """A hung install/test inside the sandbox must not block indefinitely.
    Container.wait(timeout=...) -- the Docker SDK's own timeout mechanism --
    bounds it; the container is killed and still reliably removed.

    Phase 8 (Task #31): dependency preparation for this Node project happens
    in its own, separate container first (mocked here as succeeding
    normally); the hang being tested is in the SECOND, network-isolated
    verification container that actually runs the test command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        engine = VerificationEngine(timeout=5)
        prep_container = _fake_container(0, b"added 6 packages in 1s\n")
        hung_container = MagicMock()
        hung_container.wait.side_effect = ReadTimeout("timed out")
        hung_container.logs.return_value = b"REPOPILOT_PHASE:test\n=== RUN TestSlow\n"
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.side_effect = [prep_container, hung_container]

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        # Dependency preparation succeeded normally and was cleaned up.
        prep_container.wait.assert_called_once_with(timeout=_INSTALL_TIMEOUT_SECONDS)
        prep_container.remove.assert_called_once_with(force=True)

        # The verification container is the one that hung and was killed.
        hung_container.wait.assert_called_once_with(timeout=5)
        hung_container.kill.assert_called_once()
        hung_container.remove.assert_called_once_with(force=True)

        assert result["exit_code"] == 124
        assert "timed out" in result["output"].lower()
        assert result["success"] is False
        # a bounded timeout during the test phase (not install) is a
        # controlled result, not a toolchain/install/environment verdict
        assert result["available"] is True


# ---------------------------------------------------------------------------
# Docker path: dependency-install failure must never be reported as a
# missing toolchain, and (Phase 8 / Task #31) Node's own install step is no
# longer subject to SANDBOX_NETWORK_MODE=none at all -- it moved to a
# separate, network-enabled container -- which is the actual Task #31 fix.
# ---------------------------------------------------------------------------
def test_verify_docker_node_dependency_prep_failure_reports_unable_to_verify_and_test_never_runs():
    """Historically (Task #15) `npm ci` failed because the sandbox had no
    network access under network_mode="none", surfacing as 'Required tool
    npm is not available' once npm test then hit its own missing
    devDependency. Task #31's fix moves Node's dependency preparation into a
    separate, network-ENABLED container specifically so that no-network
    failure can no longer happen to Node at all -- proven here by asserting
    the preparation container gets real network even though the engine
    itself is configured with network_mode="none".

    What must still hold regardless of *why* preparation failed: npm is
    never blamed as a missing tool, dependency-preparation failure is
    reported as an environment/setup problem (available=False), and the
    verification container -- which would run the test command against
    incomplete dependencies -- must never even be created. This is a
    stronger, more direct guarantee than the old single-container script's
    in-script short-circuit ordering, since the second container simply
    never exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "react-scripts test"}}))
        _write(root, "package-lock.json")

        engine = VerificationEngine(network_mode="none")
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = _fake_container(
            1,
            b"npm error 404 Not Found - GET https://registry.npmjs.org/some-dep\n",
        )

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["available"] is False
        assert result["success"] is False
        assert "npm" not in result["detail"]
        assert "could not be prepared" in result["detail"]

        # Only the dependency-preparation container was ever created -- the
        # network-isolated verification container ("react-scripts test")
        # never ran against incomplete/absent dependencies.
        fake_docker_client.containers.run.assert_called_once()
        run_kwargs = fake_docker_client.containers.run.call_args.kwargs
        assert "network_mode" not in run_kwargs  # real network, not "none"
        assert "react-scripts" not in " ".join(run_kwargs["command"])


# ---------------------------------------------------------------------------
# .NET, Go, Rust missing-toolchain coverage (subprocess path) for completeness
# ---------------------------------------------------------------------------
def test_verify_dotnet_project_reports_unable_to_verify_when_dotnet_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "App.sln", "solution\n")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run") as mock_run, \
             patch("app.services.verification.engine._tool_is_available", return_value=False):
            result = engine.verify(root)

        assert result["ecosystem"] == "dotnet"
        assert result["available"] is False
        mock_run.assert_not_called()


def test_verify_repository_unsupported_ecosystem_reports_unavailable():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "README.md", "no recognizable project here\n")

        engine = VerificationEngine()
        result = engine.verify_repository(root, task_description="fix the docs")

        assert result["available"] is False


# ---------------------------------------------------------------------------
# Phase 8 (Task #31): phase-marker timeout classification
#
# Exercised via Go (rather than Node) because Go still passes its
# install_argv straight through to _execute_in_docker -- proving this fix is
# a shared, ecosystem-agnostic mechanism in the engine itself, not something
# specific to Node's own new dependency-preparation container (see the
# dedicated Node tests further below for that).
# ---------------------------------------------------------------------------
def test_verify_docker_timeout_during_install_is_classified_as_environment_failure():
    """The Task #31 mechanism: a Docker wait() timeout that happened while
    the container was still inside the dependency-install step (confirmed
    via the REPOPILOT_PHASE:install marker, with no REPOPILOT_PHASE:test
    marker ever seen) must be classified as an environment/dependency
    failure -- available=False -- never an ordinary retryable test failure.
    Diagnostic output actually produced before the kill must be preserved,
    not discarded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")

        engine = VerificationEngine(timeout=5)
        fake_container = MagicMock()
        fake_container.wait.side_effect = ReadTimeout("timed out")
        fake_container.logs.return_value = (
            b"REPOPILOT_PHASE:install\n"
            b"go: downloading example.com/dep v1.2.3\n"
        )
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = fake_container

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "go"
        assert result["available"] is False
        assert result["success"] is False
        assert "environment" in result["detail"].lower()
        # Diagnostic output retained, not discarded.
        assert "go: downloading example.com/dep" in result["output"]
        fake_container.logs.assert_called()
        fake_container.kill.assert_called_once()


def test_verify_docker_timeout_during_test_phase_is_not_classified_as_environment_failure():
    """A timeout that happened AFTER the install step completed and the
    project's own test command had already started (confirmed via the
    REPOPILOT_PHASE:test marker) must NOT be auto-classified as an
    environment failure -- it may be a genuinely slow or hanging test
    suite, which existing retry semantics must still be allowed to handle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")

        engine = VerificationEngine(timeout=5)
        fake_container = MagicMock()
        fake_container.wait.side_effect = ReadTimeout("timed out")
        fake_container.logs.return_value = (
            b"REPOPILOT_PHASE:install\n"
            b"go: downloaded example.com/dep v1.2.3\n"
            b"REPOPILOT_PHASE:test\n"
            b"=== RUN TestSlow\n"
        )
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = fake_container

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "go"
        assert result["available"] is True  # never auto-classified as environment failure
        assert result["success"] is False
        assert result["exit_code"] == 124
        assert "TestSlow" in result["output"]


def test_verify_docker_timeout_with_no_phase_marker_is_left_ordinary_and_retains_logs():
    """No phase marker observed at all (e.g. the container was killed before
    even the install step's own marker had printed) must be handled
    conservatively -- never guessed at as either phase -- while whatever
    partial output the container had actually produced is still preserved,
    not discarded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")

        engine = VerificationEngine(timeout=5)
        fake_container = MagicMock()
        fake_container.wait.side_effect = ReadTimeout("timed out")
        fake_container.logs.return_value = b"some partial startup output\n"
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = fake_container

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["available"] is True  # never guessed at
        assert "some partial startup output" in result["output"]
        fake_container.logs.assert_called()


# ---------------------------------------------------------------------------
# Phase 8 (Task #31): Node dependency preparation -- isolated,
# network-enabled container; verification container stays network-isolated
# ---------------------------------------------------------------------------
def test_verify_node_project_with_dependencies_prepares_isolated_and_mounts_readonly():
    """A dependency-bearing Node project: dependency preparation must run in
    the isolated, network-enabled helper BEFORE the network-isolated
    verification container, its result (node_modules only) mounted
    read-only at /workspace/node_modules, the verification container itself
    must receive no install step of its own (no network needed inside
    network_mode="none" at all), and the staging directory must be cleaned
    up afterward -- mirroring the shape of Python's isolated-install fix
    without merging the two mechanics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        deps_dir = tempfile.mkdtemp()
        (Path(deps_dir) / "node_modules").mkdir()

        engine = VerificationEngine(network_mode="none")
        with patch.object(type(engine._docker_runner), "is_docker_available", True), \
             patch.object(
                 engine, "_prepare_node_deps_isolated",
                 return_value=(deps_dir, "added 1 package\n"),
             ) as mock_prep, \
             patch.object(engine, "_execute_in_docker", return_value=("2 passed", 0)) as mock_exec:
            result = engine.verify(root)

        assert result["success"] is True
        assert result["available"] is True
        mock_prep.assert_called_once()

        # The container call received install_argv=None (no in-container
        # install/network step) plus the read-only node_modules mount.
        _, _, install_argv, _test_argv = mock_exec.call_args.args
        assert install_argv is None
        kwargs = mock_exec.call_args.kwargs
        assert kwargs.get("extra_env") is None
        (mount_path,) = kwargs["extra_volumes"].keys()
        assert mount_path == str(Path(deps_dir) / "node_modules")
        assert kwargs["extra_volumes"][mount_path] == {"bind": "/workspace/node_modules", "mode": "ro"}

        # Cleaned up reliably afterward.
        assert not Path(deps_dir).exists()


def test_prepare_node_deps_isolated_uses_ignore_scripts_and_network_enabled_container():
    """Security regression: the dependency-preparation container must use
    the project's correct package manager (per existing lockfile
    detection), force --ignore-scripts so no repository-declared lifecycle
    script ever executes, use the fixed adapter-declared image (never an
    arbitrary/repository-influenced one), get real (non-"none") network
    access, and never mount the Docker socket."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "pnpm-lock.yaml")

        fake_container = MagicMock()
        fake_container.wait.return_value = {"StatusCode": 0}
        fake_container.logs.return_value = b"Done\n"
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = fake_container

        engine = VerificationEngine()
        engine._docker_runner._docker_client = fake_docker_client

        staging_dir = None
        try:
            staging_dir, log = engine._prepare_node_deps_isolated(root, NodeAdapter())

            assert staging_dir is not None
            run_kwargs = fake_docker_client.containers.run.call_args.kwargs
            script = run_kwargs["command"][-1]
            assert "pnpm install" in script
            assert "--ignore-scripts" in script
            assert run_kwargs["image"] == "node:20-slim"
            assert "network_mode" not in run_kwargs  # real (bridged) network, not "none"
            assert "/var/run/docker.sock" not in str(run_kwargs.get("volumes", {}))
            assert run_kwargs["user"] == "1000:1000"
        finally:
            if staging_dir:
                shutil.rmtree(staging_dir, ignore_errors=True)


def test_verify_node_project_dependency_preparation_failure_is_unable_to_verify_not_failed():
    """A failed dependency-preparation container run must classify as
    available=False (an environment/setup failure) -- never as a generic
    test failure -- and the network-isolated verification container must
    never even be started."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        fake_container = MagicMock()
        fake_container.wait.return_value = {"StatusCode": 1}
        fake_container.logs.return_value = b"npm error 404 Not Found - some-bad-dep\n"
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = fake_container

        engine = VerificationEngine(network_mode="none")
        with patch.object(type(engine._docker_runner), "is_docker_available", True), \
             patch.object(engine, "_execute_in_docker") as mock_exec:
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        mock_exec.assert_not_called()
        assert result["ecosystem"] == "node"
        assert result["available"] is False
        assert result["success"] is False
        assert "not evidence that the reported issue does or does not exist" in result["detail"]
        assert "some-bad-dep" in result["output"]


def test_node_dependency_prep_command_forces_ignore_scripts():
    """Security regression: the isolated, network-enabled dependency-
    preparation command must always disable lifecycle scripts, for every
    package-manager selection, so a repository's own preinstall/install/
    postinstall hooks never execute even though this step has real network
    access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")
        assert NodeAdapter().dependency_prep_command(root) == ["npm", "ci", "--ignore-scripts"]

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "pnpm-lock.yaml")
        assert NodeAdapter().dependency_prep_command(root) == ["pnpm", "install", "--ignore-scripts"]

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "yarn.lock")
        assert NodeAdapter().dependency_prep_command(root) == ["yarn", "install", "--ignore-scripts"]


def test_verify_node_project_never_runs_npm_directly_on_host_when_docker_available():
    """Security regression: when Docker is available, Node dependency
    preparation must go through the isolated container -- subprocess.run
    (direct host execution) must never be invoked at all, so no repository's
    npm lifecycle scripts can ever run with the RepoPilot backend's own
    privileges. This holds even though the engine is configured with
    network_mode="none": that setting only ever applies to the verification
    container, never to whether host subprocess execution is used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        prep_container = _fake_container(0, b"added 1 package in 500ms\n")
        test_container = _fake_container(0, b"Tests: 0 failed, 1 passed, 1 total\n")
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.side_effect = [prep_container, test_container]

        engine = VerificationEngine(network_mode="none")
        with patch.object(type(engine._docker_runner), "is_docker_available", True), \
             patch("app.services.verification.engine.subprocess.run") as mock_subproc_run:
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        mock_subproc_run.assert_not_called()
        # Both containers succeeded (dep prep, then a passing test run) --
        # the result must actually reflect that, not an unrelated failure.
        assert fake_docker_client.containers.run.call_count == 2
        assert result["success"] is True
        assert result["passed"] == 1
        assert result["failed"] == 0
