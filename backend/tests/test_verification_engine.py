"""Unit tests for VerificationEngine: ecosystem-aware execution wiring.

No real Docker or network calls are made -- Docker is exercised via a
MagicMock client (never invoked here since the CI/test environment has no
Docker daemon, so `is_docker_available` is naturally False) and subprocess
execution is exercised via `subprocess.run` patched with `unittest.mock`.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ReadTimeout

from app.core.config import settings
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


# ---------------------------------------------------------------------------
# verify_repository: task-aware, multi-project verification
# ---------------------------------------------------------------------------
def test_verify_repository_single_project_delegates_exactly_like_verify():
    """A single-ecosystem repository must behave byte-for-byte like plain
    verify() -- no wrapping, no behavior change for existing callers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        engine = VerificationEngine()
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
    the Docker path with the toolchain available in the (mocked) container.

    This is the regression test for the reported bug: previously every
    ecosystem ran in the single shared `python:3.11-slim` image, so a Node
    repo's `npm test` failed with "npm: not found" even though Docker itself
    was available and the correct command was selected. The fix is that the
    adapter's own image (node:20-slim) is what gets run, not the Python
    sandbox's image.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        engine = VerificationEngine()
        fake_docker_client = MagicMock()
        fake_container = _fake_container(
            0, b"PASS  src/App.test.js\nTests: 0 failed, 6 passed, 6 total\n"
        )
        fake_docker_client.containers.run.return_value = fake_container

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["success"] is True
        assert result["passed"] == 6
        assert result["available"] is True

        run_kwargs = fake_docker_client.containers.run.call_args.kwargs
        assert run_kwargs["image"] == "node:20-slim"
        assert run_kwargs["image"] != settings.docker_sandbox_image
        # detach=True is required so the engine can apply its own
        # wait(timeout=...) instead of the unbounded synchronous helper.
        assert run_kwargs["detach"] is True
        command = run_kwargs["command"]
        assert command[-2:] == ["npm", "test"]
        # the preflight check for npm's presence is embedded in the script
        # run before install/test, not skipped
        assert "command -v npm" in command[2]

        fake_container.wait.assert_called_once_with(timeout=engine.timeout)
        fake_container.remove.assert_called_once_with(force=True)


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
    bounds it; the container is killed and still reliably removed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        _write(root, "package-lock.json")

        engine = VerificationEngine(timeout=5)
        fake_container = MagicMock()
        fake_container.wait.side_effect = ReadTimeout("timed out")
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = fake_container

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        fake_container.wait.assert_called_once_with(timeout=5)
        fake_container.kill.assert_called_once()
        fake_container.remove.assert_called_once_with(force=True)

        assert result["exit_code"] == 124
        assert "timed out" in result["output"].lower()
        assert result["success"] is False
        # a bounded timeout is a controlled result, not a toolchain/install verdict
        assert result["available"] is True


# ---------------------------------------------------------------------------
# Docker path: dependency-install failure (e.g. no network under
# SANDBOX_NETWORK_MODE=none) must never be reported as a missing toolchain
# ---------------------------------------------------------------------------
def test_verify_docker_install_failure_reports_unable_to_verify_with_network_detail():
    """The exact Task #15 root cause: npm ci fails because the sandbox has
    no network access, so node_modules is never populated. Previously this
    surfaced as 'Required tool npm is not available' once npm test then hit
    its own missing devDependency. It must instead be reported as a
    dependency-installation failure, with npm never blamed, and the test
    command must never actually run against incomplete dependencies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "react-scripts test"}}))
        _write(root, "package-lock.json")

        engine = VerificationEngine(network_mode="none")
        fake_docker_client = MagicMock()
        fake_docker_client.containers.run.return_value = _fake_container(
            1,
            b"npm warn ERESOLVE overriding peer dependency\n"
            b"npm error Exit handler never called!\n"
            b"REPOPILOT_INSTALL_FAILED\n",
        )

        with patch.object(type(engine._docker_runner), "is_docker_available", True):
            engine._docker_runner._docker_client = fake_docker_client
            result = engine.verify(root)

        assert result["ecosystem"] == "node"
        assert result["available"] is False
        assert result["success"] is False
        assert "npm" not in result["detail"]
        assert "no network access" in result["detail"]

        # Structurally, the script must short-circuit before the test
        # command whenever install fails -- "react-scripts test" (via the
        # trailing "$@") is only ever reached if install succeeded first.
        script = fake_docker_client.containers.run.call_args.kwargs["command"][2]
        assert "npm ci" in script
        assert script.index("REPOPILOT_INSTALL_FAILED") < script.index('"$@"')


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
        assert result["success"] is False
