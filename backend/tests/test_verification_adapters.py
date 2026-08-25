"""Unit tests for verification adapter command selection and output parsing."""

import json
import tempfile
from pathlib import Path

import pytest

from app.services.verification.adapters import (
    DotnetAdapter,
    GoAdapter,
    JavaGradleAdapter,
    JavaMavenAdapter,
    NodeAdapter,
    PythonAdapter,
    RustAdapter,
)


def _write(root: Path, rel_path: str, content: str = "") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
class TestPythonAdapter:
    def test_install_command_prefers_requirements_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "requirements.txt", "flask\n")
            _write(root, "pyproject.toml", "[project]\nname='x'\n")

            assert PythonAdapter().install_command(root) == ["pip", "install", "-r", "requirements.txt"]

    def test_install_command_falls_back_to_pyproject(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pyproject.toml", "[project]\nname='x'\n")

            assert PythonAdapter().install_command(root) == ["pip", "install", "."]

    def test_install_command_none_when_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert PythonAdapter().install_command(Path(tmpdir)) is None

    def test_test_command_defaults_to_whole_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = PythonAdapter().test_command(Path(tmpdir))
            assert cmd == ["pytest", "-v", "-o", "testpaths=.", "."]

    def test_test_command_honors_test_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = PythonAdapter().test_command(Path(tmpdir), test_path="tests/test_x.py::test_y")
            assert cmd[-1] == "tests/test_x.py::test_y"

    def test_parse_output_extracts_passed_and_failed(self):
        counts = PythonAdapter().parse_output("===== 5 passed, 2 failed in 0.4s =====", 1)
        assert counts == {"passed": 5, "failed": 2}


# ---------------------------------------------------------------------------
# Node / React / Next.js
# ---------------------------------------------------------------------------
class TestNodeAdapter:
    def test_package_manager_prefers_pnpm_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pnpm-lock.yaml")
            _write(root, "yarn.lock")
            assert NodeAdapter().package_manager(root) == "pnpm"

    def test_package_manager_prefers_yarn_over_npm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "yarn.lock")
            _write(root, "package-lock.json")
            assert NodeAdapter().package_manager(root) == "yarn"

    def test_package_manager_uses_npm_with_package_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "package-lock.json")
            assert NodeAdapter().package_manager(root) == "npm"

    def test_package_manager_defaults_to_npm_with_no_lockfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert NodeAdapter().package_manager(Path(tmpdir)) == "npm"

    def test_install_command_uses_npm_ci_when_package_lock_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "package-lock.json")
            assert NodeAdapter().install_command(root) == ["npm", "ci"]

    def test_install_command_uses_npm_install_with_no_lockfile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert NodeAdapter().install_command(Path(tmpdir)) == ["npm", "install"]

    def test_install_command_uses_pnpm_install(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pnpm-lock.yaml")
            assert NodeAdapter().install_command(root) == ["pnpm", "install"]

    def test_test_command_uses_test_script_when_defined(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "package.json", json.dumps({"scripts": {"test": "jest", "build": "next build"}}))

            adapter = NodeAdapter()
            cmd = adapter.test_command(root)

            assert cmd == ["npm", "test"]
            assert adapter._used_build_fallback is False

    def test_test_command_falls_back_to_build_script(self):
        """A Next.js repo with no test suite is verified via its build script,
        never by assuming pytest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "package.json", json.dumps({"scripts": {"build": "next build"}}))

            adapter = NodeAdapter()
            cmd = adapter.test_command(root)

            assert cmd == ["npm", "run", "build"]
            assert adapter._used_build_fallback is True

    def test_test_command_uses_pnpm_when_pnpm_lock_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pnpm-lock.yaml")
            _write(root, "package.json", json.dumps({"scripts": {"test": "vitest run"}}))

            cmd = NodeAdapter().test_command(root)

            assert cmd == ["pnpm", "test"]

    def test_test_command_none_when_neither_test_nor_build_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "package.json", json.dumps({"scripts": {"start": "node index.js"}}))

            adapter = NodeAdapter()
            assert adapter.test_command(root) is None
            assert "test" in adapter.unavailable_reason(root).lower() or "build" in adapter.unavailable_reason(root).lower()

    def test_test_command_none_when_no_scripts_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "package.json", json.dumps({"name": "app"}))
            assert NodeAdapter().test_command(root) is None

    def test_test_command_none_when_package_json_is_malformed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "package.json", "{not valid json")
            assert NodeAdapter().test_command(root) is None

    def test_parse_output_extracts_jest_style_summary(self):
        adapter = NodeAdapter()
        adapter.test_command(_tmp_with_package_json({"scripts": {"test": "jest"}}))
        counts = adapter.parse_output("Tests:       2 failed, 3 passed, 5 total", 1)
        assert counts == {"passed": 3, "failed": 2}

    def test_parse_output_falls_back_to_exit_code_when_no_summary_found(self):
        adapter = NodeAdapter()
        adapter.test_command(_tmp_with_package_json({"scripts": {"test": "some-custom-runner"}}))
        assert adapter.parse_output("no recognizable summary here", 0) == {"passed": 1, "failed": 0}
        assert adapter.parse_output("no recognizable summary here", 1) == {"passed": 0, "failed": 1}

    def test_parse_output_uses_exit_code_only_for_build_fallback(self):
        adapter = NodeAdapter()
        adapter.test_command(_tmp_with_package_json({"scripts": {"build": "next build"}}))
        assert adapter.parse_output("Compiled successfully, 10 passed some unrelated log", 0) == {
            "passed": 1,
            "failed": 0,
        }
        assert adapter.parse_output("Build error", 1) == {"passed": 0, "failed": 1}


def _tmp_with_package_json(package_json: dict) -> Path:
    tmpdir = tempfile.mkdtemp()
    _write(Path(tmpdir), "package.json", json.dumps(package_json))
    return Path(tmpdir)


# ---------------------------------------------------------------------------
# Java: Maven
# ---------------------------------------------------------------------------
class TestJavaMavenAdapter:
    def test_install_command_is_none(self):
        assert JavaMavenAdapter().install_command(Path(".")) is None

    def test_test_command_default(self):
        assert JavaMavenAdapter().test_command(Path(".")) == ["mvn", "-B", "test"]

    def test_test_command_with_test_path(self):
        cmd = JavaMavenAdapter().test_command(Path("."), test_path="com.example.FooTest")
        assert cmd == ["mvn", "-B", "test", "-Dtest=com.example.FooTest"]

    def test_parse_output_single_module_summary(self):
        output = "Tests run: 10, Failures: 1, Errors: 2, Skipped: 0"
        counts = JavaMavenAdapter().parse_output(output, 1)
        assert counts == {"passed": 7, "failed": 3}

    def test_parse_output_no_summary_line(self):
        assert JavaMavenAdapter().parse_output("BUILD SUCCESS", 0) == {"passed": 0, "failed": 0}


# ---------------------------------------------------------------------------
# Java: Gradle
# ---------------------------------------------------------------------------
class TestJavaGradleAdapter:
    def test_test_command_uses_wrapper_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "gradlew", "#!/bin/sh\n")
            cmd = JavaGradleAdapter().test_command(root)
            assert cmd[0] == "./gradlew"

    def test_test_command_uses_bare_gradle_without_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = JavaGradleAdapter().test_command(Path(tmpdir))
            assert cmd[0] == "gradle"

    def test_test_command_with_test_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = JavaGradleAdapter().test_command(Path(tmpdir), test_path="com.example.FooTest")
            assert "--tests" in cmd and "com.example.FooTest" in cmd

    def test_parse_output_build_successful(self):
        assert JavaGradleAdapter().parse_output("BUILD SUCCESSFUL in 3s", 0) == {"passed": 1, "failed": 0}

    def test_parse_output_build_failed(self):
        assert JavaGradleAdapter().parse_output("BUILD FAILED in 3s", 1) == {"passed": 0, "failed": 1}


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------
class TestGoAdapter:
    def test_test_command_default(self):
        assert GoAdapter().test_command(Path(".")) == ["go", "test", "./...", "-v"]

    def test_test_command_with_test_path(self):
        cmd = GoAdapter().test_command(Path("."), test_path="./pkg/foo/...")
        assert cmd == ["go", "test", "./pkg/foo/...", "-v"]

    def test_install_command_is_go_mod_download(self):
        assert GoAdapter().install_command(Path(".")) == ["go", "mod", "download"]

    def test_parse_output_verbose_pass_fail_lines(self):
        output = "--- PASS: TestA (0.00s)\n--- PASS: TestB (0.00s)\n--- FAIL: TestC (0.00s)\nFAIL"
        assert GoAdapter().parse_output(output, 1) == {"passed": 2, "failed": 1}

    def test_parse_output_package_level_ok_fail(self):
        output = "ok  \texample.com/pkg1\t0.002s\nFAIL\texample.com/pkg2\t0.003s"
        assert GoAdapter().parse_output(output, 1) == {"passed": 1, "failed": 1}


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------
class TestRustAdapter:
    def test_test_command_default(self):
        assert RustAdapter().test_command(Path(".")) == ["cargo", "test"]

    def test_test_command_with_test_path(self):
        assert RustAdapter().test_command(Path("."), test_path="my_test") == ["cargo", "test", "my_test"]

    def test_parse_output_summary_line(self):
        output = "running 3 tests\ntest result: FAILED. 2 passed; 1 failed; 0 ignored"
        assert RustAdapter().parse_output(output, 1) == {"passed": 2, "failed": 1}

    def test_parse_output_all_passed(self):
        output = "test result: ok. 4 passed; 0 failed; 0 ignored"
        assert RustAdapter().parse_output(output, 0) == {"passed": 4, "failed": 0}


# ---------------------------------------------------------------------------
# .NET
# ---------------------------------------------------------------------------
class TestDotnetAdapter:
    def test_install_command(self):
        assert DotnetAdapter().install_command(Path(".")) == ["dotnet", "restore"]

    def test_test_command_default(self):
        assert DotnetAdapter().test_command(Path(".")) == ["dotnet", "test"]

    def test_test_command_with_filter(self):
        cmd = DotnetAdapter().test_command(Path("."), test_path="FullyQualifiedName~FooTests")
        assert cmd == ["dotnet", "test", "--filter", "FullyQualifiedName~FooTests"]

    def test_parse_output_failed_before_passed_in_text(self):
        """dotnet's own summary often prints Failed: before Passed: -- order must not matter."""
        output = "Passed!  - Failed:     0, Passed:     5, Skipped:     0, Total:     5"
        assert DotnetAdapter().parse_output(output, 0) == {"passed": 5, "failed": 0}

    def test_parse_output_legacy_format(self):
        output = "Total tests: 5. Passed: 4. Failed: 1."
        assert DotnetAdapter().parse_output(output, 1) == {"passed": 4, "failed": 1}

    def test_parse_output_no_summary_uses_exit_code(self):
        assert DotnetAdapter().parse_output("unexpected crash", 1) == {"passed": 0, "failed": 1}
        assert DotnetAdapter().parse_output("", 0) == {"passed": 0, "failed": 0}
