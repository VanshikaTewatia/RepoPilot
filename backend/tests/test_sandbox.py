"""Unit tests for Docker and subprocess sandbox execution."""

import tempfile
from pathlib import Path
from app.services.sandbox.docker_runner import DockerTestRunner


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
