"""Python verification adapter.

Phase 7: executed through the same generic ``VerificationEngine._run_adapter``
path every other ecosystem already uses (no more Python-specific bypass in
``VerificationEngine.verify``), so a failed dependency install is classified
exactly like it is for Node/Go/etc -- ``available=False``, never a misleading
generic test failure. ``docker_image`` points at a pre-baked image carrying
``pytest`` (see ``docker/sandbox/python/Dockerfile``) since the stock
``python:3.11-slim`` ships no test framework and installing one at
verification-run time would need network the sandbox correctly denies. A
target repository's OWN third-party dependencies are handled separately, by
``VerificationEngine._install_python_deps_for_docker`` (host-side install,
mounted read-only) -- never by this adapter or this image.
"""

import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from app.services.sandbox.docker_runner import _pyproject_has_dependencies
from app.services.verification.base import VerificationAdapter


class PythonAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "python"
    manifest_files: ClassVar[List[str]] = ["pyproject.toml", "requirements.txt", "setup.py"]
    # Built ahead of time via `docker build -t repopilot-sandbox-python:3.11
    # docker/sandbox/python` -- see that Dockerfile for why pytest can't just
    # be installed at verification-run time.
    docker_image: ClassVar[str] = "repopilot-sandbox-python:3.11"

    @classmethod
    def detect(cls, workspace: Path) -> bool:
        if cls.find_manifests(workspace):
            return True
        # No packaging manifest, but pytest-discoverable test files exist
        # (e.g. a minimal script repo without pyproject.toml/requirements.txt).
        # This was RepoPilot's original unconditional default for any
        # workspace; kept here as the terminal fallback (Python is last in
        # ADAPTER_PRECEDENCE) so pre-existing manifest-less workflows keep
        # working exactly as before.
        return any(workspace.rglob("test_*.py")) or any(workspace.rglob("*_test.py"))

    def install_command(self, workspace: Path) -> Optional[List[str]]:
        """A pyproject.toml alone does not imply an install is needed -- see
        app.services.sandbox.docker_runner._pyproject_has_dependencies.
        setup.py-only projects keep the original, more conservative
        behavior (always attempt install) since their dependencies can't
        be determined without executing the file."""
        if (workspace / "requirements.txt").is_file():
            return ["pip", "install", "-r", "requirements.txt"]
        if (workspace / "setup.py").is_file():
            return ["pip", "install", "."]
        if (workspace / "pyproject.toml").is_file() and _pyproject_has_dependencies(workspace):
            return ["pip", "install", "."]
        return None

    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> List[str]:
        return ["pytest", "-v", "-o", "testpaths=.", test_path or "."]

    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        passed = 0
        failed = 0

        passed_match = re.search(r"(\d+)\s+passed", output)
        if passed_match:
            passed = int(passed_match.group(1))

        failed_match = re.search(r"(\d+)\s+failed", output)
        if failed_match:
            failed = int(failed_match.group(1))

        if passed == 0 and failed == 0:
            passed = len(re.findall(r"::\w+\s+PASSED", output))
            failed = len(re.findall(r"::\w+\s+FAILED", output))

        return {"passed": passed, "failed": failed}
