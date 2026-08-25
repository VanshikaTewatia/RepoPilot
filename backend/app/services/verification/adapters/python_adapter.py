"""Python verification adapter.

Command construction here mirrors ``app.services.sandbox.docker_runner``
exactly (``pip install -r requirements.txt`` / ``pip install .`` then
``pytest``), purely so command *selection* is documented and unit-testable
like every other ecosystem. Actual execution for a detected Python project is
delegated wholesale to the existing, already-hardened ``DockerTestRunner``
(see ``VerificationEngine.verify``) so demo_repo and all pre-existing sandbox
behavior are preserved byte-for-byte.
"""

import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from app.services.verification.base import VerificationAdapter


class PythonAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "python"
    manifest_files: ClassVar[List[str]] = ["pyproject.toml", "requirements.txt", "setup.py"]

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
        if (workspace / "requirements.txt").is_file():
            return ["pip", "install", "-r", "requirements.txt"]
        if (workspace / "pyproject.toml").is_file() or (workspace / "setup.py").is_file():
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
