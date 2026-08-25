"""Node / React / Next.js verification adapter.

Never assumes pytest. Picks the package manager from whichever lockfile is
present (pnpm-lock.yaml > yarn.lock > package-lock.json > npm default), runs
the project's own "test" script when package.json defines one, and falls
back to the "build" script otherwise -- a Next.js repo with no test suite is
still meaningfully verified by confirming it builds. If package.json defines
neither script, verification is reported unavailable rather than guessed at.
"""

import json
import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from app.services.verification.base import VerificationAdapter

# Order matters: first lockfile present wins.
_LOCKFILE_PACKAGE_MANAGERS = (
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("package-lock.json", "npm"),
)


class NodeAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "node"
    manifest_files: ClassVar[List[str]] = [
        "package.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "package-lock.json",
    ]

    def __init__(self) -> None:
        self._used_build_fallback = False

    @classmethod
    def detect(cls, workspace: Path) -> bool:
        # package.json is required to know what to run; a stray lockfile
        # without it isn't enough to call this a runnable Node project.
        return (workspace / "package.json").is_file()

    def package_manager(self, workspace: Path) -> str:
        for lockfile, manager in _LOCKFILE_PACKAGE_MANAGERS:
            if (workspace / lockfile).is_file():
                return manager
        return "npm"

    def _read_scripts(self, workspace: Path) -> Dict[str, str]:
        try:
            data = json.loads((workspace / "package.json").read_text(encoding="utf-8"))
        except Exception:
            return {}
        scripts = data.get("scripts")
        return scripts if isinstance(scripts, dict) else {}

    def install_command(self, workspace: Path) -> Optional[List[str]]:
        manager = self.package_manager(workspace)
        if manager == "npm":
            return ["npm", "ci"] if (workspace / "package-lock.json").is_file() else ["npm", "install"]
        return [manager, "install"]

    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> Optional[List[str]]:
        manager = self.package_manager(workspace)
        scripts = self._read_scripts(workspace)

        if scripts.get("test"):
            self._used_build_fallback = False
            return ["npm", "test"] if manager == "npm" else [manager, "test"]

        if scripts.get("build"):
            self._used_build_fallback = True
            return ["npm", "run", "build"] if manager == "npm" else [manager, "run", "build"]

        return None

    def unavailable_reason(self, workspace: Path) -> Optional[str]:
        return (
            "package.json has no \"test\" or \"build\" script defined; RepoPilot "
            "never assumes a pytest-style verification command for Node projects."
        )

    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        if self._used_build_fallback:
            # A build has no individual pass/fail counts -- its exit code is the verdict.
            return {"passed": 1, "failed": 0} if returncode == 0 else {"passed": 0, "failed": 1}

        failed_match = re.search(r"(\d+)\s+failed", output, re.IGNORECASE)
        passed_match = re.search(r"(\d+)\s+passed", output, re.IGNORECASE)
        failed = int(failed_match.group(1)) if failed_match else 0
        passed = int(passed_match.group(1)) if passed_match else 0

        if passed == 0 and failed == 0:
            # No recognizable summary line from the project's test reporter:
            # fall back to the process exit code rather than guessing counts.
            return {"passed": 1, "failed": 0} if returncode == 0 else {"passed": 0, "failed": 1}
        return {"passed": passed, "failed": failed}
