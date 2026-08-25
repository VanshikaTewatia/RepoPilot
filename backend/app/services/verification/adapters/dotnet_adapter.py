""".NET verification adapter (detected via *.sln / *.csproj)."""

import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from app.services.verification.base import VerificationAdapter


class DotnetAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "dotnet"
    # Documentation only -- these are extensions/globs, not literal filenames,
    # so find_manifests() is overridden below rather than relying on the
    # default literal-filename matching in VerificationAdapter.
    manifest_files: ClassVar[List[str]] = ["*.sln", "*.csproj"]

    @classmethod
    def find_manifests(cls, workspace: Path) -> List[str]:
        found: List[str] = sorted(p.name for p in workspace.glob("*.sln"))
        found += sorted(p.name for p in workspace.glob("*.csproj"))
        if not found:
            # .NET solutions commonly nest project files a level or two deep
            # (e.g. src/MyApp/MyApp.csproj); bounded so detection stays cheap.
            found = sorted(
                str(p.relative_to(workspace)) for p in list(workspace.rglob("*.csproj"))[:5]
            )
        return found

    def install_command(self, workspace: Path) -> Optional[List[str]]:
        return ["dotnet", "restore"]

    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> List[str]:
        cmd = ["dotnet", "test"]
        if test_path:
            cmd.extend(["--filter", test_path])
        return cmd

    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        # dotnet's console summary format varies by version/reporter, and the
        # order of "Passed:"/"Failed:" is not guaranteed (e.g. "Passed!  -
        # Failed: 0, Passed: 5, Skipped: 0, Total: 5"), so each is searched
        # independently instead of assuming a fixed order.
        passed_match = re.search(r"Passed:\s*(\d+)", output, re.IGNORECASE)
        failed_match = re.search(r"Failed:\s*(\d+)", output, re.IGNORECASE)
        if passed_match or failed_match:
            return {
                "passed": int(passed_match.group(1)) if passed_match else 0,
                "failed": int(failed_match.group(1)) if failed_match else 0,
            }
        return {"passed": 0, "failed": 0} if returncode == 0 else {"passed": 0, "failed": 1}
