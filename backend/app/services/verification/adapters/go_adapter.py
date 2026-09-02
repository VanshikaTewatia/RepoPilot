"""Go verification adapter."""

import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from app.services.verification.base import VerificationAdapter


class GoAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "go"
    docker_image: ClassVar[str] = "golang:1.22-alpine"
    manifest_files: ClassVar[List[str]] = ["go.mod"]

    def install_command(self, workspace: Path) -> Optional[List[str]]:
        return ["go", "mod", "download"]

    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> List[str]:
        return ["go", "test", test_path or "./...", "-v"]

    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        passed = len(re.findall(r"^--- PASS:", output, re.MULTILINE))
        failed = len(re.findall(r"^--- FAIL:", output, re.MULTILINE))
        if passed == 0 and failed == 0:
            # go test without -v (or a build-only failure) prints per-package
            # ok/FAIL lines instead of per-test --- PASS/FAIL lines.
            passed = len(re.findall(r"^ok\s+\S+", output, re.MULTILINE))
            failed = len(re.findall(r"^FAIL\s+\S+", output, re.MULTILINE))
        return {"passed": passed, "failed": failed}
