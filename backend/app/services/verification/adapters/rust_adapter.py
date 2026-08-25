"""Rust verification adapter."""

import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from app.services.verification.base import VerificationAdapter


class RustAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "rust"
    manifest_files: ClassVar[List[str]] = ["Cargo.toml"]

    def install_command(self, workspace: Path) -> Optional[List[str]]:
        return ["cargo", "fetch"]

    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> List[str]:
        cmd = ["cargo", "test"]
        if test_path:
            cmd.append(test_path)
        return cmd

    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        passed = failed = 0
        for m in re.finditer(
            r"test result:\s*\w+\.\s*(\d+)\s+passed;\s*(\d+)\s+failed", output
        ):
            passed += int(m.group(1))
            failed += int(m.group(2))
        return {"passed": passed, "failed": failed}
