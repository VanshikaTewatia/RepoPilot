"""Java verification adapters: Maven (pom.xml) and Gradle (build.gradle[.kts]).

Kept as two separate adapters (rather than one "java" adapter with an
internal branch) so each build tool's manifest, command, and output format
are independently detectable, testable, and extensible.
"""

import re
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from app.services.verification.base import VerificationAdapter


class JavaMavenAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "java-maven"
    docker_image: ClassVar[str] = "maven:3.9-eclipse-temurin-21"
    manifest_files: ClassVar[List[str]] = ["pom.xml"]

    def install_command(self, workspace: Path) -> Optional[List[str]]:
        return None  # `mvn test` resolves its own dependencies.

    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> List[str]:
        # Prefer the repository-provided wrapper over an ambient Maven
        # install, mirroring JavaGradleAdapter's ./gradlew preference.
        executable = "./mvnw" if (workspace / "mvnw").is_file() else "mvn"
        cmd = [executable, "-B", "test"]
        if test_path:
            cmd.append(f"-Dtest={test_path}")
        return cmd

    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        total_run = total_failures = total_errors = 0
        for m in re.finditer(
            r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)",
            output,
        ):
            total_run += int(m.group(1))
            total_failures += int(m.group(2))
            total_errors += int(m.group(3))
        failed = total_failures + total_errors
        passed = max(total_run - failed, 0)
        return {"passed": passed, "failed": failed}


class JavaGradleAdapter(VerificationAdapter):
    ecosystem: ClassVar[str] = "java-gradle"
    docker_image: ClassVar[str] = "gradle:8-jdk21"
    manifest_files: ClassVar[List[str]] = ["build.gradle", "build.gradle.kts"]

    def install_command(self, workspace: Path) -> Optional[List[str]]:
        return None  # `gradle test` resolves its own dependencies.

    def test_command(self, workspace: Path, test_path: Optional[str] = None) -> List[str]:
        executable = "./gradlew" if (workspace / "gradlew").is_file() else "gradle"
        cmd = [executable, "test", "--console=plain"]
        if test_path:
            cmd.extend(["--tests", test_path])
        return cmd

    def parse_output(self, output: str, returncode: int) -> Dict[str, int]:
        m = re.search(r"(\d+)\s+tests?\s+completed,\s*(\d+)\s+failed", output, re.IGNORECASE)
        if m:
            total = int(m.group(1))
            failed = int(m.group(2))
            return {"passed": max(total - failed, 0), "failed": failed}
        if "BUILD SUCCESSFUL" in output:
            return {"passed": 1, "failed": 0}
        if "BUILD FAILED" in output or returncode != 0:
            return {"passed": 0, "failed": 1}
        return {"passed": 0, "failed": 0}
