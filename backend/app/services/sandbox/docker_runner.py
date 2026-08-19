"""Docker sandbox test runner with isolated ephemeral containers."""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.logging import logger

try:
    import docker
    from docker.errors import DockerException, APIError
    _DOCKER_LIB_AVAILABLE = True
except ImportError:
    _DOCKER_LIB_AVAILABLE = False

# Global cache to prevent repeated failed Docker socket attempts
_GLOBAL_DOCKER_DISABLED = False


class DockerTestRunner:
    """Executes pytest suites inside isolated Docker sandbox containers or secure local subprocess fallback."""

    def __init__(
        self,
        image: Optional[str] = None,
        timeout: Optional[int] = None,
        network_mode: Optional[str] = None,
    ):
        self.image = image or settings.docker_sandbox_image
        self.timeout = timeout or settings.sandbox_timeout_seconds
        self.network_mode = network_mode or settings.sandbox_network_mode
        self._docker_client: Optional[Any] = None
        self._docker_checked: bool = False
        self._docker_available: bool = False

    @property
    def is_docker_available(self) -> bool:
        """Check if Docker daemon is responsive with fast non-blocking probe."""
        global _GLOBAL_DOCKER_DISABLED
        if _GLOBAL_DOCKER_DISABLED or not _DOCKER_LIB_AVAILABLE:
            return False

        if self._docker_checked:
            return self._docker_available

        self._docker_checked = True
        try:
            self._docker_client = docker.from_env(timeout=1)
            self._docker_available = bool(self._docker_client.ping())
        except Exception:
            self._docker_client = None
            self._docker_available = False

        return self._docker_available

    def run_tests(
        self,
        workspace_path: Path | str,
        test_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute pytest in the given workspace directory."""
        workspace = Path(workspace_path).resolve()
        if not workspace.is_dir():
            return {
                "success": False,
                "exit_code": 1,
                "output": f"Workspace directory does not exist: {workspace}",
                "passed": 0,
                "failed": 1,
                "duration": 0.0,
            }

        start_time = time.time()

        if self.is_docker_available:
            try:
                return self._run_in_docker(workspace, test_path, start_time)
            except Exception as e:
                logger.info(f"Docker container run unavailable ({e}). Falling back to isolated subprocess.")
                return self._run_in_subprocess(workspace, test_path, start_time)
        else:
            return self._run_in_subprocess(workspace, test_path, start_time)

    def _run_in_docker(
        self,
        workspace: Path,
        test_path: Optional[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Run test command inside ephemeral Docker container."""
        global _GLOBAL_DOCKER_DISABLED
        cmd = ["pytest", "-v", "-o", "testpaths=."]
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append(".")

        try:
            volumes = {
                str(workspace): {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            }

            container = self._docker_client.containers.run(
                image=self.image,
                command=cmd,
                working_dir="/workspace",
                volumes=volumes,
                network_mode=self.network_mode,
                nano_cpus=int(settings.sandbox_max_cpu * 1e9),
                mem_limit=f"{settings.sandbox_max_memory_mb}m",
                detach=False,
                stdout=True,
                stderr=True,
                remove=True,
                user="1000:1000",
            )
            raw_output = container.decode("utf-8", errors="replace") if isinstance(container, bytes) else str(container)
            duration = time.time() - start_time
            parsed = self._parse_pytest_output(raw_output)

            success = (parsed["failed"] == 0 and parsed["passed"] > 0)
            return {
                "success": success,
                "exit_code": 0 if success else 1,
                "output": raw_output,
                "passed": parsed["passed"],
                "failed": parsed["failed"],
                "duration": round(duration, 2),
            }
        except (DockerException, APIError) as e:
            _GLOBAL_DOCKER_DISABLED = True
            self._docker_available = False
            return self._run_in_subprocess(workspace, test_path, start_time)

    def _run_in_subprocess(
        self,
        workspace: Path,
        test_path: Optional[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Local subprocess test execution fallback using current python environment."""
        cmd = [sys.executable, "-m", "pytest", "-v", "-o", "testpaths=."]
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append(".")

        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration = time.time() - start_time
            combined_output = f"{result.stdout}\n{result.stderr}".strip()
            parsed = self._parse_pytest_output(combined_output)

            success = (result.returncode == 0 and parsed["failed"] == 0 and parsed["passed"] > 0)
            return {
                "success": success,
                "exit_code": result.returncode,
                "output": combined_output,
                "passed": parsed["passed"],
                "failed": parsed["failed"],
                "duration": round(duration, 2),
            }
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "exit_code": 124,
                "output": f"Execution timed out after {self.timeout} seconds.",
                "passed": 0,
                "failed": 1,
                "duration": round(duration, 2),
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "exit_code": 1,
                "output": f"Execution error: {e}",
                "passed": 0,
                "failed": 1,
                "duration": round(duration, 2),
            }

    def _parse_pytest_output(self, output: str) -> Dict[str, int]:
        """Extract passed and failed count from pytest summary."""
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
