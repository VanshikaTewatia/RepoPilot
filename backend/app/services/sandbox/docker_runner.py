"""Docker sandbox test runner with isolated ephemeral containers."""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import logger

try:
    import docker
    from docker.errors import APIError, ContainerError, DockerException
    _DOCKER_LIB_AVAILABLE = True
except ImportError:
    _DOCKER_LIB_AVAILABLE = False

try:
    from requests.exceptions import ConnectionError as _RequestsConnectionError
    from requests.exceptions import ReadTimeout as _RequestsReadTimeout
    # A read-timeout on Container.wait(timeout=...) is documented as raising
    # requests.exceptions.ReadTimeout, but on Windows (Docker Desktop's named
    # -pipe transport) the same timeout instead surfaces as a ConnectionError
    # wrapping a urllib3 ReadTimeoutError -- see
    # VerificationEngine._execute_in_docker, which documents the identical,
    # empirically-confirmed quirk. Both must be treated as our own wait()
    # timeout, not a daemon-connectivity failure.
    _DOCKER_WAIT_TIMEOUT_EXCEPTIONS: Tuple[type, ...] = (
        _RequestsReadTimeout,
        _RequestsConnectionError,
    )
except ImportError:  # pragma: no cover - requests always installed alongside docker
    _DOCKER_WAIT_TIMEOUT_EXCEPTIONS = ()

# Global cache to prevent repeated failed Docker socket attempts
_GLOBAL_DOCKER_DISABLED = False

# Dependency installation is best-effort and non-fatal: if it fails (or the
# target repo has no network access, e.g. inside a network_mode="none"
# sandbox container) the install log is still surfaced in the test output,
# and pytest is attempted anyway.
_DEPENDENCY_INSTALL_TIMEOUT_SECONDS = 120


def _detect_dependency_install_args(workspace: Path) -> Optional[List[str]]:
    """Return pip install arguments for the repo's dependency manifest, if any.

    Covers the two common cases (a plain requirements.txt, or an installable
    package via pyproject.toml/setup.py). Repos using other package managers
    (poetry.lock-only, Pipenv, etc.) are not covered by this best-effort step.
    """
    if (workspace / "requirements.txt").is_file():
        return ["-r", "requirements.txt"]
    if (workspace / "pyproject.toml").is_file() or (workspace / "setup.py").is_file():
        return ["."]
    return None


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
        """Run test command inside ephemeral Docker container.

        If the repo has a recognized dependency manifest, it is installed
        inside the (ephemeral, single-use) container immediately before
        pytest runs. This only succeeds when the sandbox's network mode
        permits outbound access -- under the default network_mode="none" the
        install has no network and will fail, but pytest still runs
        afterwards so a real bug fix isn't masked by an unrelated install
        failure.

        Execution is bounded by ``self.timeout`` using ``Container.wait
        (timeout=...)`` (the same approach as
        ``VerificationEngine._execute_in_docker``): the blocking, single-call
        ``containers.run(detach=False, ...)`` convenience wrapper has no way
        to bound how long it blocks, so this uses the equivalent manual
        create/start/wait/logs/remove sequence instead. That wrapper also
        raises ``ContainerError`` whenever the container's command exits
        non-zero -- indistinguishable, to a bare ``except DockerException``,
        from Docker itself being broken. A failing test (the normal,
        expected outcome of most self-correction attempts before a real fix
        lands) must never be treated as "Docker is unavailable"; only a
        genuine daemon-level failure (unreachable daemon, image pull
        failure, ...) should disable Docker sandboxing.
        """
        global _GLOBAL_DOCKER_DISABLED
        test_target = test_path if test_path else "."
        install_args = _detect_dependency_install_args(workspace)

        if install_args:
            install_cmd = (
                "pip install --quiet --disable-pip-version-check --no-input "
                + " ".join(install_args)
            )
            # ';' (not '&&'): a failed/network-less install must not prevent
            # pytest from running -- its output stays visible either way.
            script = f'{install_cmd}; pytest -v -o testpaths=. "$@"'
            cmd = ["sh", "-c", script, "sh", test_target]
        else:
            cmd = ["pytest", "-v", "-o", "testpaths=.", test_target]

        volumes = {
            str(workspace): {
                "bind": "/workspace",
                "mode": "rw",
            }
        }

        container = None
        try:
            container = self._docker_client.containers.run(
                image=self.image,
                command=cmd,
                working_dir="/workspace",
                volumes=volumes,
                network_mode=self.network_mode,
                nano_cpus=int(settings.sandbox_max_cpu * 1e9),
                mem_limit=f"{settings.sandbox_max_memory_mb}m",
                detach=True,
                user="1000:1000",
            )

            try:
                wait_result = container.wait(timeout=self.timeout)
            except _DOCKER_WAIT_TIMEOUT_EXCEPTIONS:
                logger.warning(
                    f"Docker sandbox run exceeded {self.timeout}s timeout; killing container."
                )
                try:
                    container.kill()
                except (DockerException, APIError):
                    pass
                duration = time.time() - start_time
                return {
                    "success": False,
                    "exit_code": 124,
                    "output": f"Execution timed out after {self.timeout} seconds.",
                    "passed": 0,
                    "failed": 1,
                    "duration": round(duration, 2),
                }

            exit_code = wait_result.get("StatusCode", 1)
            raw = container.logs(stdout=True, stderr=True)
            raw_output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            duration = time.time() - start_time
            parsed = self._parse_pytest_output(raw_output)

            success = (exit_code == 0 and parsed["failed"] == 0 and parsed["passed"] > 0)
            return {
                "success": success,
                "exit_code": exit_code,
                "output": raw_output,
                "passed": parsed["passed"],
                "failed": parsed["failed"],
                "duration": round(duration, 2),
            }
        except ContainerError as e:
            # Docker successfully ran the container and the command exited
            # non-zero -- a normal failed-test result (e.g. real pytest
            # failures), never a signal that Docker itself is broken.
            # _GLOBAL_DOCKER_DISABLED must never be set for this branch.
            duration = time.time() - start_time
            raw_stderr = e.stderr
            raw_output = (
                raw_stderr.decode("utf-8", errors="replace")
                if isinstance(raw_stderr, bytes)
                else str(raw_stderr or e)
            )
            parsed = self._parse_pytest_output(raw_output)
            return {
                "success": False,
                "exit_code": e.exit_status if e.exit_status is not None else 1,
                "output": raw_output,
                "passed": parsed["passed"],
                "failed": parsed["failed"],
                "duration": round(duration, 2),
            }
        except (DockerException, APIError) as e:
            # A genuine daemon-level failure (unreachable daemon, image not
            # pulled/built, connection dropped, ...) -- disable Docker
            # sandboxing and fall back to the subprocess runner for this and
            # subsequent calls.
            _GLOBAL_DOCKER_DISABLED = True
            self._docker_available = False
            return self._run_in_subprocess(workspace, test_path, start_time)
        finally:
            # Explicit removal (rather than run()'s remove=True/auto_remove)
            # since detach=True is required to apply our own wait() timeout.
            if container is not None:
                try:
                    container.remove(force=True)
                except (DockerException, APIError):
                    pass

    def _install_dependencies(self, workspace: Path) -> tuple[str, Optional[str]]:
        """Best-effort install of the repo's dependency manifest for the subprocess fallback.

        Installs into an ephemeral, isolated `--target` directory -- never
        into the backend's own environment -- so a target repo's
        dependencies can never mutate or break RepoPilot's own packages.
        Returns (log_text, pythonpath_dir). pythonpath_dir is None if there
        was nothing to install or the target directory could not be created.
        """
        install_args = _detect_dependency_install_args(workspace)
        if not install_args:
            return "", None

        target_dir = tempfile.mkdtemp(prefix="repopilot_deps_")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--quiet", "--disable-pip-version-check", "--no-input",
            "--target", target_dir,
        ] + install_args
        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=_DEPENDENCY_INSTALL_TIMEOUT_SECONDS,
            )
            log = f"$ pip install {' '.join(install_args)}\n{result.stdout}\n{result.stderr}".strip()
            return log, target_dir
        except subprocess.TimeoutExpired:
            return f"Dependency installation timed out after {_DEPENDENCY_INSTALL_TIMEOUT_SECONDS}s.", target_dir
        except Exception as e:
            return f"Dependency installation error: {e}", target_dir

    def _run_in_subprocess(
        self,
        workspace: Path,
        test_path: Optional[str],
        start_time: float,
    ) -> Dict[str, Any]:
        """Local subprocess test execution fallback using the current python interpreter.

        If the repo has a recognized dependency manifest, it is installed
        first into an isolated, ephemeral directory (see _install_dependencies)
        so pytest can import the repo's real dependencies without RepoPilot's
        own environment ever being modified. A failed install does not block
        the test run -- its log is prepended to the output either way.
        """
        install_log, pythonpath_dir = self._install_dependencies(workspace)

        cmd = [sys.executable, "-m", "pytest", "-v", "-o", "testpaths=."]
        if test_path:
            cmd.append(test_path)
        else:
            cmd.append(".")

        env = os.environ.copy()
        if pythonpath_dir:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{pythonpath_dir}{os.pathsep}{existing}" if existing else pythonpath_dir

        try:
            result = subprocess.run(
                cmd,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            duration = time.time() - start_time
            combined_output = f"{install_log}\n\n{result.stdout}\n{result.stderr}".strip()
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
                "output": f"{install_log}\n\nExecution timed out after {self.timeout} seconds.".strip(),
                "passed": 0,
                "failed": 1,
                "duration": round(duration, 2),
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "exit_code": 1,
                "output": f"{install_log}\n\nExecution error: {e}".strip(),
                "passed": 0,
                "failed": 1,
                "duration": round(duration, 2),
            }
        finally:
            if pythonpath_dir:
                shutil.rmtree(pythonpath_dir, ignore_errors=True)

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
