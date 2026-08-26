"""Generic, adapter-driven project verification engine.

Chooses the correct verification strategy for a workspace based on
``ProjectDetector`` and executes it with the same isolation guarantees as the
existing sandbox (Docker when available, a secure subprocess fallback
otherwise: commands always run with the workspace directory as their cwd/
mount, the same resource limits, network mode, and timeout as configured for
the sandbox). A detected Python project is delegated wholesale to the
existing, already-hardened ``DockerTestRunner`` so demo_repo and all
pre-existing sandbox behavior are preserved byte-for-byte; every other
ecosystem is executed generically here using its adapter's command argv.
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.services.sandbox.docker_runner import DockerTestRunner
from app.services.verification.base import VerificationAdapter, VerificationResult
from app.services.verification.detector import ProjectDetector
from app.services.verification.project_analyzer import RepositoryAnalyzer, select_relevant_projects

try:
    from docker.errors import APIError, ContainerError, DockerException
except ImportError:  # pragma: no cover - docker SDK not installed
    class DockerException(Exception):
        pass

    class APIError(DockerException):
        pass

    class ContainerError(DockerException):
        exit_status = 1
        stderr = b""


_INSTALL_TIMEOUT_SECONDS = 120


def _looks_like_missing_tool(output: str, exit_code: int) -> bool:
    """True when a verification command failed because its toolchain isn't
    installed in this environment (e.g. "gradle: command not found"), as
    opposed to the command running and finding a real test failure.

    This distinction is load-bearing: a missing toolchain must be reported
    as UNABLE_TO_VERIFY, never conflated with "the bug doesn't exist" or
    "the fix failed". 127 is the POSIX convention for "command not found"
    on both the sh-based Docker path and the subprocess fallback (which
    also maps a Python FileNotFoundError to exit code 127 explicitly).
    """
    if exit_code == 127:
        return True
    lowered = output.lower()
    return "required toolchain not found" in lowered


class VerificationEngine:
    """Detects a repository's ecosystem and runs its verification adapter."""

    def __init__(
        self,
        timeout: Optional[int] = None,
        network_mode: Optional[str] = None,
    ):
        self.timeout = timeout or settings.sandbox_timeout_seconds
        self.network_mode = network_mode or settings.sandbox_network_mode
        # Reused purely for its Docker-availability probe, client, and image
        # settings; Python verification also delegates its execution to it.
        self._docker_runner = DockerTestRunner(timeout=self.timeout, network_mode=self.network_mode)

    def verify(self, workspace_path: Path | str, test_path: Optional[str] = None) -> Dict[str, Any]:
        """Detect the workspace's ecosystem and run its verification command."""
        workspace = Path(workspace_path).resolve()
        if not workspace.is_dir():
            return VerificationResult(
                ecosystem="unknown",
                success=False,
                exit_code=1,
                output=f"Workspace directory does not exist: {workspace}",
                passed=0,
                failed=1,
                duration=0.0,
                available=False,
            ).to_dict()

        detection = ProjectDetector.detect(workspace)

        if detection.adapter is None:
            detail = (
                "Could not detect a supported project ecosystem in this repository. "
                f"Scanned for: {', '.join(detection.manifests_scanned)}. "
                "None of these manifest files were found, so verification was not "
                "attempted -- no pass/fail result is assumed."
            )
            logger.warning(f"Verification unavailable for workspace {workspace}: {detail}")
            return VerificationResult(
                ecosystem="unknown",
                success=False,
                exit_code=1,
                output=detail,
                passed=0,
                failed=0,
                duration=0.0,
                available=False,
                manifests_found=[],
                detail=detail,
            ).to_dict()

        adapter = detection.adapter

        if adapter.ecosystem == "python":
            # Preserve the exact, already-hardened Python/pytest execution path.
            result = self._docker_runner.run_tests(workspace_path=workspace, test_path=test_path)
            result.setdefault("ecosystem", "python")
            result.setdefault("available", True)
            result.setdefault("manifests_found", detection.manifests_found)
            result.setdefault("detail", None)
            result.setdefault("command", None)
            return result

        return self._run_adapter(adapter, workspace, test_path, detection.manifests_found)

    # -------------------------------------------------------------------
    # Repository-wide, task-aware, multi-project verification
    # -------------------------------------------------------------------
    def verify_repository(
        self,
        workspace_path: Path | str,
        task_description: str = "",
        keyword_matches: Optional[List[Dict[str, Any]]] = None,
        test_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Task-aware, multi-project-capable verification entry point.

        Detects every real project in the repository (see
        ``RepositoryAnalyzer``), narrows to the one(s) actually relevant to
        this task using repository evidence -- detected languages/frameworks
        plus where investigation keyword matches physically live -- and runs
        ``verify()`` scoped to each selected project's own root.

        Whenever the repository resolves to a single project, this delegates
        to ``verify()`` with that project's root exactly as before (a
        single-ecosystem repo rooted at "." behaves byte-for-byte like plain
        ``verify()``), so demo_repo and all existing single-project behavior
        is unaffected.
        """
        workspace = Path(workspace_path).resolve()
        if not workspace.is_dir():
            return self.verify(workspace_path=workspace_path, test_path=test_path)

        projects = RepositoryAnalyzer.analyze(workspace)
        if not projects:
            # No recognizable ecosystem anywhere -- verify() already reports
            # this as an unavailable "unknown" ecosystem without guessing.
            return self.verify(workspace_path=workspace, test_path=test_path)

        selected = (
            projects
            if len(projects) == 1
            else select_relevant_projects(projects, task_description=task_description, keyword_matches=keyword_matches or [])
        )

        if len(selected) == 1:
            proj = selected[0]
            proj_path = workspace if proj.root == "." else (workspace / proj.root)
            result = self.verify(workspace_path=proj_path, test_path=test_path)
            result["project_root"] = proj.root
            result["detected_projects"] = [p.to_dict() for p in projects]
            return result

        results: List[Dict[str, Any]] = []
        for proj in selected:
            proj_path = workspace if proj.root == "." else (workspace / proj.root)
            r = self.verify(workspace_path=proj_path, test_path=test_path)
            r["project_root"] = proj.root
            results.append(r)

        overall_available = all(r.get("available", True) for r in results)
        overall_success = overall_available and all(r.get("success", False) for r in results)
        unavailable_details = [
            f"[{r.get('project_root')}] {r.get('detail')}"
            for r in results
            if not r.get("available", True) and r.get("detail")
        ]

        return {
            "success": overall_success,
            "exit_code": 0 if overall_success else 1,
            "output": "\n\n".join(f"[{r.get('project_root', '.')}] {r.get('output', '')}" for r in results),
            "passed": sum(r.get("passed", 0) for r in results),
            "failed": sum(r.get("failed", 0) for r in results),
            "duration": round(sum(r.get("duration", 0.0) for r in results), 2),
            "ecosystem": ",".join(sorted({r.get("ecosystem", "unknown") for r in results})),
            "available": overall_available,
            "manifests_found": sorted({m for r in results for m in r.get("manifests_found", [])}),
            "detail": "; ".join(unavailable_details) or None,
            "command": " && ".join(r.get("command") or "" for r in results if r.get("command")),
            "project_results": results,
            "detected_projects": [p.to_dict() for p in projects],
        }

    # -------------------------------------------------------------------
    # Generic (non-Python) adapter execution
    # -------------------------------------------------------------------
    def _run_adapter(
        self,
        adapter: VerificationAdapter,
        workspace: Path,
        test_path: Optional[str],
        manifests_found: List[str],
    ) -> Dict[str, Any]:
        start_time = time.time()
        test_argv = adapter.test_command(workspace, test_path)

        if test_argv is None:
            duration = time.time() - start_time
            detail = adapter.unavailable_reason(workspace) or (
                f"No runnable verification command could be determined for "
                f"ecosystem '{adapter.ecosystem}'."
            )
            logger.warning(f"Verification unavailable for {adapter.ecosystem} at {workspace}: {detail}")
            return VerificationResult(
                ecosystem=adapter.ecosystem,
                success=False,
                exit_code=1,
                output=detail,
                passed=0,
                failed=0,
                duration=round(duration, 2),
                available=False,
                manifests_found=manifests_found,
                detail=detail,
            ).to_dict()

        install_argv = adapter.install_command(workspace)

        if self._docker_runner.is_docker_available:
            output, exit_code = self._execute_in_docker(workspace, install_argv, test_argv)
        else:
            output, exit_code = self._execute_in_subprocess(workspace, install_argv, test_argv)

        duration = time.time() - start_time

        if _looks_like_missing_tool(output, exit_code):
            detail = (
                f"Required tool '{test_argv[0]}' for {adapter.ecosystem} verification is not "
                "available in this environment, so the claimed behavior could not be verified. "
                "This is not evidence that the reported issue does or does not exist."
            )
            logger.warning(f"Verification tool missing for {adapter.ecosystem} at {workspace}: {detail}")
            return VerificationResult(
                ecosystem=adapter.ecosystem,
                success=False,
                exit_code=exit_code,
                output=output,
                passed=0,
                failed=0,
                duration=round(duration, 2),
                available=False,
                manifests_found=manifests_found,
                detail=detail,
                command=" ".join(test_argv),
            ).to_dict()

        counts = adapter.parse_output(output, exit_code)
        success = exit_code == 0 and counts.get("failed", 0) == 0

        return VerificationResult(
            ecosystem=adapter.ecosystem,
            success=success,
            exit_code=exit_code,
            output=output,
            passed=counts.get("passed", 0),
            failed=counts.get("failed", 0),
            duration=round(duration, 2),
            available=True,
            manifests_found=manifests_found,
            command=" ".join(test_argv),
        ).to_dict()

    def _execute_in_docker(
        self, workspace: Path, install_argv: Optional[List[str]], test_argv: List[str]
    ) -> Tuple[str, int]:
        """Run install (best-effort) + test commands in an ephemeral Docker container."""
        script_parts = []
        if install_argv:
            # '|| true': a failed/network-less install must not prevent the
            # test command from running (same non-fatal semantics as Python).
            script_parts.append(" ".join(install_argv) + " || true")
        script_parts.append('"$@"')
        script = "; ".join(script_parts)
        cmd = ["sh", "-c", script, "sh"] + test_argv

        try:
            raw = self._docker_runner._docker_client.containers.run(
                image=self._docker_runner.image,
                command=cmd,
                working_dir="/workspace",
                volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
                network_mode=self.network_mode,
                nano_cpus=int(settings.sandbox_max_cpu * 1e9),
                mem_limit=f"{settings.sandbox_max_memory_mb}m",
                detach=False,
                stdout=True,
                stderr=True,
                remove=True,
                user="1000:1000",
            )
            output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            return output, 0
        except ContainerError as e:
            stderr = e.stderr
            if isinstance(stderr, (bytes, bytearray)):
                stderr = stderr.decode("utf-8", errors="replace")
            return (stderr or str(e)), (e.exit_status or 1)
        except (DockerException, APIError) as e:
            logger.info(f"Docker verification run unavailable ({e}); falling back to subprocess.")
            return self._execute_in_subprocess(workspace, install_argv, test_argv)

    def _execute_in_subprocess(
        self, workspace: Path, install_argv: Optional[List[str]], test_argv: List[str]
    ) -> Tuple[str, int]:
        """Run install (best-effort) + test commands as local subprocesses, cwd=workspace."""
        install_log = ""
        if install_argv:
            try:
                install_result = subprocess.run(
                    install_argv,
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=min(self.timeout, _INSTALL_TIMEOUT_SECONDS),
                )
                install_log = (
                    f"$ {' '.join(install_argv)}\n{install_result.stdout}\n{install_result.stderr}"
                ).strip()
            except FileNotFoundError as e:
                install_log = f"Dependency installation error: toolchain not found ({e})"
            except subprocess.TimeoutExpired:
                install_log = "Dependency installation timed out."
            except Exception as e:
                install_log = f"Dependency installation error: {e}"

        try:
            result = subprocess.run(
                test_argv,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            combined = f"{install_log}\n\n{result.stdout}\n{result.stderr}".strip()
            return combined, result.returncode
        except subprocess.TimeoutExpired:
            return (
                f"{install_log}\n\nExecution timed out after {self.timeout} seconds.".strip(),
                124,
            )
        except FileNotFoundError as e:
            return (
                f"{install_log}\n\nRequired toolchain not found: {e}".strip(),
                127,
            )
        except Exception as e:
            return (f"{install_log}\n\nExecution error: {e}".strip(), 1)
