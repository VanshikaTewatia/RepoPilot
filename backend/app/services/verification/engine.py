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

import os
import shlex
import shutil
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
    from docker.errors import APIError, DockerException
    from requests.exceptions import ConnectionError as _RequestsConnectionError
    from requests.exceptions import ReadTimeout as _RequestsReadTimeout
    # A read-timeout on Container.wait(timeout=...) is documented as raising
    # requests.exceptions.ReadTimeout, but on Windows (Docker Desktop's named
    # -pipe transport, docker.transport.npipesocket) the same timeout instead
    # surfaces as a ConnectionError wrapping a urllib3 ReadTimeoutError --
    # confirmed empirically against a real Docker Desktop instance. Both must
    # be treated as our own wait() timeout, not a daemon-connectivity failure.
    _DOCKER_WAIT_TIMEOUT_EXCEPTIONS: Tuple[type, ...] = (
        _RequestsReadTimeout,
        _RequestsConnectionError,
    )
except ImportError:  # pragma: no cover - docker SDK not installed
    class DockerException(Exception):
        pass

    class APIError(DockerException):
        pass

    _DOCKER_WAIT_TIMEOUT_EXCEPTIONS = (Exception,)


_INSTALL_TIMEOUT_SECONDS = 120


# Bare exit code 127 is NOT a reliable signal that a toolchain is missing:
# a project's own test command can just as easily exit 127 when ITS
# dependency is missing (e.g. `npm test` -> `sh: react-scripts: not found`
# because `npm ci` never installed it) even though `npm` itself is present
# and working fine. Conflating the two produced a real bug (Task #15):
# "Required tool 'npm' is not available" when npm was never the problem.
#
# So only OUR OWN preflight check (see _preflight_snippet) is trusted to
# report a missing toolchain, and it does so via this unambiguous sentinel
# rather than exit code alone -- a project's own script can never
# accidentally produce this exact string.
_TOOLCHAIN_MISSING_SENTINEL = "REPOPILOT_TOOLCHAIN_MISSING:"

# Emitted when the install step itself fails (e.g. `npm ci` with no network
# under SANDBOX_NETWORK_MODE=none). Distinct from a missing toolchain: the
# tool ran fine, it just couldn't fetch the project's dependencies.
_INSTALL_FAILED_SENTINEL = "REPOPILOT_INSTALL_FAILED"

# npm has a documented, reproducible bug ("Exit handler never called!",
# https://github.com/npm/cli/issues) where `npm ci` can exit 0 even though
# it never finished installing -- under total network denial it can get
# most of the way through fetching packages and then abort without linking
# their node_modules/.bin executables, so a project's own test runner (e.g.
# react-scripts) ends up genuinely missing despite a "successful" install
# exit code. This is npm self-reporting its own internal failure, not a
# downstream project's error -- checked in addition to the exit code
# (below) because the exit code alone is provably unreliable here.
_INSTALL_LIED_ABOUT_SUCCESS_MARKER = "npm error Exit handler never called"


def _extract_missing_toolchain(output: str) -> Optional[str]:
    """Return the missing tool's name if our own preflight check reported
    it missing, else None. Never triggered by a project's own command
    happening to exit 127 for an unrelated reason."""
    idx = output.find(_TOOLCHAIN_MISSING_SENTINEL)
    if idx == -1:
        return None
    rest = output[idx + len(_TOOLCHAIN_MISSING_SENTINEL):].split(None, 1)
    return rest[0] if rest else None


def _install_failed(output: str) -> bool:
    """True when the dependency-install step itself failed."""
    return _INSTALL_FAILED_SENTINEL in output


# Every non-Python toolchain writes cache/config under $HOME by default (npm,
# go, cargo, gradle, dart/flutter's pub, dotnet's first-run files). Containers
# run as a raw, passwordless uid:gid ("1000:1000") for isolation, so $HOME is
# unset there and those writes would fail. Pointing everything at /tmp (world
# -writable in every base image used here) fixes that generically, for every
# ecosystem, without the engine needing to know which tool wants what.
_CONTAINER_ENV: Dict[str, str] = {
    "HOME": "/tmp",
    "NPM_CONFIG_CACHE": "/tmp/.npm-cache",
    "GOCACHE": "/tmp/.cache/go-build",
    "GOPATH": "/tmp/go",
    "CARGO_HOME": "/tmp/.cargo",
    "GRADLE_USER_HOME": "/tmp/.gradle",
    "PUB_CACHE": "/tmp/.pub-cache",
    "DOTNET_CLI_HOME": "/tmp",
    "DOTNET_NOLOGO": "1",
    "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
}


def _tool_is_available(argv0: str, workspace: Path) -> bool:
    """True if the verification command's executable actually exists.

    Checked host-side, before the subprocess fallback ever runs anything, so
    a missing toolchain is reported as UNABLE_TO_VERIFY instead of letting a
    FileNotFoundError happen mid-run. A repository-provided wrapper (e.g.
    "./gradlew") is resolved relative to the workspace and must be an
    executable file; a bare command name (e.g. "npm") is resolved via PATH.
    """
    if argv0.startswith("./") or argv0.startswith("../") or os.path.isabs(argv0):
        candidate = Path(argv0)
        if not candidate.is_absolute():
            candidate = workspace / argv0
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(argv0) is not None


def _preflight_snippet(argv0: str) -> str:
    """POSIX sh snippet that verifies ``argv0`` exists before anything runs.

    Mirrors ``_tool_is_available`` inside the container: ``[ -x path ]`` for
    a repository-provided wrapper, ``command -v`` (PATH lookup) for a bare
    command name. Exits 127 with the ``_TOOLCHAIN_MISSING_SENTINEL`` marker
    so ``_extract_missing_toolchain`` recognizes it unambiguously, and the
    install/test steps that follow never execute.
    """
    quoted = shlex.quote(argv0)
    return (
        f'{{ [ -x {quoted} ] || command -v {quoted} >/dev/null 2>&1; }} || '
        f'{{ echo "{_TOOLCHAIN_MISSING_SENTINEL}{argv0}" >&2; exit 127; }}'
    )


def _ensure_wrapper_executable(workspace: Path, argv0: str) -> None:
    """Best-effort chmod +x for a repository-provided wrapper script.

    A wrapper committed to a repo (mvnw, gradlew) is normally already
    executable via git's mode bits, but some checkout paths (zip download,
    certain CI clones) lose it. Restoring it here is harmless when it was
    already set and avoids a spurious "toolchain not found" on the wrapper
    itself.
    """
    if not (argv0.startswith("./") or argv0.startswith("../")):
        return
    wrapper = workspace / argv0
    try:
        if wrapper.is_file():
            wrapper.chmod(wrapper.stat().st_mode | 0o111)
    except OSError:
        pass


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
        _ensure_wrapper_executable(workspace, test_argv[0])

        if self._docker_runner.is_docker_available:
            output, exit_code = self._execute_in_docker(
                workspace, adapter.docker_image, install_argv, test_argv
            )
        else:
            output, exit_code = self._execute_in_subprocess(workspace, install_argv, test_argv)

        duration = time.time() - start_time

        missing_tool = _extract_missing_toolchain(output)
        if missing_tool:
            detail = (
                f"Required tool '{missing_tool}' for {adapter.ecosystem} verification is not "
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

        if _install_failed(output):
            if self.network_mode == "none":
                detail = (
                    "Project dependencies could not be installed because the verification "
                    "sandbox has no network access. This is not evidence that the reported "
                    "issue does or does not exist."
                )
            else:
                detail = (
                    f"Project dependencies could not be installed, so {adapter.ecosystem} "
                    "verification could not be run. This is not evidence that the reported "
                    "issue does or does not exist."
                )
            logger.warning(
                f"Dependency installation failed for {adapter.ecosystem} at {workspace}: {detail}"
            )
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
        self,
        workspace: Path,
        image: str,
        install_argv: Optional[List[str]],
        test_argv: List[str],
    ) -> Tuple[str, int]:
        """Run install + test commands in an ephemeral Docker container.

        Uses ``image`` -- the ecosystem's own adapter-declared toolchain
        image, e.g. node:20-slim for Node, golang:1.22-alpine for Go -- never
        a single shared image, so a Node project is never executed somewhere
        without npm. The required executable is verified present (via
        ``_preflight_snippet``) before install or test ever run, reported
        through ``_TOOLCHAIN_MISSING_SENTINEL`` rather than a bare exit code
        -- a project's own test command can also exit 127 for an unrelated
        reason (its own missing devDependency), and conflating the two
        wrongly blamed a present, working toolchain (see
        ``_extract_missing_toolchain``).

        If the install step itself fails (e.g. no network under
        ``network_mode="none"``), the script stops there rather than running
        the test command against an incomplete/absent dependency tree --
        it's reported via ``_INSTALL_FAILED_SENTINEL`` (see
        ``_install_failed``) instead of letting the test command fail on its
        own and misreport as a missing toolchain or a real test failure. The
        install step's own exit code is checked, but not trusted alone --
        npm can report success (exit 0) while having actually aborted
        (``_INSTALL_LIED_ABOUT_SUCCESS_MARKER``), so its own output is
        checked too.

        Execution is bounded by ``self.timeout`` using the Docker SDK's own
        timeout mechanism, ``Container.wait(timeout=...)``: the synchronous
        ``containers.run()`` helper has no way to bound how long it blocks,
        so this uses the equivalent manual create/start/wait/logs/remove
        sequence instead. A hung install or test is killed and reported as a
        clean timeout rather than blocking indefinitely.
        """
        script_parts = [_preflight_snippet(test_argv[0])]
        if install_argv:
            install_cmd = " ".join(install_argv)
            script_parts.append(
                f'{install_cmd} >/tmp/.repopilot-install.log 2>&1; ec=$?; '
                f'cat /tmp/.repopilot-install.log; '
                f'if [ "$ec" -ne 0 ] || '
                f'grep -q "{_INSTALL_LIED_ABOUT_SUCCESS_MARKER}" /tmp/.repopilot-install.log; then '
                f'echo "{_INSTALL_FAILED_SENTINEL}" >&2; exit 1; '
                f'fi'
            )
        script_parts.append('"$@"')
        script = "; ".join(script_parts)
        cmd = ["sh", "-c", script, "sh"] + test_argv

        container = None
        try:
            container = self._docker_runner._docker_client.containers.run(
                image=image,
                command=cmd,
                working_dir="/workspace",
                volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
                environment=_CONTAINER_ENV,
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
                    f"Docker verification run exceeded {self.timeout}s timeout; killing container."
                )
                try:
                    container.kill()
                except (DockerException, APIError):
                    pass
                return f"Verification timed out after {self.timeout} seconds.", 124

            exit_code = wait_result.get("StatusCode", 1)
            raw = container.logs(stdout=True, stderr=True)
            output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            return output, exit_code
        except (DockerException, APIError) as e:
            # Covers e.g. ImageNotFound -- an ecosystem image that hasn't
            # been pulled/built yet (see docker/sandbox/*/Dockerfile) -- as
            # well as any other daemon-level failure. Degrading to the
            # subprocess fallback still runs its own preflight check, so this
            # never silently reports success without the tool actually
            # existing somewhere it was run.
            logger.info(f"Docker verification run unavailable ({e}); falling back to subprocess.")
            return self._execute_in_subprocess(workspace, install_argv, test_argv)
        finally:
            # Explicit removal (rather than run()'s remove=True/auto_remove)
            # since detach=True is required to apply our own wait() timeout.
            if container is not None:
                try:
                    container.remove(force=True)
                except (DockerException, APIError):
                    pass

    def _execute_in_subprocess(
        self, workspace: Path, install_argv: Optional[List[str]], test_argv: List[str]
    ) -> Tuple[str, int]:
        """Run install (best-effort) + test commands as local subprocesses, cwd=workspace.

        Used when Docker itself is unavailable, so this executes against
        whatever toolchains happen to be on the *host* machine's PATH -- the
        host is never an implicit product requirement, just the last-resort
        fallback. The required executable is checked with
        ``_tool_is_available`` before anything is spawned: on a miss, install
        and test are both skipped and a synthetic "not found" result (exit
        127) is returned immediately, so a missing toolchain is never
        discovered mid-run via a FileNotFoundError.

        Unlike the Docker path, an install failure here is intentionally
        left non-fatal (the test command still runs): this executes directly
        against the host's own checkout/caches, which -- unlike a fresh,
        network-isolated container -- may already have the dependencies
        present from a prior install even if this particular install command
        failed, so a hard "dependencies unavailable" verdict isn't
        warranted.
        """
        if not _tool_is_available(test_argv[0], workspace):
            return (
                f"{_TOOLCHAIN_MISSING_SENTINEL}{test_argv[0]}\n"
                f"Required toolchain not found: '{test_argv[0]}' is not installed "
                "in this environment.",
                127,
            )

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
                f"{install_log}\n\n{_TOOLCHAIN_MISSING_SENTINEL}{test_argv[0]}\n"
                f"Required toolchain not found: {e}".strip(),
                127,
            )
        except Exception as e:
            return (f"{install_log}\n\nExecution error: {e}".strip(), 1)
