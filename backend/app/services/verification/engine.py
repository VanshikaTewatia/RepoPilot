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
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.services.sandbox.docker_runner import DockerTestRunner, _detect_dependency_install_args
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

# Phase 8 / Task #31: literal, fixed markers the verification script itself
# echoes immediately before starting each phase -- never anything parsed out
# of a project's own install/test tool output, which varies by tool and
# version and would be fragile to match. Read back (via container.logs(),
# now fetched even after a timeout -- see _execute_in_docker) to tell a
# timeout that happened while dependencies were still being installed apart
# from one that happened during the project's own test run, without
# guessing at ecosystem-specific output.
_PHASE_INSTALL_MARKER = "REPOPILOT_PHASE:install"
_PHASE_TEST_MARKER = "REPOPILOT_PHASE:test"

# Emitted onto the returned output (alongside the ordinary timeout message)
# ONLY when a Docker wait() timeout is positively confirmed -- via the phase
# markers above -- to have happened while the dependency-install step was
# still running: an environment/dependency-preparation failure, not a
# verdict on the reported issue (see Task #31: npm ci retried DNS lookups
# under network_mode="none" for ~75s, well past the 45s verification
# timeout, and was misclassified as an ordinary failing test). Never emitted
# for a timeout during the test phase, and never emitted/guessed when no
# phase marker was observed at all -- see _execute_in_docker's timeout
# handling, which treats both of those as an ordinary, still-retryable
# timeout exactly as before this fix.
_TIMEOUT_DURING_INSTALL_SENTINEL = "REPOPILOT_TIMEOUT_DURING_INSTALL"


def _timed_out_during_install(output: str) -> bool:
    """True only when a Docker wait() timeout was positively confirmed (via
    phase markers) to have happened while the dependency-install step was
    still running. Never true for a timeout during the test phase, and
    never guessed when no phase marker was observed at all."""
    return _TIMEOUT_DURING_INSTALL_SENTINEL in output


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
    # Widely-respected, generic convention (Jest/CRA/Mocha/Cypress and many
    # other tools across ecosystems all check this) for "running
    # non-interactively" -- e.g. it's what keeps `react-scripts test`
    # (CRA's default Jest wrapper) from launching interactive watch mode,
    # which would otherwise never terminate inside a non-TTY container.
    # Deliberately generic rather than a framework-specific flag (e.g.
    # `--watchAll=false`) since RepoPilot has no reliable way to know which
    # test framework a given project's "test" script actually invokes.
    "CI": "1",
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

    def execute_command(
        self,
        workspace: Path | str,
        command: List[str],
        image: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run an arbitrary command with the same sandboxing guarantees as
        verification (Docker when available, hardened subprocess fallback
        otherwise; toolchain preflight; timeout handling; cleanup) -- but
        with no ecosystem detection and no pass/fail interpretation. Returns
        raw execution facts only.

        Intended for callers that need a command run that isn't a
        verification adapter's own test command (e.g.
        app.services.baseline's reproduction executor), without duplicating
        any of the Docker/subprocess execution machinery below.
        """
        workspace_path = Path(workspace).resolve()
        exec_image = image or self._docker_runner.image

        # `timeout` is passed straight through to the execution methods below
        # rather than mutating `self.timeout` -- this engine instance may be
        # reused across multiple execute_command() calls (e.g. one per
        # baseline reproduction command), and mutating shared instance state
        # for the duration of a call is not safe if two such calls are ever
        # in flight concurrently on the same engine.
        if self._docker_runner.is_docker_available:
            output, exit_code = self._execute_in_docker(
                workspace_path, exec_image, None, command, timeout=timeout
            )
        else:
            output, exit_code = self._execute_in_subprocess(
                workspace_path, None, command, timeout=timeout
            )

        return {
            "output": output,
            "exit_code": exit_code,
            "toolchain_missing": _extract_missing_toolchain(output),
        }

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

        # Phase 7: Python is no longer special-cased to the legacy
        # DockerTestRunner.run_tests() delegation -- that path predates (and
        # never applied) the install-failure/toolchain-missing classification
        # _run_adapter already provides for every other ecosystem, so a
        # Python dependency-install failure under network_mode="none" was
        # never distinguishable from a genuine test failure (always
        # available=True). Routing Python through the same _run_adapter path
        # every other ecosystem already uses fixes that classification for
        # free -- see PythonAdapter.docker_image (a pre-baked image carrying
        # pytest) and _run_adapter's Python-specific host-side dependency
        # install below for what makes verification actually able to run,
        # not just fail honestly.
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

        # Phase 7: for Python specifically, resolve the target repository's
        # OWN dependencies (never the harness's own tooling -- pytest itself
        # is pre-baked into PythonAdapter.docker_image, see the Dockerfile
        # under docker/sandbox/python/) into an ISOLATED directory, outside
        # both the network-isolated Docker container and RepoPilot's own
        # live Python environment, then expose the result via PYTHONPATH.
        # Mirrors DockerTestRunner._install_dependencies's existing
        # "install outside, mount inside" pattern (used today only by the
        # legacy Python subprocess fallback) -- applied here to BOTH
        # execution paths below, Docker (read-only bind mount) and
        # subprocess (PYTHONPATH env var only, same isolated directory),
        # so neither path ever runs a bare `pip install .` directly against
        # a network-isolated container (impossible) or directly into this
        # backend process's own environment (unsafe) -- both were real
        # risks of naively routing Python through the fully-generic
        # execution helpers below unchanged. Only Python gets this
        # treatment; every other ecosystem's install/test flow is
        # completely unchanged.
        host_deps_dir: Optional[str] = None
        extra_volumes: Optional[Dict[str, Dict[str, str]]] = None
        extra_env: Optional[Dict[str, str]] = None

        if adapter.ecosystem == "python" and install_argv:
            py_install_args = _detect_dependency_install_args(workspace)
            if py_install_args:
                host_deps_dir, host_install_log = self._install_python_deps_isolated(
                    workspace, py_install_args
                )
                if host_deps_dir is None:
                    duration = time.time() - start_time
                    detail = (
                        "Project dependencies could not be installed ahead of verification "
                        "(the isolated pip install failed), so Python verification could not "
                        "be run. This is not evidence that the reported issue does or does not "
                        "exist."
                    )
                    logger.warning(
                        f"Isolated Python dependency installation failed at {workspace}: {detail}"
                    )
                    return VerificationResult(
                        ecosystem=adapter.ecosystem,
                        success=False,
                        exit_code=1,
                        output=host_install_log,
                        passed=0,
                        failed=0,
                        duration=round(duration, 2),
                        available=False,
                        manifests_found=manifests_found,
                        detail=detail,
                        command=" ".join(test_argv),
                    ).to_dict()
                extra_volumes = {host_deps_dir: {"bind": "/repopilot-deps", "mode": "ro"}}
                extra_env = {"PYTHONPATH": "/repopilot-deps"}
                # Already installed in isolation -- neither execution path
                # below needs its own install step (Docker: no network
                # needed either; subprocess: never touches this process's
                # own environment).
                install_argv = None

        # Phase 8 (Task #31): Node dependency installation moves the same
        # direction as Python's above, but NOT via a host subprocess --
        # `npm ci`/`install` (and pnpm/yarn's equivalents) run arbitrary
        # repository-declared lifecycle scripts (preinstall/install/
        # postinstall) for the top-level package AND every transitive
        # dependency, a far more commonly-abused supply-chain vector than
        # Python's sdist build step, so running it directly on the backend
        # host (even into an isolated --target directory) was rejected.
        # Instead this uses a second, throwaway, network-ENABLED Docker
        # container (the one deliberate exception to network_mode="none" in
        # this codebase, scoped to exactly this step) with lifecycle
        # scripts explicitly disabled via --ignore-scripts (see
        # NodeAdapter.dependency_prep_command) -- see
        # _prepare_node_deps_isolated for the full security boundary. Only
        # engaged when Docker itself is available (this step launches its
        # own container); when Docker is unavailable, Node keeps its
        # existing, unchanged subprocess fallback further down (running
        # install_argv as before -- a pre-existing, unrelated code path).
        if adapter.ecosystem == "node" and install_argv and self._docker_runner.is_docker_available:
            host_deps_dir, host_install_log = self._prepare_node_deps_isolated(workspace, adapter)
            if host_deps_dir is None:
                duration = time.time() - start_time
                detail = (
                    "Node dependencies could not be prepared ahead of verification "
                    "(the isolated, network-enabled dependency-preparation container "
                    "failed), so Node verification could not be run. This is not "
                    "evidence that the reported issue does or does not exist."
                )
                logger.warning(
                    f"Isolated Node dependency preparation failed at {workspace}: {detail}"
                )
                return VerificationResult(
                    ecosystem=adapter.ecosystem,
                    success=False,
                    exit_code=1,
                    output=host_install_log,
                    passed=0,
                    failed=0,
                    duration=round(duration, 2),
                    available=False,
                    manifests_found=manifests_found,
                    detail=detail,
                    command=" ".join(test_argv),
                ).to_dict()
            node_modules_dir = Path(host_deps_dir) / "node_modules"
            if node_modules_dir.is_dir():
                extra_volumes = {str(node_modules_dir): {"bind": "/workspace/node_modules", "mode": "ro"}}
            # No extra_env needed here (unlike Python's PYTHONPATH above) --
            # Node module resolution just walks up from cwd looking for a
            # node_modules directory, so mounting it directly at
            # /workspace/node_modules is sufficient on its own.
            install_argv = None

        try:
            if self._docker_runner.is_docker_available:
                output, exit_code = self._execute_in_docker(
                    workspace,
                    adapter.docker_image,
                    install_argv,
                    test_argv,
                    extra_volumes=extra_volumes,
                    extra_env=extra_env,
                )
            else:
                output, exit_code = self._execute_in_subprocess(
                    workspace, install_argv, test_argv, extra_env=extra_env
                )
        finally:
            if host_deps_dir:
                shutil.rmtree(host_deps_dir, ignore_errors=True)

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

        if _timed_out_during_install(output):
            # Task #31: a Docker wait() timeout confirmed (via phase
            # markers, see _execute_in_docker) to have happened while
            # dependency installation was still running -- an environment/
            # tooling failure, never a verdict on the reported issue.
            # Positively confirmed only; a timeout during the test phase, or
            # with no phase marker observed at all, does not reach here (see
            # _execute_in_docker's own conservative classification).
            detail = (
                f"Dependency installation had not finished when the {self.timeout}s "
                f"verification timeout was reached, so {adapter.ecosystem} verification "
                "could not be run. This is an environment/tooling problem, not evidence "
                "that the reported issue does or does not exist."
            )
            logger.warning(
                f"Verification timed out during dependency installation for "
                f"{adapter.ecosystem} at {workspace}: {detail}"
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
        timeout: Optional[int] = None,
        extra_volumes: Optional[Dict[str, Dict[str, str]]] = None,
        extra_env: Optional[Dict[str, str]] = None,
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

        Execution is bounded by ``timeout`` (falling back to ``self.timeout``
        when not given -- every existing caller omits it and gets identical
        behavior to before) using the Docker SDK's own timeout mechanism,
        ``Container.wait(timeout=...)``: the synchronous ``containers.run()``
        helper has no way to bound how long it blocks, so this uses the
        equivalent manual create/start/wait/logs/remove sequence instead. A
        hung install or test is killed and reported as a clean timeout
        rather than blocking indefinitely. Passed through explicitly rather
        than mutating ``self.timeout`` so concurrent calls on the same
        engine instance can never leak a custom timeout into one another.

        ``extra_volumes``/``extra_env`` (Phase 7): additional read-only host
        mounts and environment variables for this one run only, used by
        ``_run_adapter`` to give Python verification its host-side-installed
        dependencies via a read-only mount and ``PYTHONPATH`` -- both
        default to ``None``, identical to omitting them, so no other
        ecosystem's call is affected.

        Phase 8 (Task #31): the script echoes ``_PHASE_INSTALL_MARKER``/
        ``_PHASE_TEST_MARKER`` immediately before each phase starts, so a
        ``Container.wait(timeout=...)`` timeout can be told apart -- "still
        installing dependencies" vs. "the project's own test command is
        running" -- from whatever the container had actually flushed by the
        time it was killed, without ever parsing the install/test tool's own
        (ecosystem-specific, version-dependent) output.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        script_parts = [_preflight_snippet(test_argv[0])]
        if install_argv:
            install_cmd = " ".join(install_argv)
            script_parts.append(f'echo "{_PHASE_INSTALL_MARKER}"')
            script_parts.append(
                f'{install_cmd} >/tmp/.repopilot-install.log 2>&1; ec=$?; '
                f'cat /tmp/.repopilot-install.log; '
                f'if [ "$ec" -ne 0 ] || '
                f'grep -q "{_INSTALL_LIED_ABOUT_SUCCESS_MARKER}" /tmp/.repopilot-install.log; then '
                f'echo "{_INSTALL_FAILED_SENTINEL}" >&2; exit 1; '
                f'fi'
            )
        script_parts.append(f'echo "{_PHASE_TEST_MARKER}"')
        script_parts.append('"$@"')
        script = "; ".join(script_parts)
        cmd = ["sh", "-c", script, "sh"] + test_argv

        volumes = {str(workspace): {"bind": "/workspace", "mode": "rw"}}
        if extra_volumes:
            volumes.update(extra_volumes)
        environment = dict(_CONTAINER_ENV)
        if extra_env:
            environment.update(extra_env)

        container = None
        try:
            container = self._docker_runner._docker_client.containers.run(
                image=image,
                command=cmd,
                working_dir="/workspace",
                volumes=volumes,
                environment=environment,
                network_mode=self.network_mode,
                nano_cpus=int(settings.sandbox_max_cpu * 1e9),
                mem_limit=f"{settings.sandbox_max_memory_mb}m",
                detach=True,
                user="1000:1000",
            )
            try:
                wait_result = container.wait(timeout=effective_timeout)
            except _DOCKER_WAIT_TIMEOUT_EXCEPTIONS:
                logger.warning(
                    f"Docker verification run exceeded {effective_timeout}s timeout; killing container."
                )
                # Phase 8 (Task #31): always retrieve whatever the container
                # had actually flushed before being killed -- previously
                # discarded entirely, which made a timeout during dependency
                # installation indistinguishable from one during the
                # project's own test run. container.logs() works on a
                # running/just-killed container the same as a finished one.
                partial_output = ""
                try:
                    raw = container.logs(stdout=True, stderr=True)
                    partial_output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                except (DockerException, APIError):
                    pass
                try:
                    container.kill()
                except (DockerException, APIError):
                    pass

                timeout_msg = f"Verification timed out after {effective_timeout} seconds."
                output = f"{partial_output}\n\n{timeout_msg}".strip() if partial_output else timeout_msg

                saw_install_marker = _PHASE_INSTALL_MARKER in partial_output
                saw_test_marker = _PHASE_TEST_MARKER in partial_output
                if install_argv and saw_install_marker and not saw_test_marker:
                    # Positively confirmed: the container was still inside
                    # the dependency-install step when the timeout fired --
                    # an environment/dependency-preparation failure, not
                    # evidence about the reported issue. _run_adapter
                    # classifies this as available=False via
                    # _TIMEOUT_DURING_INSTALL_SENTINEL.
                    output = f"{output}\n{_TIMEOUT_DURING_INSTALL_SENTINEL}"
                # Every other case -- no install step existed, the test
                # phase had already started (a slow/hanging test command --
                # never automatically an environment failure), or no phase
                # marker was observed at all (e.g. logs hadn't flushed in
                # time) -- is left as an ordinary, still-retryable timeout
                # exactly as before this fix. Never guessed at.
                return output, 124

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
            return self._execute_in_subprocess(workspace, install_argv, test_argv, timeout=timeout)
        finally:
            # Explicit removal (rather than run()'s remove=True/auto_remove)
            # since detach=True is required to apply our own wait() timeout.
            if container is not None:
                try:
                    container.remove(force=True)
                except (DockerException, APIError):
                    pass

    def _install_python_deps_isolated(
        self, workspace: Path, install_args: List[str]
    ) -> Tuple[Optional[str], str]:
        """Best-effort ``pip install --target`` for a Python project's OWN
        declared dependencies, run into a fresh, isolated directory --
        never the network-isolated verification container, and never
        RepoPilot's own live Python environment (Phase 7).

        Mirrors ``DockerTestRunner._install_dependencies`` -- the same
        "install outside, mount inside" pattern already used today by the
        legacy Python subprocess fallback -- generalized here for
        ``_run_adapter`` so BOTH the Docker path (read-only bind mount) and
        the subprocess path (``PYTHONPATH`` only) can verify a Python
        project with real third-party dependencies without either a
        ``network_mode="none"`` container needing network access, or a bare
        ``pip install .`` landing directly in this backend process's own
        environment. Installs into a fresh, per-call temp directory (never
        shared across calls); the caller is responsible for removing it
        once the run it served has finished.

        Returns ``(target_dir, log)``. ``target_dir`` is ``None`` when the
        install itself failed or errored -- the caller treats that as an
        environment/setup failure (``available=False``), never as a masked
        test failure: retrying the identical install (in a container with
        no network, or against the same unresolvable requirement) would
        fail identically, so there is no "maybe it'll work anyway" fallback
        to attempt.
        """
        target_dir = tempfile.mkdtemp(prefix="repopilot_pydeps_")
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
                timeout=min(self.timeout, _INSTALL_TIMEOUT_SECONDS) if self.timeout else _INSTALL_TIMEOUT_SECONDS,
            )
            log = (
                f"$ pip install --target <isolated dir> {' '.join(install_args)}\n"
                f"{result.stdout}\n{result.stderr}"
            ).strip()
            if result.returncode != 0:
                shutil.rmtree(target_dir, ignore_errors=True)
                return None, log
            return target_dir, log
        except subprocess.TimeoutExpired:
            shutil.rmtree(target_dir, ignore_errors=True)
            return None, f"Host-side dependency installation timed out after {_INSTALL_TIMEOUT_SECONDS}s."
        except Exception as e:
            shutil.rmtree(target_dir, ignore_errors=True)
            return None, f"Host-side dependency installation error: {e}"

    def _prepare_node_deps_isolated(
        self, workspace: Path, adapter: VerificationAdapter
    ) -> Tuple[Optional[str], str]:
        """Install a Node project's OWN declared dependencies inside a
        throwaway, network-ENABLED Docker container -- never on the
        RepoPilot backend host, and never inside the network-isolated
        verification container itself (Phase 8 / Task #31).

        This is deliberately NOT a Node port of
        ``_install_python_deps_isolated`` above: ``npm ci``/``install`` (and
        pnpm/yarn's equivalents) can run arbitrary repository-controlled
        lifecycle scripts (``preinstall``/``install``/``postinstall``) for
        the top-level package AND every transitive dependency -- a far more
        commonly-abused supply-chain vector than Python's sdist build step
        (most PyPI packages today ship prebuilt wheels, sidestepping
        arbitrary code at install time far more often than npm does).
        Running that directly on the backend host, even into an isolated
        ``--target`` directory, would let any untrusted repository's
        dependency tree execute code with this process's own privileges --
        ruled out for exactly that reason.

        The security boundary here is instead a second, ephemeral
        container:
          - Uses the SAME fixed, adapter-declared image verification itself
            uses (``adapter.docker_image``) -- never a repository-influenced
            or otherwise dynamic image name.
          - Gets Docker's *default* bridged network (network_mode is simply
            omitted below) -- the one deliberate, narrowly-scoped exception
            to ``network_mode="none"`` anywhere in this codebase, needed
            because dependency resolution genuinely requires reaching the
            package registry. The actual verification container that runs
            the project's own test command is completely unaffected and
            stays network-isolated exactly as before (see ``_run_adapter``,
            which sets this container's *output* -- a read-only
            ``node_modules`` mount -- as the ONLY thing that crosses back
            into that network-isolated run).
          - Mounts ONLY a fresh, empty temp directory containing a COPY of
            ``package.json``/the lockfile -- never the real workspace being
            verified, never any other host path, and deliberately never a
            repository-provided ``.npmrc`` (which could redirect the
            registry or otherwise reconfigure npm).
          - Never mounts the Docker socket and is never run privileged.
          - Runs as the same unprivileged ``uid:gid`` as every other
            sandbox container.
          - Gets no backend credentials/secrets: ``environment`` here is the
            same generic ``_CONTAINER_ENV`` (HOME/npm cache location) every
            other ecosystem's container already receives -- nothing else.
          - Forces lifecycle scripts off via
            ``adapter.dependency_prep_command()`` (``--ignore-scripts``),
            so nothing the repository declares as a dependency ever gets to
            execute code here even with real network access. A package that
            genuinely needs a postinstall step (e.g. a native binary
            download) may legitimately fail to install under this
            constraint -- that is an accepted, honest "could not verify"
            outcome, not something this method works around.
          - Is bounded by its own timeout (``_INSTALL_TIMEOUT_SECONDS``),
            entirely separate from the 45s verification budget.

        Returns ``(staging_dir, log)``. ``staging_dir`` is ``None`` when the
        preparation itself failed, timed out, or errored -- the caller
        treats that as an environment/setup failure (``available=False``),
        exactly like a Python dependency-install failure; retrying
        identically would fail identically. On success, ``staging_dir`` may
        or may not contain a populated ``node_modules`` (a project with zero
        real dependencies legitimately produces none) -- the caller checks
        for that directory's existence before mounting anything. The caller
        is responsible for removing ``staging_dir`` once the run it served
        has finished.
        """
        prep_cmd = adapter.dependency_prep_command(workspace)
        if not prep_cmd:
            return None, "No dependency preparation command available for this Node project."

        staging_dir = tempfile.mkdtemp(prefix="repopilot_nodedeps_")
        for manifest in ("package.json", "pnpm-lock.yaml", "yarn.lock", "package-lock.json"):
            src = workspace / manifest
            if src.is_file():
                shutil.copy2(src, Path(staging_dir) / manifest)

        script = f'{_preflight_snippet(prep_cmd[0])}; {" ".join(prep_cmd)}'
        cmd = ["sh", "-c", script]

        container = None
        try:
            container = self._docker_runner._docker_client.containers.run(
                image=adapter.docker_image,
                command=cmd,
                working_dir="/deps",
                volumes={staging_dir: {"bind": "/deps", "mode": "rw"}},
                environment=dict(_CONTAINER_ENV),
                # network_mode intentionally omitted -- Docker's default
                # bridged (outbound-only) network, never the host network,
                # never the Docker socket. See the docstring above.
                nano_cpus=int(settings.sandbox_max_cpu * 1e9),
                mem_limit=f"{settings.sandbox_max_memory_mb}m",
                detach=True,
                user="1000:1000",
            )
            try:
                wait_result = container.wait(timeout=_INSTALL_TIMEOUT_SECONDS)
            except _DOCKER_WAIT_TIMEOUT_EXCEPTIONS:
                try:
                    container.kill()
                except (DockerException, APIError):
                    pass
                shutil.rmtree(staging_dir, ignore_errors=True)
                return None, (
                    f"Node dependency preparation timed out after "
                    f"{_INSTALL_TIMEOUT_SECONDS}s in the isolated, network-enabled "
                    "preparation container."
                )

            exit_code = wait_result.get("StatusCode", 1)
            raw = container.logs(stdout=True, stderr=True)
            log = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        except (DockerException, APIError) as e:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return None, f"Node dependency preparation container error: {e}"
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except (DockerException, APIError):
                    pass

        missing_tool = _extract_missing_toolchain(log)
        if missing_tool:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return None, (
                f"Required tool '{missing_tool}' is not available for Node dependency "
                f"preparation.\n{log}"
            )
        if exit_code != 0:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return None, log

        return staging_dir, log

    def _execute_in_subprocess(
        self,
        workspace: Path,
        install_argv: Optional[List[str]],
        test_argv: List[str],
        timeout: Optional[int] = None,
        extra_env: Optional[Dict[str, str]] = None,
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

        ``extra_env`` (Phase 7): additional environment variables for the
        TEST command only (e.g. ``PYTHONPATH`` pointing at
        ``_install_python_deps_isolated``'s isolated install directory) --
        defaults to ``None``, identical to omitting it, so no other
        ecosystem's call is affected.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
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
                    timeout=min(effective_timeout, _INSTALL_TIMEOUT_SECONDS),
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

        env = None
        if extra_env:
            env = os.environ.copy()
            env.update(extra_env)

        try:
            result = subprocess.run(
                test_argv,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=env,
            )
            combined = f"{install_log}\n\n{result.stdout}\n{result.stderr}".strip()
            return combined, result.returncode
        except subprocess.TimeoutExpired:
            return (
                f"{install_log}\n\nExecution timed out after {effective_timeout} seconds.".strip(),
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
