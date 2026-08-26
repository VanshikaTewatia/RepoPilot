"""Deterministic project-ecosystem detection from workspace manifest files.

Precedence when multiple ecosystems' manifests are present in the same
workspace (first match wins):

    Go > Rust > .NET > Flutter > Dart > Java(Maven) > Java(Gradle) > Node > Python

Rationale: go.mod, Cargo.toml, *.sln/*.csproj, and pubspec.yaml are
single-purpose, unambiguous build manifests that are essentially never
present unless that toolchain is the project's actual build system. Node's
package.json and Python's requirements.txt/pyproject.toml, by contrast, are
common as *secondary* tooling manifests (e.g. a repo whose primary language
is something else may still carry a small package.json for a docs/lint tool,
or a requirements.txt for a Python-based CI script). Python is checked last
both for that reason and because it was RepoPilot's historical default --
putting it last ensures a more specific manifest correctly takes precedence
when one is also present, while repos that are purely Python (like
demo_repo, which has only pyproject.toml) are unaffected and still resolve
to Python.

Flutter/Dart is placed ahead of Java so that a Flutter project's nested
``android/build.gradle`` scaffolding can never cause the *root* workspace
itself to be misclassified as Java/Gradle if both happen to be visible to a
single ``detect()`` call; the primary defense against that misclassification
in practice is that ``JavaGradleAdapter`` only ever looks for
``build.gradle``/``build.gradle.kts`` at the exact directory passed in, and
multi-project repositories are analyzed per-directory by
``app.services.verification.project_analyzer``, which also explicitly
excludes a Flutter project's ``android``/``ios``/etc. subdirectories from
being reported as independent projects at all.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Type

from app.services.verification.adapters import (
    DartAdapter,
    DotnetAdapter,
    FlutterAdapter,
    GoAdapter,
    JavaGradleAdapter,
    JavaMavenAdapter,
    NodeAdapter,
    PythonAdapter,
    RustAdapter,
)
from app.services.verification.base import VerificationAdapter

ADAPTER_PRECEDENCE: List[Type[VerificationAdapter]] = [
    GoAdapter,
    RustAdapter,
    DotnetAdapter,
    FlutterAdapter,
    DartAdapter,
    JavaMavenAdapter,
    JavaGradleAdapter,
    NodeAdapter,
    PythonAdapter,
]


@dataclass
class DetectionResult:
    """Result of scanning a workspace for a known project ecosystem."""

    adapter: Optional[VerificationAdapter]
    ecosystem: Optional[str]
    manifests_found: List[str] = field(default_factory=list)
    all_manifests_found: List[str] = field(default_factory=list)
    manifests_scanned: List[str] = field(default_factory=list)


class ProjectDetector:
    """Inspects a workspace directory and selects a verification adapter."""

    @staticmethod
    def detect(workspace: Path) -> DetectionResult:
        workspace = Path(workspace)

        manifests_scanned: List[str] = []
        for adapter_cls in ADAPTER_PRECEDENCE:
            manifests_scanned.extend(adapter_cls.manifest_files)

        matched_cls: Optional[Type[VerificationAdapter]] = None
        matched_manifests: List[str] = []
        all_found: List[str] = []

        for adapter_cls in ADAPTER_PRECEDENCE:
            # detect() is consulted for every adapter regardless of whether
            # find_manifests() found anything -- some adapters (e.g. Python)
            # define a fallback detection strategy that isn't manifest-based.
            found = adapter_cls.find_manifests(workspace)
            if found:
                all_found.extend(found)
            if matched_cls is None and adapter_cls.detect(workspace):
                matched_cls = adapter_cls
                matched_manifests = found

        if matched_cls is None:
            return DetectionResult(
                adapter=None,
                ecosystem=None,
                manifests_found=[],
                all_manifests_found=sorted(set(all_found)),
                manifests_scanned=sorted(set(manifests_scanned)),
            )

        return DetectionResult(
            adapter=matched_cls(),
            ecosystem=matched_cls.ecosystem,
            manifests_found=matched_manifests,
            all_manifests_found=sorted(set(all_found)),
            manifests_scanned=sorted(set(manifests_scanned)),
        )
