"""Evidence-based, multi-project repository analysis.

A repository is never assumed to contain exactly one language, framework, or
build system. This module walks the repository tree (bounded depth) looking
for the same manifest files each ``VerificationAdapter`` already recognizes,
groups them into independent ``ProjectInfo`` entries, and applies containment
rules so that platform scaffolding owned by a cross-platform framework (e.g.
a Flutter project's generated ``android/`` and ``ios/`` folders) is never
reported as its own, independent project.

The task description is never used to decide *what ecosystems exist* -- only
real repository evidence (manifests, dependency declarations, directory
structure) does that. ``select_relevant_projects`` uses the task description
and code-search evidence only to narrow *which already-detected* project(s)
are relevant to a given task.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

from app.services.verification.adapters.node_adapter import NodeAdapter
from app.services.verification.adapters.python_adapter import PythonAdapter
from app.services.verification.base import VerificationAdapter

# Directories never descended into: VCS internals, dependency caches, and
# build output. Nothing meaningful for project detection lives in these, and
# descending into them (especially node_modules) would be prohibitively slow
# and would surface vendored manifests as if they were real projects.
IGNORED_DIR_NAMES = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "build", "dist", ".dart_tool", ".gradle", ".idea", ".vscode",
    "target", "bin", "obj", ".next", ".nuxt", "vendor", "Pods", ".pub-cache",
}

# Subdirectory names that are platform scaffolding generated/managed by
# `flutter create` -- never independent projects even though they carry
# their own native build manifests (android/build.gradle, ios/*.xcodeproj,
# ...). Keyed by ecosystem so future cross-platform frameworks can add their
# own containment rule without touching the walking/exclusion logic itself.
_CONTAINED_SUBDIRS_BY_ECOSYSTEM: Dict[str, set] = {
    "flutter": {"android", "ios", "macos", "windows", "linux", "web"},
}

MAX_SCAN_DEPTH = 5


@dataclass
class ProjectInfo:
    """Structured, evidence-based description of one detected project."""

    root: str  # "." for the repository root, else a POSIX-style relative path
    ecosystem: str
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    build_system: Optional[str] = None
    package_manager: Optional[str] = None
    test_system: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0
    excluded_subpaths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "root": self.root,
            "ecosystem": self.ecosystem,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "build_system": self.build_system,
            "package_manager": self.package_manager,
            "test_system": self.test_system,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "excluded_subpaths": self.excluded_subpaths,
        }


_ECOSYSTEM_METADATA: Dict[str, Dict[str, object]] = {
    "go": {"languages": ["Go"], "build_system": "go", "package_manager": "go modules", "test_system": "go test"},
    "rust": {"languages": ["Rust"], "build_system": "cargo", "package_manager": "cargo", "test_system": "cargo test"},
    "dotnet": {"languages": ["C#"], "build_system": "dotnet", "package_manager": "NuGet", "test_system": "dotnet test"},
    "java-maven": {"languages": ["Java"], "build_system": "Maven", "package_manager": "Maven", "test_system": "mvn test"},
    "java-gradle": {"languages": ["Java", "Kotlin"], "build_system": "Gradle", "package_manager": "Gradle", "test_system": "gradle test"},
    "node": {"languages": ["JavaScript", "TypeScript"], "build_system": "node", "package_manager": None, "test_system": None},
    "python": {"languages": ["Python"], "build_system": "pip", "package_manager": "pip", "test_system": "pytest"},
    "flutter": {"languages": ["Dart"], "build_system": "flutter", "package_manager": "pub", "test_system": "flutter test"},
    "dart": {"languages": ["Dart"], "build_system": "dart", "package_manager": "pub", "test_system": "dart test"},
}


def _iter_candidate_dirs(root: Path, max_depth: int) -> List[Path]:
    """Breadth-bounded walk of the repository, skipping noise directories."""
    candidates = [root]

    def walk(current: Path, depth: int) -> None:
        if depth >= max_depth:
            return
        try:
            children = sorted((p for p in current.iterdir() if p.is_dir()), key=lambda p: p.name)
        except OSError:
            return
        for child in children:
            if child.name.startswith(".") or child.name in IGNORED_DIR_NAMES:
                continue
            candidates.append(child)
            walk(child, depth + 1)

    walk(root, 0)
    return candidates


def _detect_ecosystem_for_dir(directory: Path) -> Optional[Type[VerificationAdapter]]:
    """Pick the adapter (if any) whose manifest evidence is present in this
    exact directory, in ``ADAPTER_PRECEDENCE`` order.

    Deliberately does NOT use ``PythonAdapter.detect()``'s recursive
    test-file fallback here: that heuristic is only appropriate as a
    whole-repository last resort (applied once, separately, in ``analyze``)
    -- applying it per-directory during a multi-project walk would flag
    every subdirectory that happens to contain a stray ``test_*.py`` file
    (e.g. a docs/lint tool inside a Node or Java project) as its own,
    spurious Python project.
    """
    from app.services.verification.detector import ADAPTER_PRECEDENCE

    for adapter_cls in ADAPTER_PRECEDENCE:
        if adapter_cls is PythonAdapter:
            if PythonAdapter.find_manifests(directory):
                return PythonAdapter
            continue
        if adapter_cls.detect(directory):
            return adapter_cls
    return None


def _detect_node_frameworks(directory: Path) -> List[str]:
    import json

    pkg = directory / "package.json"
    try:
        data = json.loads(pkg.read_text(encoding="utf-8"))
    except Exception:
        return []

    deps: Dict[str, str] = {}
    deps.update(data.get("dependencies") or {})
    deps.update(data.get("devDependencies") or {})
    lower_deps = {str(k).lower() for k in deps}

    frameworks: List[str] = []
    if "next" in lower_deps:
        frameworks.append("Next.js")
    if "nuxt" in lower_deps or "nuxt3" in lower_deps:
        frameworks.append("Nuxt")
    if "@angular/core" in lower_deps:
        frameworks.append("Angular")
    if "vue" in lower_deps:
        frameworks.append("Vue")
    if "react-native" in lower_deps:
        frameworks.append("React Native")
    elif "react" in lower_deps:
        frameworks.append("React")
    if "express" in lower_deps:
        frameworks.append("Express")
    if (
        "vite" in lower_deps
        or (directory / "vite.config.ts").is_file()
        or (directory / "vite.config.js").is_file()
    ):
        frameworks.append("Vite")
    return frameworks


def _detect_python_frameworks(directory: Path) -> List[str]:
    text = ""
    for name in ("requirements.txt", "pyproject.toml", "setup.py"):
        p = directory / name
        if p.is_file():
            try:
                text += p.read_text(encoding="utf-8", errors="ignore").lower() + "\n"
            except Exception:
                pass

    frameworks: List[str] = []
    if "django" in text:
        frameworks.append("Django")
    if "fastapi" in text:
        frameworks.append("FastAPI")
    if "flask" in text:
        frameworks.append("Flask")
    return frameworks


def _detect_java_frameworks(directory: Path) -> List[str]:
    text = ""
    for name in ("pom.xml", "build.gradle", "build.gradle.kts"):
        p = directory / name
        if p.is_file():
            try:
                text += p.read_text(encoding="utf-8", errors="ignore").lower() + "\n"
            except Exception:
                pass

    if "spring-boot" in text or "springframework.boot" in text:
        return ["Spring Boot"]
    if "springframework" in text:
        return ["Spring"]
    return []


def _build_project_info(root: Path, directory: Path, adapter_cls: Type[VerificationAdapter]) -> ProjectInfo:
    ecosystem = adapter_cls.ecosystem
    rel = "." if directory == root else directory.relative_to(root).as_posix()
    meta = _ECOSYSTEM_METADATA.get(ecosystem, {})

    manifests = adapter_cls.find_manifests(directory)
    evidence = [f"{m} at {rel}" for m in manifests] if manifests else [f"{ecosystem} evidence detected at {rel}"]

    frameworks: List[str] = []
    package_manager = meta.get("package_manager")
    if ecosystem == "node":
        frameworks = _detect_node_frameworks(directory)
        package_manager = NodeAdapter().package_manager(directory)
    elif ecosystem == "python":
        frameworks = _detect_python_frameworks(directory)
    elif ecosystem in ("java-maven", "java-gradle"):
        frameworks = _detect_java_frameworks(directory)
    elif ecosystem == "flutter":
        frameworks = ["Flutter"]

    return ProjectInfo(
        root=rel,
        ecosystem=ecosystem,
        languages=list(meta.get("languages", [])),
        frameworks=frameworks,
        build_system=meta.get("build_system"),
        package_manager=package_manager,
        test_system=meta.get("test_system"),
        evidence=evidence,
        confidence=1.0 if manifests else 0.6,
    )


def _apply_containment(projects: List[ProjectInfo]) -> List[ProjectInfo]:
    """Fold platform-scaffolding subdirectories into their owning project."""

    def depth(p: ProjectInfo) -> int:
        return 0 if p.root == "." else len(Path(p.root).parts)

    ordered = sorted(projects, key=depth)
    excluded_roots: set = set()
    kept: List[ProjectInfo] = []

    for proj in ordered:
        if proj.root in excluded_roots:
            continue
        kept.append(proj)

        contained_names = _CONTAINED_SUBDIRS_BY_ECOSYSTEM.get(proj.ecosystem)
        if not contained_names:
            continue

        prefix = "" if proj.root == "." else proj.root.rstrip("/") + "/"
        for other in ordered:
            if other is proj or other.root in excluded_roots:
                continue
            if not other.root.startswith(prefix):
                continue
            tail = other.root[len(prefix):]
            first_segment = tail.split("/")[0]
            if first_segment in contained_names:
                excluded_roots.add(other.root)
                proj.excluded_subpaths.append(other.root)

    return kept


def analyze(root: Path, max_depth: int = MAX_SCAN_DEPTH) -> List[ProjectInfo]:
    """Detect every real project in a repository, evidence-first.

    Returns an empty list only when nothing recognizable exists anywhere in
    the tree (callers should treat that as "unsupported ecosystem", never as
    a silent single-project default).
    """
    root = Path(root).resolve()
    candidates = _iter_candidate_dirs(root, max_depth)

    hits: List[Tuple[Path, Type[VerificationAdapter]]] = []
    root_matched = False
    for directory in candidates:
        adapter_cls = _detect_ecosystem_for_dir(directory)
        if adapter_cls is not None:
            hits.append((directory, adapter_cls))
            if directory == root:
                root_matched = True

    if not root_matched:
        # Preserve RepoPilot's original unconditional whole-repo default: a
        # manifest-less repository with pytest-style test files anywhere is
        # still a Python project, exactly like ProjectDetector.detect() at
        # the root level.
        if any(root.rglob("test_*.py")) or any(root.rglob("*_test.py")):
            hits.append((root, PythonAdapter))

    if not hits:
        return []

    projects = [_build_project_info(root, directory, adapter_cls) for directory, adapter_cls in hits]
    return _apply_containment(projects)


class RepositoryAnalyzer:
    """Entry point for multi-project, evidence-based repository analysis."""

    @staticmethod
    def analyze(root: Path, max_depth: int = MAX_SCAN_DEPTH) -> List[ProjectInfo]:
        return analyze(root, max_depth=max_depth)


def select_relevant_projects(
    projects: Sequence[ProjectInfo],
    task_description: str = "",
    keyword_matches: Optional[Sequence[Dict]] = None,
) -> List[ProjectInfo]:
    """Narrow detected projects to the one(s) relevant to a task.

    The task description is a hypothesis, not ground truth: mentioning a
    project's ecosystem/language/framework by name earns it only a small
    assist. Real evidence -- where investigation keyword matches physically
    live in the repository -- is weighted more heavily and wins whenever the
    two disagree (e.g. the task says "Next.js" but the matching code lives
    in the plain React/Vite project). When nothing points to a specific
    project, every detected project is returned rather than guessing.
    """
    if not projects:
        return []
    if len(projects) == 1:
        return list(projects)

    desc_lower = (task_description or "").lower()
    matches = keyword_matches or []

    # Real code evidence outweighs the task merely naming a framework: a
    # single keyword match under a project's root counts for more than a
    # textual mention of that project's ecosystem/language/framework.
    _KEYWORD_MATCH_WEIGHT = 3.0
    _MENTION_WEIGHT = 1.0

    scores: Dict[str, float] = {}
    for proj in projects:
        score = 0.0
        tokens = {proj.ecosystem.lower()}
        tokens.update(l.lower() for l in proj.languages)
        tokens.update(f.lower() for f in proj.frameworks)
        for token in tokens:
            if token and token in desc_lower:
                score += _MENTION_WEIGHT

        prefix = "" if proj.root == "." else proj.root.rstrip("/") + "/"
        for match in matches:
            file_path = match.get("file", "") if isinstance(match, dict) else ""
            if isinstance(file_path, str) and file_path.startswith(prefix):
                score += _KEYWORD_MATCH_WEIGHT

        scores[proj.root] = score

    max_score = max(scores.values())
    if max_score <= 0:
        return list(projects)
    return [p for p in projects if scores[p.root] == max_score]
