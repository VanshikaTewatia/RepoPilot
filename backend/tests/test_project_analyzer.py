"""Unit tests for RepositoryAnalyzer: multi-project and nested-project
detection, framework inference, and evidence-based project selection."""

import json
import tempfile
from pathlib import Path

from app.services.verification.project_analyzer import (
    RepositoryAnalyzer,
    select_relevant_projects,
)

_FLUTTER_PUBSPEC = """\
name: my_app
dependencies:
  flutter:
    sdk: flutter
"""

_DART_ONLY_PUBSPEC = """\
name: my_lib
dependencies:
  path: ^1.8.0
"""


def _write(root: Path, rel_path: str, content: str = "") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _roots(projects):
    return sorted(p.root for p in projects)


# ---------------------------------------------------------------------------
# Flutter + nested Android/iOS must not be reported as separate projects
# ---------------------------------------------------------------------------
def test_flutter_with_android_gradle_detected_as_single_flutter_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pubspec.yaml", _FLUTTER_PUBSPEC)
        _write(root, "android/build.gradle", "buildscript {}\n")
        _write(root, "android/settings.gradle", "include ':app'\n")
        _write(root, "android/app/build.gradle", "apply plugin: 'com.android.application'\n")
        _write(root, "ios/Runner.xcodeproj/project.pbxproj", "// stub\n")

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert projects[0].ecosystem == "flutter"
        assert projects[0].root == "."
        assert "Flutter" in projects[0].frameworks
        assert "android" in projects[0].excluded_subpaths


def test_dart_only_project_detected_without_flutter_framework():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pubspec.yaml", _DART_ONLY_PUBSPEC)
        _write(root, "test/lib_test.dart", "void main() {}\n")

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert projects[0].ecosystem == "dart"
        assert "Flutter" not in projects[0].frameworks
        assert "Dart" in projects[0].languages


# ---------------------------------------------------------------------------
# Single-ecosystem repos (regression coverage for existing ecosystems)
# ---------------------------------------------------------------------------
def test_java_maven_single_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pom.xml", "<project></project>")

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert projects[0].ecosystem == "java-maven"
        assert projects[0].root == "."


def test_java_gradle_single_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "build.gradle", "plugins { id 'java' }")

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert projects[0].ecosystem == "java-gradle"


def test_react_project_detected_with_react_framework():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({
            "dependencies": {"react": "^18.0.0", "react-dom": "^18.0.0"},
            "scripts": {"test": "jest"},
        }))

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert projects[0].ecosystem == "node"
        assert projects[0].frameworks == ["React"]


def test_nextjs_project_detected_with_nextjs_framework():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({
            "dependencies": {"next": "^14.0.0", "react": "^18.0.0", "react-dom": "^18.0.0"},
            "scripts": {"build": "next build"},
        }))

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert "Next.js" in projects[0].frameworks


def test_vite_react_project_is_not_classified_as_nextjs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({
            "dependencies": {"react": "^18.0.0"},
            "devDependencies": {"vite": "^5.0.0"},
            "scripts": {"build": "vite build"},
        }))
        _write(root, "vite.config.js", "export default {}\n")

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert "Next.js" not in projects[0].frameworks
        assert "React" in projects[0].frameworks
        assert "Vite" in projects[0].frameworks


def test_generic_node_project_with_no_recognized_framework():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", json.dumps({"scripts": {"test": "mocha"}}))

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert projects[0].ecosystem == "node"
        assert projects[0].frameworks == []


def test_dart_only_project_via_analyzer_no_manifest_confusion():
    """A Dart-only project must never surface as Java/Gradle just because
    project_analyzer walks the whole tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pubspec.yaml", _DART_ONLY_PUBSPEC)

        projects = RepositoryAnalyzer.analyze(root)

        assert _roots(projects) == ["."]
        assert projects[0].ecosystem == "dart"


# ---------------------------------------------------------------------------
# Multi-project monorepos
# ---------------------------------------------------------------------------
def test_spring_backend_and_react_frontend_monorepo():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "backend/pom.xml", "<project><dependencies>"
               "<dependency><artifactId>spring-boot-starter-web</artifactId></dependency>"
               "</dependencies></project>")
        _write(root, "frontend/package.json", json.dumps({
            "dependencies": {"react": "^18.0.0"},
            "scripts": {"test": "jest"},
        }))

        projects = RepositoryAnalyzer.analyze(root)

        assert _roots(projects) == ["backend", "frontend"]
        by_root = {p.root: p for p in projects}
        assert by_root["backend"].ecosystem == "java-maven"
        assert "Spring Boot" in by_root["backend"].frameworks
        assert by_root["frontend"].ecosystem == "node"
        assert "React" in by_root["frontend"].frameworks


def test_flutter_app_and_android_module_are_still_one_project_in_monorepo():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "mobile/pubspec.yaml", _FLUTTER_PUBSPEC)
        _write(root, "mobile/android/build.gradle", "buildscript {}\n")
        _write(root, "server/pom.xml", "<project></project>")

        projects = RepositoryAnalyzer.analyze(root)

        assert _roots(projects) == ["mobile", "server"]
        by_root = {p.root: p for p in projects}
        assert by_root["mobile"].ecosystem == "flutter"
        assert by_root["server"].ecosystem == "java-maven"


def test_multiple_independent_projects_detected_separately():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "svc-go/go.mod", "module example.com/svc\n")
        _write(root, "svc-rust/Cargo.toml", "[package]\nname='svc'\n")
        _write(root, "svc-python/requirements.txt", "flask\n")

        projects = RepositoryAnalyzer.analyze(root)

        assert _roots(projects) == ["svc-go", "svc-python", "svc-rust"]
        by_root = {p.root: p for p in projects}
        assert by_root["svc-go"].ecosystem == "go"
        assert by_root["svc-rust"].ecosystem == "rust"
        assert by_root["svc-python"].ecosystem == "python"


def test_unsupported_ecosystem_returns_no_projects():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "README.md", "Just docs, no recognizable build system.\n")
        _write(root, "notes.txt", "nothing here\n")

        projects = RepositoryAnalyzer.analyze(root)

        assert projects == []


def test_bare_python_test_files_fallback_still_works_whole_repo():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        projects = RepositoryAnalyzer.analyze(root)

        assert len(projects) == 1
        assert projects[0].ecosystem == "python"
        assert projects[0].root == "."


# ---------------------------------------------------------------------------
# Task-aware project selection: repository evidence over task wording
# ---------------------------------------------------------------------------
def _monorepo_projects():
    from app.services.verification.project_analyzer import ProjectInfo

    return [
        ProjectInfo(root="backend", ecosystem="java-maven", languages=["Java"], frameworks=["Spring Boot"]),
        ProjectInfo(root="frontend", ecosystem="node", languages=["JavaScript"], frameworks=["React"]),
    ]


def test_select_relevant_projects_prefers_the_one_matching_task_and_evidence():
    projects = _monorepo_projects()
    matches = [{"file": "backend/src/main/java/UserRepository.java"}]

    selected = select_relevant_projects(projects, "Fix the Spring JDBC query", matches)

    assert len(selected) == 1
    assert selected[0].root == "backend"


def test_select_relevant_projects_react_task_selects_frontend():
    projects = _monorepo_projects()
    matches = [{"file": "frontend/src/components/ProductCard.jsx"}]

    selected = select_relevant_projects(projects, "Fix the React product card", matches)

    assert len(selected) == 1
    assert selected[0].root == "frontend"


def test_select_relevant_projects_returns_all_when_ambiguous():
    """No evidence favors one project over another -- never guess which one
    the task means; verify all of them instead."""
    projects = _monorepo_projects()

    selected = select_relevant_projects(projects, "Fix the login bug", [])

    assert {p.root for p in selected} == {"backend", "frontend"}


def test_select_relevant_projects_task_wording_alone_does_not_win_without_evidence():
    """Repository evidence (where code matches actually live) outweighs a
    project merely being *named* in the task description with no supporting
    keyword matches under its root."""
    projects = _monorepo_projects()
    # Matches live under frontend, but the task text mentions "spring" --
    # code evidence should still decide.
    matches = [
        {"file": "frontend/src/App.jsx"},
        {"file": "frontend/src/App.jsx"},
    ]

    selected = select_relevant_projects(projects, "something about spring cleaning the UI", matches)

    assert len(selected) == 1
    assert selected[0].root == "frontend"


def _nextjs_and_vite_projects():
    from app.services.verification.project_analyzer import ProjectInfo

    return [
        ProjectInfo(root="web", ecosystem="node", languages=["JavaScript"], frameworks=["Next.js", "React"]),
        ProjectInfo(root="admin", ecosystem="node", languages=["JavaScript"], frameworks=["React", "Vite"]),
    ]


def test_user_says_react_but_repo_evidence_points_to_the_nextjs_project():
    """The user's wording ('React') matches both candidates equally (Next.js
    apps use React too) -- code-search evidence, not the task's framework
    name, must decide which project is actually relevant."""
    projects = _nextjs_and_vite_projects()
    matches = [{"file": "web/components/ProductCard.jsx"}]

    selected = select_relevant_projects(projects, "Fix the React component", matches)

    assert len(selected) == 1
    assert selected[0].root == "web"


def test_user_says_nextjs_but_repo_evidence_points_to_the_vite_react_project():
    """The task claims 'Next.js' but the code evidence lives in the plain
    Vite+React project -- the task's framework name must never override
    repository evidence."""
    projects = _nextjs_and_vite_projects()
    matches = [{"file": "admin/src/ProductCard.jsx"}]

    selected = select_relevant_projects(projects, "Fix the Next.js component", matches)

    assert len(selected) == 1
    assert selected[0].root == "admin"
