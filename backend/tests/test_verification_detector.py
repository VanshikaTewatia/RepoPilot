"""Unit tests for ProjectDetector: manifest scanning and deterministic precedence."""

import tempfile
from pathlib import Path

from app.services.verification.adapters import (
    DotnetAdapter,
    GoAdapter,
    JavaGradleAdapter,
    JavaMavenAdapter,
    NodeAdapter,
    PythonAdapter,
    RustAdapter,
)
from app.services.verification.detector import ProjectDetector


def _write(root: Path, rel_path: str, content: str = "") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Single-ecosystem detection
# ---------------------------------------------------------------------------
def test_detects_python_via_pyproject_toml():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "python"
        assert isinstance(result.adapter, PythonAdapter)
        assert "pyproject.toml" in result.manifests_found


def test_detects_python_via_requirements_txt():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "requirements.txt", "flask\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "python"
        assert "requirements.txt" in result.manifests_found


def test_detects_python_via_setup_py():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "setup.py", "from setuptools import setup\nsetup()\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "python"
        assert "setup.py" in result.manifests_found


def test_detects_python_via_test_files_when_no_manifest_present():
    """Backward-compat fallback: a bare workspace with pytest-style test files
    but no packaging manifest still resolves to Python (RepoPilot's original
    unconditional default), instead of being reported unknown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "test_math.py", "def test_add(): assert 1 + 1 == 2\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "python"
        assert result.manifests_found == []  # matched via fallback, not a manifest


def test_detects_node_via_package_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", '{"name": "app", "scripts": {"test": "jest"}}')

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "node"
        assert isinstance(result.adapter, NodeAdapter)
        assert "package.json" in result.manifests_found


def test_detects_java_maven_via_pom_xml():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pom.xml", "<project></project>")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "java-maven"
        assert isinstance(result.adapter, JavaMavenAdapter)


def test_detects_java_gradle_via_build_gradle():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "build.gradle", "plugins { id 'java' }")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "java-gradle"
        assert isinstance(result.adapter, JavaGradleAdapter)


def test_detects_java_gradle_via_build_gradle_kts():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "build.gradle.kts", "plugins { java }")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "java-gradle"


def test_detects_go_via_go_mod():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n\ngo 1.21\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "go"
        assert isinstance(result.adapter, GoAdapter)


def test_detects_rust_via_cargo_toml():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "Cargo.toml", "[package]\nname = 'app'\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "rust"
        assert isinstance(result.adapter, RustAdapter)


def test_detects_dotnet_via_sln():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "App.sln", "Microsoft Visual Studio Solution File\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "dotnet"
        assert isinstance(result.adapter, DotnetAdapter)
        assert "App.sln" in result.manifests_found


def test_detects_dotnet_via_root_csproj():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "App.csproj", "<Project Sdk='Microsoft.NET.Sdk'></Project>")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "dotnet"


def test_detects_dotnet_via_nested_csproj():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "src/App/App.csproj", "<Project Sdk='Microsoft.NET.Sdk'></Project>")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "dotnet"
        assert any("App.csproj" in m for m in result.manifests_found)


# ---------------------------------------------------------------------------
# Unknown ecosystem
# ---------------------------------------------------------------------------
def test_unknown_ecosystem_when_no_manifest_and_no_python_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "README.md", "# Just docs, no recognizable project\n")

        result = ProjectDetector.detect(root)

        assert result.adapter is None
        assert result.ecosystem is None
        assert result.manifests_found == []
        assert "package.json" in result.manifests_scanned
        assert "pyproject.toml" in result.manifests_scanned
        assert "go.mod" in result.manifests_scanned


def test_unknown_ecosystem_for_completely_empty_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        result = ProjectDetector.detect(root)

        assert result.adapter is None
        assert result.ecosystem is None


# ---------------------------------------------------------------------------
# Deterministic precedence when multiple manifests coexist
# ---------------------------------------------------------------------------
def test_node_takes_precedence_over_python_when_both_present():
    """The reported bug scenario: a JS/TS repo that also happens to carry an
    auxiliary Python tool manifest must still resolve to Node, not Python."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "package.json", '{"name": "app", "scripts": {"build": "next build"}}')
        _write(root, "requirements.txt", "mkdocs\n")  # e.g. a docs toolchain

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "node"


def test_go_takes_precedence_over_node_and_python():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")
        _write(root, "package.json", '{"name": "tooling"}')
        _write(root, "pyproject.toml", "[project]\nname='x'\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "go"


def test_rust_takes_precedence_over_dotnet_java_node_python():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "Cargo.toml", "[package]\nname='app'\n")
        _write(root, "App.sln", "solution\n")
        _write(root, "pom.xml", "<project></project>")
        _write(root, "package.json", '{"name": "tooling"}')

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "rust"


def test_dotnet_takes_precedence_over_java_and_node():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "App.sln", "solution\n")
        _write(root, "pom.xml", "<project></project>")
        _write(root, "package.json", '{"name": "tooling"}')

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "dotnet"


def test_maven_takes_precedence_over_gradle_and_node():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pom.xml", "<project></project>")
        _write(root, "build.gradle", "plugins { id 'java' }")
        _write(root, "package.json", '{"name": "tooling"}')

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "java-maven"


def test_gradle_takes_precedence_over_node_and_python():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "build.gradle.kts", "plugins { java }")
        _write(root, "package.json", '{"name": "tooling"}')
        _write(root, "requirements.txt", "sphinx\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "java-gradle"


def test_python_is_lowest_precedence_terminal_fallback():
    """With no other ecosystem manifest present, Python (via manifest or the
    test-file fallback) is always the final resolution -- never 'unknown'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pyproject.toml", "[project]\nname='x'\n")

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "python"


def test_all_manifests_found_reports_every_ecosystem_present():
    """Even the non-winning ecosystems' manifests are reported for visibility,
    while `manifests_found` narrows to just the winning ecosystem's."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "go.mod", "module example.com/app\n")
        _write(root, "package.json", '{"name": "tooling"}')

        result = ProjectDetector.detect(root)

        assert result.ecosystem == "go"
        assert result.manifests_found == ["go.mod"]
        assert "package.json" in result.all_manifests_found
        assert "go.mod" in result.all_manifests_found
