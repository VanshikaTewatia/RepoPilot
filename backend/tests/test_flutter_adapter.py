"""Unit tests for the Flutter/Dart verification adapters."""

import tempfile
from pathlib import Path

from app.services.verification.adapters import DartAdapter, FlutterAdapter
from app.services.verification.detector import ProjectDetector


def _write(root: Path, rel_path: str, content: str = "") -> None:
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


_FLUTTER_PUBSPEC = """\
name: my_app
description: A sample app.

environment:
  sdk: '>=2.19.0 <3.0.0'

dependencies:
  flutter:
    sdk: flutter
  cupertino_icons: ^1.0.2

dev_dependencies:
  flutter_test:
    sdk: flutter
"""

_DART_ONLY_PUBSPEC = """\
name: my_dart_lib
description: A pure Dart package.

environment:
  sdk: '>=2.19.0 <3.0.0'

dependencies:
  path: ^1.8.0

dev_dependencies:
  test: ^1.24.0
"""


# ---------------------------------------------------------------------------
# Detection: Flutter vs Dart vs Java/Gradle
# ---------------------------------------------------------------------------
class TestFlutterVsDartDetection:
    def test_flutter_project_detected_via_pubspec_sdk_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _FLUTTER_PUBSPEC)

            assert FlutterAdapter.detect(root) is True
            assert DartAdapter.detect(root) is False

    def test_dart_only_project_detected_when_no_flutter_sdk_dependency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _DART_ONLY_PUBSPEC)

            assert FlutterAdapter.detect(root) is False
            assert DartAdapter.detect(root) is True

    def test_project_detector_resolves_flutter_project_to_flutter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _FLUTTER_PUBSPEC)

            result = ProjectDetector.detect(root)

            assert result.ecosystem == "flutter"
            assert isinstance(result.adapter, FlutterAdapter)

    def test_project_detector_resolves_dart_only_project_to_dart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _DART_ONLY_PUBSPEC)

            result = ProjectDetector.detect(root)

            assert result.ecosystem == "dart"

    def test_flutter_project_with_nested_android_gradle_still_resolves_to_flutter(self):
        """The critical regression case: a Flutter app's generated
        android/build.gradle must never cause the *root* workspace to be
        misclassified as Java/Gradle when detected directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _FLUTTER_PUBSPEC)
            _write(root, "android/build.gradle", "buildscript { repositories { google() } }\n")
            _write(root, "android/settings.gradle", "include ':app'\n")
            _write(root, "ios/Runner.xcodeproj/project.pbxproj", "// stub\n")

            result = ProjectDetector.detect(root)

            assert result.ecosystem == "flutter"
            assert isinstance(result.adapter, FlutterAdapter)


# ---------------------------------------------------------------------------
# FlutterAdapter command selection
# ---------------------------------------------------------------------------
class TestFlutterAdapterCommands:
    def test_install_command_is_pub_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert FlutterAdapter().install_command(Path(tmpdir)) == ["flutter", "pub", "get"]

    def test_test_command_uses_flutter_test_when_tests_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _FLUTTER_PUBSPEC)
            _write(root, "test/widget_test.dart", "void main() {}\n")

            assert FlutterAdapter().test_command(root) == ["flutter", "test"]

    def test_test_command_falls_back_to_analyze_when_no_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _FLUTTER_PUBSPEC)

            assert FlutterAdapter().test_command(root) == ["flutter", "analyze"]

    def test_test_command_honors_explicit_test_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "test/widget_test.dart", "void main() {}\n")

            cmd = FlutterAdapter().test_command(root, test_path="test/widget_test.dart")
            assert cmd == ["flutter", "test", "test/widget_test.dart"]

    def test_parse_output_extracts_passed_and_failed_from_final_summary_line(self):
        output = (
            "00:01 +1: loading test/widget_test.dart\n"
            "00:02 +1 -1: some test failed\n"
            "00:03 +3 -1: some tests failed\n"
        )
        counts = FlutterAdapter().parse_output(output, 1)
        assert counts == {"passed": 3, "failed": 1}

    def test_parse_output_all_passed(self):
        output = "00:02 +5: All tests passed!\n"
        counts = FlutterAdapter().parse_output(output, 0)
        assert counts == {"passed": 5, "failed": 0}

    def test_parse_output_analyze_no_issues(self):
        counts = FlutterAdapter().parse_output("Analyzing my_app...\nNo issues found!\n", 0)
        assert counts == {"passed": 1, "failed": 0}


# ---------------------------------------------------------------------------
# DartAdapter command selection
# ---------------------------------------------------------------------------
class TestDartAdapterCommands:
    def test_install_command_is_dart_pub_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert DartAdapter().install_command(Path(tmpdir)) == ["dart", "pub", "get"]

    def test_test_command_uses_dart_test_when_tests_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _DART_ONLY_PUBSPEC)
            _write(root, "test/lib_test.dart", "void main() {}\n")

            assert DartAdapter().test_command(root) == ["dart", "test"]

    def test_test_command_falls_back_to_analyze(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write(root, "pubspec.yaml", _DART_ONLY_PUBSPEC)

            assert DartAdapter().test_command(root) == ["dart", "analyze"]


# ---------------------------------------------------------------------------
# Missing Flutter/Dart toolchain must be reported clearly, never silently
# treated as a pass or a genuine failure.
# ---------------------------------------------------------------------------
def test_flutter_missing_toolchain_reported_as_unavailable():
    from unittest.mock import patch

    from app.services.verification.engine import VerificationEngine

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write(root, "pubspec.yaml", _FLUTTER_PUBSPEC)

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("[WinError 2] The system cannot find the file specified: 'flutter'")

        engine = VerificationEngine()
        with patch.object(type(engine._docker_runner), "is_docker_available", False), \
             patch("app.services.verification.engine.subprocess.run", side_effect=fake_run):
            result = engine.verify(root)

        assert result["ecosystem"] == "flutter"
        assert result["available"] is False
        assert result["success"] is False
        assert "flutter" in (result["detail"] or "").lower() or "toolchain" in (result["output"] or "").lower()
