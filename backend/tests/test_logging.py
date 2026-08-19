"""Unit tests for structured logging."""

import json
import logging
from app.core.logging import JSONFormatter, StandardFormatter


def test_json_formatter():
    """Test that JSONFormatter outputs valid JSON with expected fields."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test message %s",
        args=("foo",),
        exc_info=None,
    )
    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Test message foo"
    assert parsed["line"] == 42
    assert "timestamp" in parsed


def test_standard_formatter():
    """Test standard development formatter."""
    formatter = StandardFormatter()
    record = logging.LogRecord(
        name="test_dev",
        level=logging.WARNING,
        pathname="test.py",
        lineno=10,
        msg="Dev warning",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    assert "WARNING" in output
    assert "test_dev:10" in output
    assert "Dev warning" in output
