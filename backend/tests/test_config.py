"""Unit tests for Pydantic Settings configuration module."""

from pathlib import Path
import os
import pytest

from app.core.config import Settings, get_settings


def test_default_settings(monkeypatch):
    """Test default values for Settings."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = Settings()
    assert settings.app_name == "RepoPilot"
    assert settings.environment == "development"
    assert settings.is_development is True
    assert settings.is_production is False
    assert settings.gemini_model_name == "gemini-3.6-flash"
    assert settings.gemini_embedding_model == "gemini-embedding-2"
    assert settings.gemini_embedding_dimension == 3072
    assert "postgresql+asyncpg://" in settings.database_url
    assert settings.sync_database_url.startswith("postgresql://")
    assert settings.sandbox_timeout_seconds == 45
    assert settings.agent_max_iterations == 3
    assert isinstance(settings.workspace_dir, Path)


def test_cors_origins_parsing():
    """Test various formats for CORS_ORIGINS."""
    # JSON list string
    s1 = Settings(cors_origins='["http://localhost:3000", "https://app.repopilot.dev"]')
    assert s1.cors_origins == ["http://localhost:3000", "https://app.repopilot.dev"]

    # Comma-separated string
    s2 = Settings(cors_origins="http://localhost:3000, https://app.repopilot.dev")
    assert s2.cors_origins == ["http://localhost:3000", "https://app.repopilot.dev"]

    # Native list
    s3 = Settings(cors_origins=["http://localhost:3000"])
    assert s3.cors_origins == ["http://localhost:3000"]

    # Empty string
    s4 = Settings(cors_origins="")
    assert s4.cors_origins == []


def test_environment_override(monkeypatch):
    """Test loading settings from environment variables."""
    monkeypatch.setenv("APP_NAME", "CustomRepoPilot")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("SANDBOX_TIMEOUT_SECONDS", "60")

    settings = Settings()
    assert settings.app_name == "CustomRepoPilot"
    assert settings.environment == "production"
    assert settings.is_production is True
    assert settings.gemini_api_key == "test-gemini-key"
    assert settings.port == 9000
    assert settings.sandbox_timeout_seconds == 60


def test_workspace_dir_resolution():
    """Test that workspace_dir is cast to a Path object."""
    settings = Settings(workspace_dir="./custom/workspace")
    assert isinstance(settings.workspace_dir, Path)
    assert str(settings.workspace_dir).replace("\\", "/") == "custom/workspace"


def test_get_settings_cached():
    """Test that get_settings caches the instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
