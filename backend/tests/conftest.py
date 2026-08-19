"""Pytest fixtures and configuration."""

import os
import pytest
from fastapi.testclient import TestClient

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["GEMINI_API_KEY"] = "test_gemini_key_123"

from app.main import app
from app.core.config import get_settings

# Force reload of settings with test environment
get_settings.cache_clear()


@pytest.fixture
def client():
    """Synchronous test client for API endpoint testing."""
    with TestClient(app) as test_client:
        yield test_client
