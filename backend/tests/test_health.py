"""Unit tests for GET /health endpoint."""

from fastapi.testclient import TestClient


def test_get_health(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["app"] == "RepoPilot"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
