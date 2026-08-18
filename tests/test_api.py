"""
Integration tests for FastAPI serving endpoints.

Run with:
    python -m pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "chunks_indexed" in data
    assert data["chunks_file_present"] is True


def test_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "retrieval_benchmark" in data


def test_query_validation_error():
    # Empty query should fail validation with 422
    response = client.post("/api/query", json={"query": ""})
    assert response.status_code == 422
