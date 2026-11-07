import os

import pytest
from fastapi.testclient import TestClient

from api.main import app, _request_times


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_api_key(client):
    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://example.com"},
    )

    assert response.status_code == 401


def test_invalid_api_key(client):
    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://example.com"},
        headers={"X-API-Key": "invalid-test-key"},
    )

    assert response.status_code == 401


def test_valid_url_analysis(client, auth_headers):
    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://example.com"},
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "analyzed"
    assert "findings" in data
    assert "ml_analysis" in data
    assert "phishing_class_score" in data["ml_analysis"]


def test_invalid_url(client, auth_headers):
    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://bad host.com"},
        headers=auth_headers,
    )

    assert response.status_code == 400


def test_missing_url_field(client, auth_headers):
    response = client.post(
        "/api/v1/analyze/url",
        json={},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_model_failure_returns_safe_503(client, auth_headers, monkeypatch):
    """Inference failures must not expose internal exception details."""

    def broken_prediction(url):
        raise RuntimeError("Internal model failure: secret filesystem path")

    monkeypatch.setattr("api.main.predict_url", broken_prediction)

    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://example.com"},
        headers=auth_headers,
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Analysis service temporarily unavailable."
    }
    assert "secret filesystem path" not in response.text
