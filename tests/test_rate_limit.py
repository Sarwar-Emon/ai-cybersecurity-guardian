import os

import pytest
from fastapi.testclient import TestClient

from api.main import app, _request_times, RATE_LIMIT


def test_rate_limit_enforced(client):
    key = os.environ.get("CYBER_GUARDIAN_API_KEY")
    assert key, "Development API key missing."

    headers = {"X-API-Key": key}
    endpoint = "/api/v1/analyze/url"
    payload = {"url": "https://example.com"}

    for request_number in range(RATE_LIMIT):
        response = client.post(
            endpoint,
            json=payload,
            headers=headers,
        )

        assert response.status_code == 200, (
            f"Request {request_number + 1} failed: "
            f"{response.text}"
        )

    blocked = client.post(
        endpoint,
        json=payload,
        headers=headers,
    )

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_invalid_key_rejected(client):
    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://example.com"},
        headers={"X-API-Key": "incorrect-key"},
    )

    assert response.status_code == 401


def test_health_not_rate_limited(client):
    for _ in range(RATE_LIMIT + 2):
        response = client.get("/health")
        assert response.status_code == 200
