import os

import pytest
from fastapi.testclient import TestClient

from api.main import app, _request_times
from src.ml_detector import predict_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "https://www.google.com",
        "http://paypal-login.example.com/verify",
    ],
)
def test_api_matches_ml_engine(client, url):
    key = os.environ.get("CYBER_GUARDIAN_API_KEY")
    assert key, "Development API key missing."

    response = client.post(
        "/api/v1/analyze/url",
        json={"url": url},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200, response.text

    api_result = response.json()["ml_analysis"]
    direct_result = predict_url(url)

    assert api_result["phishing_class_score"] == pytest.approx(
        direct_result["phishing_score"],
        abs=1e-10,
    )

    expected_classification = (
        "flagged"
        if direct_result["phishing_score"]
        >= direct_result["threshold"]
        else "not_flagged"
    )

    assert api_result["classification"] == expected_classification
    assert api_result["threshold"] == direct_result["threshold"]
    assert api_result["model"] == direct_result["model"]


def test_ml_score_is_bounded(client):
    key = os.environ.get("CYBER_GUARDIAN_API_KEY")
    assert key

    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://example.com"},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200

    score = response.json()["ml_analysis"]["phishing_class_score"]

    assert 0.0 <= score <= 1.0


def test_ml_disclaimer_present(client):
    key = os.environ.get("CYBER_GUARDIAN_API_KEY")
    assert key

    response = client.post(
        "/api/v1/analyze/url",
        json={"url": "https://example.com"},
        headers={"X-API-Key": key},
    )

    assert response.status_code == 200
    assert "not a calibrated" in response.json()["disclaimer"]
