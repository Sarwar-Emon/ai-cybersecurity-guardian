import os

from fastapi.testclient import TestClient

from api.main import app, _request_times


client = TestClient(app)


def setup_function():
    os.environ["CYBER_GUARDIAN_API_KEY"] = "test-email-api-key"
    _request_times.clear()


def test_valid_email_analysis():
    response = client.post(
        "/api/v1/analyze/email",
        headers={"X-API-Key": "test-email-api-key"},
        json={
            "email_text": (
                "From: PayPal Support <support@paypa1.com>\n"
                "Subject: Urgent verification\n\n"
                "Your account has been suspended."
            )
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "analyzed"
    assert data["level"] == "High Risk"
    assert "findings" in data
    assert "sender_analysis" in data


def test_missing_api_key():
    response = client.post(
        "/api/v1/analyze/email",
        json={"email_text": "Test message"},
    )
    assert response.status_code == 401


def test_empty_email_rejected():
    response = client.post(
        "/api/v1/analyze/email",
        headers={"X-API-Key": "test-email-api-key"},
        json={"email_text": ""},
    )
    assert response.status_code == 422

def test_email_analyzes_embedded_url_separately(monkeypatch):
    """Email and embedded URL analyses remain separate."""

    url = "http" + chr(58) + chr(47) * 2 + "paypal-login.example.com/verify-account"

    def fake_predict_url(received_url):
        assert received_url == url
        return {
            "phishing_score": 0.91,
            "threshold": 0.5,
            "model": "test-random-forest",
        }

    monkeypatch.setattr("api.main.predict_url", fake_predict_url)

    response = client.post(
        "/api/v1/analyze/email",
        headers={"X-API-Key": "test-email-api-key"},
        json={
            "email_text": (
                "URGENT: Your account has been suspended. "
                "Verify your identity at " + url
            )
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["score"] == 60
    assert data["level"] == "High Risk"

    embedded = data["embedded_url_analysis"]
    assert len(embedded) == 1

    item = embedded[0]
    assert item["url"] == url
    assert item["status"] == "analyzed"
    assert isinstance(item["findings"], list)
    assert item["ml_analysis"]["classification"] == "flagged"
    assert item["ml_analysis"]["phishing_class_score"] == 0.91
    assert item["ml_analysis"]["threshold"] == 0.5
