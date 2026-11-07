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
