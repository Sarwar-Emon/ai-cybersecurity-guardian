import pytest
from fastapi.testclient import TestClient

from api.main import app, _request_times


@pytest.fixture(autouse=True)
def isolated_test_environment(monkeypatch):
    """Use a temporary API key and reset rate-limit state."""
    monkeypatch.setenv(
        "CYBER_GUARDIAN_API_KEY",
        "guardian-test-key-not-for-production",
    )

    _request_times.clear()
    yield
    _request_times.clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {
        "X-API-Key": "guardian-test-key-not-for-production"
    }
