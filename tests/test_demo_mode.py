import pytest
from fastapi.testclient import TestClient

import api.main as main
from api.main import RATE_LIMIT, app


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("CYBER_GUARDIAN_API_KEY", raising=False)
    monkeypatch.setattr(main, "DEMO_DAILY_LIMIT", 1000)
    monkeypatch.setattr(main, "_demo_used", 0)
    with TestClient(app) as c:
        yield c


URL = {"url": "https://example.com"}


def test_config_reports_demo_flag(client, demo):
    assert demo.get("/api/v1/config").json()["demo"] is True


def test_config_is_off_by_default(client, monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert client.get("/api/v1/config").json()["demo"] is False


def test_demo_needs_no_api_key(demo):
    assert demo.post("/api/v1/analyze/url", json=URL).status_code == 200


def test_key_is_still_required_when_demo_is_off(client):
    assert client.post("/api/v1/analyze/url", json=URL).status_code == 401


def test_each_visitor_has_their_own_rate_limit(demo):
    a = {"X-Forwarded-For": "1.1.1.1"}
    b = {"X-Forwarded-For": "2.2.2.2"}
    for _ in range(RATE_LIMIT):
        assert demo.post("/api/v1/analyze/url", json=URL, headers=a).status_code == 200
    assert demo.post("/api/v1/analyze/url", json=URL, headers=a).status_code == 429
    assert demo.post("/api/v1/analyze/url", json=URL, headers=b).status_code == 200


def test_shared_daily_budget_stops_everyone(demo, monkeypatch):
    monkeypatch.setattr(main, "DEMO_DAILY_LIMIT", 2)
    monkeypatch.setattr(main, "_demo_used", 0)
    for i in range(2):
        r = demo.post("/api/v1/analyze/url", json=URL, headers={"X-Forwarded-For": f"9.9.9.{i}"})
        assert r.status_code == 200
    r = demo.post("/api/v1/analyze/url", json=URL, headers={"X-Forwarded-For": "9.9.9.9"})
    assert r.status_code == 429 and "limit for today" in r.json()["detail"]
