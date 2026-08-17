"""
CI smoke tests (roadmap ops #6): verify the FastAPI app imports and its core
endpoints respond, without triggering the startup event (no network needed).
Also covers the institutional JWT auth gate (roadmap #4).
"""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import main  # noqa: F401  (module-level engines must construct cleanly)
    return TestClient(main.app)


def test_home_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "QUANT-PORTAL" in resp.text


def test_telegram_mini_app_served(client):
    resp = client.get("/telegram")
    assert resp.status_code == 200
    assert "Q-Bot" in resp.text or "Telegram" in resp.text


def test_status_endpoint(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "mode" in body
    assert body["mode"] in ("DEMO", "REAL")


def test_telemetry_endpoint(client):
    resp = client.get("/api/telemetry")
    assert resp.status_code == 200
    body = resp.json()
    assert "last_price" in body
    assert "strategy_weights" in body  # meta-engine live weights exposed
    assert "active_models" in body     # LOT 46 status exposed


def test_prometheus_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "quant_uptime_seconds" in resp.text


def test_auth_gate_blocks_unauthenticated_actions(client, monkeypatch):
    # Institutional rule: with AUTH_ENABLED, state-changing endpoints require a JWT
    monkeypatch.setenv("AUTH_ENABLED", "true")
    resp = client.post("/api/toggle-bot", json={"is_running": False})
    assert resp.status_code in (401, 403), f"expected auth rejection, got {resp.status_code}"


def test_auth_login_flow(client, monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("ADMIN_USER", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "TestPass123")

    # wrong password -> 401
    bad = client.post("/api/login", json={"username": "admin", "password": "wrong"})
    assert bad.status_code == 401

    # correct credentials -> token
    good = client.post("/api/login", json={"username": "admin", "password": "TestPass123"})
    assert good.status_code == 200
    token = good.json()["token"]
    assert token

    # token unlocks the protected endpoint
    resp = client.post(
        "/api/toggle-bot",
        json={"is_running": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
