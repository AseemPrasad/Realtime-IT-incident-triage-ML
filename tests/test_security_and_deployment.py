from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

from incident_triage.api.main import app


def test_api_auth_disabled_when_env_not_set(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    client = TestClient(app)
    response = client.post(
        "/api/v1/triage",
        json={
            "telemetry": {"values": [35.0, 36.0, 37.0]},
            "log_content": "Routine block verification",
        },
    )
    assert response.status_code == 200


def test_api_auth_enforced_when_env_is_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-incident-key-123")
    client = TestClient(app)

    # 1. Missing header -> 401 Unauthorized
    unauth_resp = client.post(
        "/api/v1/triage",
        json={"telemetry": {"values": [35.0, 36.0, 37.0]}},
    )
    assert unauth_resp.status_code == 401

    # 2. Wrong header -> 401 Unauthorized
    wrong_key_resp = client.post(
        "/api/v1/triage",
        headers={"X-API-Key": "wrong-key"},
        json={"telemetry": {"values": [35.0, 36.0, 37.0]}},
    )
    assert wrong_key_resp.status_code == 401

    # 3. Valid header -> 200 OK
    valid_resp = client.post(
        "/api/v1/triage",
        headers={"X-API-Key": "secret-incident-key-123"},
        json={"telemetry": {"values": [35.0, 36.0, 37.0]}},
    )
    assert valid_resp.status_code == 200


def test_payload_length_dos_guardrail():
    client = TestClient(app)
    # Exceeding 10,000 characters in log_content triggers 422 Unprocessable Entity
    oversized_log = "FATAL ERROR " * 1000  # 12,000 characters
    response = client.post(
        "/api/v1/triage",
        json={
            "telemetry": {"values": [50.0]},
            "log_content": oversized_log,
        },
    )
    assert response.status_code == 422
