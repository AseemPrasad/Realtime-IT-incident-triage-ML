from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from incident_triage.api.main import app, state
from incident_triage.fusion import FusionResult, fuse_signals
from incident_triage.stream_buffer import StreamingWindowBuffer


def test_streaming_window_buffer():
    buf = StreamingWindowBuffer(maxlen=5)
    entity = "server-01"
    for v in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]:
        win = buf.record_and_get_window(entity, v)
    assert len(win) == 5
    assert win == [20.0, 30.0, 40.0, 50.0, 60.0]


def test_fusion_with_custom_meta_model():
    class DummyLearnedFusionModel:
        def predict_fusion(self, telemetry_risk, log_severity=None, log_confidence=0.0, config=None, log_domain=None):
            return FusionResult(
                incident_score=0.99,
                priority="P1",
                route_to="meta-model-bridge",
                explanation="Predicted by learned meta-model",
            )

    model = DummyLearnedFusionModel()
    result = fuse_signals(
        telemetry_risk=0.1,
        log_severity="INFO",
        log_confidence=0.1,
        model=model,
    )
    assert result.incident_score == 0.99
    assert result.priority == "P1"
    assert result.route_to == "meta-model-bridge"


def test_api_streaming_single_value_support():
    client = TestClient(app)
    # Clear buffer for test isolation
    state.stream_buffer.clear()

    # Sending a single streaming metric point rather than an entire array
    response = client.post(
        "/api/v1/triage",
        json={
            "telemetry": {"value": 45.2},
            "asset_id": "test-cluster-host-99",
            "log_content": "PacketResponder terminating due to socket exception",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "incident_score" in data
    assert "priority" in data
    assert data["telemetry"]["risk_score"] >= 0.0
