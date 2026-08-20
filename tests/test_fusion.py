from incident_triage.fusion import fuse_signals


def test_fusion_prioritizes_high_risk_combination():
    result = fuse_signals(telemetry_risk=0.92, log_severity="CRITICAL", log_confidence=0.95)
    assert result.incident_score >= 0.85
    assert result.priority == "P1"

