from __future__ import annotations

import pytest
import numpy as np
from fastapi.testclient import TestClient

from incident_triage.api.main import app, state
from incident_triage.drift import calculate_psi, TelemetryDriftMonitor
from incident_triage.logs import generate_structured_log_analysis, LogAnalysis
from incident_triage.stream_buffer import InMemoryStreamingBuffer, create_stream_buffer


def test_psi_identical_distributions():
    # Exactly identical data should yield a PSI near 0.0
    baseline = np.random.normal(loc=50.0, scale=10.0, size=200)
    psi = calculate_psi(baseline, baseline)
    assert psi < 0.05


def test_psi_shifted_distribution():
    # Heavy shift from mean 50 to mean 85 should trigger significant drift (PSI > 0.25)
    baseline = np.random.normal(loc=50.0, scale=5.0, size=200)
    shifted = np.random.normal(loc=85.0, scale=5.0, size=200)
    psi = calculate_psi(baseline, shifted)
    assert psi >= 0.25


def test_drift_monitor_assess_drift():
    # Continuous telemetry samples from normal operating distribution
    np.random.seed(42)
    baseline = np.random.normal(loc=50.0, scale=5.0, size=500).tolist()
    monitor = TelemetryDriftMonitor(baseline_samples=baseline)

    # Healthy incoming samples from same distribution
    healthy_samples = np.random.normal(loc=50.0, scale=5.0, size=300).tolist()
    assessment = monitor.assess_drift(healthy_samples)
    assert assessment["status"] == "HEALTHY"
    assert assessment["drift_detected"] is False

    # Shifted incoming samples (mean jumps from 50 to 80)
    shifted_samples = np.random.normal(loc=80.0, scale=5.0, size=200).tolist()
    drifted_assessment = monitor.assess_drift(shifted_samples)
    assert drifted_assessment["status"] in {"MODERATE_DRIFT", "CRITICAL_DRIFT"}
    assert drifted_assessment["drift_detected"] is True


def test_api_health_drift_endpoints_direct():
    from incident_triage.api.main import health, drift_status
    h = health()
    assert h["status"] == "ok"
    assert "buffer_backend" in h

    d = drift_status()
    assert "status" in d
    assert "psi" in d

