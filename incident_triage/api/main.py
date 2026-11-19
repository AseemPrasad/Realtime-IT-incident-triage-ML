from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from ..artifacts import load_joblib
from ..config import ARTIFACT_DIR, DEFAULT_FUSION_CONFIG
from ..drift import TelemetryDriftMonitor
from ..fusion import fuse_signals
from ..logs import HeuristicLogAnalyzer
from ..schemas import IncidentRequest, LogAnalysis, TelemetryRisk, TriageResponse
from ..stream_buffer import StreamBufferProtocol, create_stream_buffer
from ..tabular_model import TelemetryAnomalyModel
from ..telemetry import TelemetryFeatureExtractor


app = FastAPI(title="Real-Time IT Incident Triage Engine", version="0.1.0")

# Security & Auth guardrails:
# If API_KEY is set in environment, enforces X-API-Key header.
# If not set (local dev / tests), access is automatically permitted with zero breaking changes.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    expected_key = os.getenv("API_KEY")
    if expected_key:
        if not api_key or api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key. Provide a valid 'X-API-Key' header.",
            )
    return api_key



class RuntimeState:
    telemetry_model: Optional[TelemetryAnomalyModel] = None
    log_analyzer: HeuristicLogAnalyzer = HeuristicLogAnalyzer()
    extractor: TelemetryFeatureExtractor = TelemetryFeatureExtractor()
    stream_buffer: StreamBufferProtocol = create_stream_buffer(maxlen=30)
    drift_monitor: TelemetryDriftMonitor = TelemetryDriftMonitor()


state = RuntimeState()


@app.on_event("startup")
def load_artifacts() -> None:
    model_path = ARTIFACT_DIR / "tabular_model.joblib"
    if model_path.exists():
        try:
            state.telemetry_model = load_joblib(model_path)
        except Exception:
            state.telemetry_model = None


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "telemetry_model": "loaded" if state.telemetry_model else "heuristic",
        "buffer_backend": type(state.stream_buffer).__name__,
    }


@app.get("/health/drift")
def drift_status() -> dict[str, object]:
    # Check drift on the default buffer if values exist
    recent_vals = state.stream_buffer.get_window("default")
    return state.drift_monitor.assess_drift(recent_vals)


@app.post("/api/v1/triage", response_model=TriageResponse, dependencies=[Depends(verify_api_key)])
def triage(request: IncidentRequest) -> TriageResponse:
    telemetry_values = request.telemetry.values
    if not telemetry_values:
        if request.telemetry.value is not None:
            entity_key = request.asset_id or request.source or "default"
            telemetry_values = state.stream_buffer.record_and_get_window(entity_key, request.telemetry.value)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either telemetry.values (historical list) or telemetry.value (streaming single point) must be provided",
            )

    if state.telemetry_model is not None:
        probability, is_anomaly, feature_snapshot = state.telemetry_model.predict_from_values(telemetry_values)
        threshold = state.telemetry_model.threshold
    else:
        probability = state.extractor.heuristic_risk(telemetry_values)
        is_anomaly = probability >= DEFAULT_FUSION_CONFIG.p3_threshold
        feature_snapshot = state.extractor.extract_from_values(telemetry_values).iloc[0].to_dict()
        threshold = DEFAULT_FUSION_CONFIG.p3_threshold

    telemetry_risk = TelemetryRisk(
        risk_score=float(probability),
        threshold=float(threshold),
        is_anomaly=bool(is_anomaly),
        probability=float(probability),
        feature_snapshot={k: float(v) for k, v in feature_snapshot.items()},
    )

    log_analysis = state.log_analyzer.analyze(request.log_content)
    log_domain = log_analysis.error_domain if log_analysis else None
    log_severity = log_analysis.severity if log_analysis else None
    log_confidence = log_analysis.confidence if log_analysis else 0.0
    fusion = fuse_signals(
        telemetry_risk=telemetry_risk.risk_score,
        log_severity=log_severity,
        log_confidence=log_confidence,
        config=DEFAULT_FUSION_CONFIG,
        log_domain=log_domain,
    )

    return TriageResponse(
        incident_score=fusion.incident_score,
        priority=fusion.priority,  # type: ignore[arg-type]
        route_to=fusion.route_to,
        explanation=fusion.explanation,
        telemetry=telemetry_risk,
        log_analysis=log_analysis,
    )
