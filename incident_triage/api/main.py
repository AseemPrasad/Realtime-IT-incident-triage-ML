from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from ..artifacts import load_joblib
from ..config import ARTIFACT_DIR, DEFAULT_FUSION_CONFIG
from ..fusion import fuse_signals
from ..logs import HeuristicLogAnalyzer
from ..schemas import IncidentRequest, LogAnalysis, TelemetryRisk, TriageResponse
from ..tabular_model import TelemetryAnomalyModel
from ..telemetry import TelemetryFeatureExtractor


app = FastAPI(title="Real-Time IT Incident Triage Engine", version="0.1.0")


class RuntimeState:
    telemetry_model: Optional[TelemetryAnomalyModel] = None
    log_analyzer: HeuristicLogAnalyzer = HeuristicLogAnalyzer()
    extractor: TelemetryFeatureExtractor = TelemetryFeatureExtractor()


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
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "telemetry_model": "loaded" if state.telemetry_model else "heuristic",
    }


@app.post("/api/v1/triage", response_model=TriageResponse)
def triage(request: IncidentRequest) -> TriageResponse:
    telemetry_values = request.telemetry.values
    if not telemetry_values:
        raise HTTPException(status_code=400, detail="telemetry.values must contain at least one value")

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
