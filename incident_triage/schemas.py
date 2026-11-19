from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


Severity = Literal["LOW", "INFO", "MEDIUM", "HIGH", "CRITICAL"]
Priority = Literal["P3", "P2", "P1"]


class TelemetryPayload(BaseModel):
    values: Optional[list[float]] = None
    value: Optional[float] = None
    timestamps: Optional[list[datetime]] = None
    metric_name: str = "cpu_utilization"


class IncidentRequest(BaseModel):
    telemetry: TelemetryPayload
    log_content: Optional[str] = Field(
        default=None,
        max_length=10_000,
        description="Raw log content up to 10KB to protect against payload flooding / DoS",
    )
    asset_id: Optional[str] = None
    source: Optional[str] = None



class TelemetryRisk(BaseModel):
    risk_score: float
    threshold: float
    is_anomaly: bool
    probability: float
    feature_snapshot: dict[str, float]


class LogAnalysis(BaseModel):
    error_domain: str
    severity: Severity
    probable_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: Optional[str] = None


class TriageResponse(BaseModel):
    incident_score: float
    priority: Priority
    route_to: str
    explanation: str
    telemetry: TelemetryRisk
    log_analysis: Optional[LogAnalysis] = None

