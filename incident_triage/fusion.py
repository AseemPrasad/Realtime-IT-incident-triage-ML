from __future__ import annotations

from dataclasses import dataclass

from .config import FusionConfig
from .logs import severity_score


@dataclass(frozen=True)
class FusionResult:
    incident_score: float
    priority: str
    route_to: str
    explanation: str


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def priority_from_score(score: float, config: FusionConfig = FusionConfig()) -> str:
    if score >= config.p1_threshold:
        return "P1"
    if score >= config.p2_threshold:
        return "P2"
    return "P3"


def route_for_priority(priority: str, log_domain: str | None = None) -> str:
    if priority == "P1":
        return "incident-bridge"
    if log_domain and "NameSystem" in log_domain:
        return "hdfs-platform"
    if log_domain and "DataNode" in log_domain:
        return "storage-oncall"
    return "general-oncall"


def fuse_signals(
    telemetry_risk: float,
    log_severity: str | None = None,
    log_confidence: float = 0.0,
    config: FusionConfig = FusionConfig(),
    log_domain: str | None = None,
) -> FusionResult:
    telemetry_risk = _clamp(telemetry_risk)
    severity_component = severity_score(log_severity or "INFO") * _clamp(log_confidence)
    synergy_bonus = 0.0
    if telemetry_risk > 0.75 and severity_component > 0.75:
        synergy_bonus = 0.08
    if log_severity == "CRITICAL":
        synergy_bonus = max(synergy_bonus, 0.05)

    score = _clamp(
        config.telemetry_weight * telemetry_risk + config.log_weight * severity_component + synergy_bonus
    )
    priority = priority_from_score(score, config)
    route = route_for_priority(priority, log_domain)
    explanation = (
        f"Telemetry risk {telemetry_risk:.2f}, log severity {log_severity or 'UNKNOWN'} "
        f"with confidence {log_confidence:.2f} produced fused score {score:.2f}."
    )
    return FusionResult(
        incident_score=score,
        priority=priority,
        route_to=route,
        explanation=explanation,
    )

