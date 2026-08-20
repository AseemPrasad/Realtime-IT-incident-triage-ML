from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .schemas import LogAnalysis


SYSTEM_PROMPT = (
    "You are an expert IT incident root-cause engine. "
    "Return only a valid JSON object with keys error_domain, severity, and probable_cause."
)


def normalize_severity(level: str) -> str:
    level = (level or "").upper()
    mapping = {
        "TRACE": "LOW",
        "DEBUG": "LOW",
        "INFO": "INFO",
        "WARN": "MEDIUM",
        "WARNING": "MEDIUM",
        "ERROR": "HIGH",
        "FATAL": "CRITICAL",
    }
    return mapping.get(level, "MEDIUM")


def severity_score(severity: str) -> float:
    return {
        "LOW": 0.15,
        "INFO": 0.25,
        "MEDIUM": 0.5,
        "HIGH": 0.8,
        "CRITICAL": 1.0,
    }.get((severity or "").upper(), 0.4)


def derive_ground_truth(row: pd.Series | dict[str, Any]) -> dict[str, str]:
    content = str(row.get("Content", "")).lower()
    level = str(row.get("Level", "")).upper()
    component = str(row.get("Component", "unknown"))
    component_suffix = component.split("$")[-1] if component else "Component"

    if any(token in content for token in ["exception", "terminating", "failure", "error", "timeout"]):
        domain = "HDFS_DataNode" if "datanode" in component.lower() else "HDFS_Runtime"
        if "namesystem" in component.lower() or "blockmap" in content:
            domain = "HDFS_NameSystem"
        severity = "CRITICAL" if level in {"ERROR", "FATAL"} or "failure" in content else "HIGH"
        cause = "Packet responder termination, timeout, or storage subsystem failure"
    elif "blockmap updated" in content or "storedblock" in content:
        domain = "HDFS_NameSystem"
        severity = "INFO"
        cause = "Normal block lifecycle update"
    else:
        domain = f"HDFS_{component_suffix}"
        severity = normalize_severity(level)
        cause = f"System event observed on {component_suffix}"

    return {
        "error_domain": domain,
        "severity": severity,
        "probable_cause": cause,
    }


def build_chat_messages(log_content: str, target: dict[str, str] | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Log Line: {log_content}"},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return messages


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [
        text.strip(),
    ]
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def heuristic_log_analysis(log_content: str) -> LogAnalysis:
    lower = (log_content or "").lower()
    if any(token in lower for token in ["terminating", "failure", "timeout", "exception", "error"]):
        severity = "CRITICAL" if any(token in lower for token in ["failure", "exception", "error"]) else "HIGH"
        domain = "HDFS_DataNode" if "datanode" in lower else "HDFS_Runtime"
        cause = "Observed termination, exception, or timeout in the data path"
    elif "blockmap updated" in lower or "storedblock" in lower:
        severity = "INFO"
        domain = "HDFS_NameSystem"
        cause = "Routine metadata update"
    else:
        severity = "MEDIUM"
        domain = "HDFS_Component"
        cause = "General operational event"
    confidence = severity_score(severity)
    return LogAnalysis(
        error_domain=domain,
        severity=severity,  # type: ignore[arg-type]
        probable_cause=cause,
        confidence=confidence,
        raw_text=log_content,
    )


@dataclass
class HeuristicLogAnalyzer:
    def analyze(self, log_content: str | None) -> LogAnalysis | None:
        if not log_content:
            return None
        return heuristic_log_analysis(log_content)

    def build_prompt(self, log_content: str) -> list[dict[str, str]]:
        return build_chat_messages(log_content)

    def training_rows(self, frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        selected = frame.head(limit) if limit else frame
        for _, row in selected.iterrows():
            target = derive_ground_truth(row)
            rows.append({"messages": build_chat_messages(str(row.get("Content", "")), target)})
        return rows

