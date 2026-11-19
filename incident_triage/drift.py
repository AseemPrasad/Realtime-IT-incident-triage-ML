from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


def calculate_psi(
    expected: np.ndarray | List[float],
    actual: np.ndarray | List[float],
    num_buckets: int = 10,
) -> float:
    """Calculates the Population Stability Index (PSI) between baseline and production feature distributions.
    
    Interpretation:
      PSI < 0.10: No significant drift; model behavior remains stable.
      0.10 <= PSI < 0.25: Moderate drift; investigation or fine-tuning recommended.
      PSI >= 0.25: Significant concept/data drift; automated re-training required.
    """
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    if len(exp_arr) < 2 or len(act_arr) < 2:
        return 0.0

    # Quantiles covering min to max of combined range so all points fall within bins
    combined_min = min(float(np.min(exp_arr)), float(np.min(act_arr)))
    combined_max = max(float(np.max(exp_arr)), float(np.max(act_arr)))
    if combined_max <= combined_min:
        return 0.0

    bins = np.linspace(combined_min - 1e-4, combined_max + 1e-4, num_buckets + 1)

    # Use additive Laplace smoothing proportional to sample size (minimum 1 count)
    # This prevents artificial drift spikes when comparing discrete or small-sample continuous bins
    exp_counts = np.histogram(exp_arr, bins=bins)[0].astype(float) + 1.0
    act_counts = np.histogram(act_arr, bins=bins)[0].astype(float) + 1.0

    exp_pct = exp_counts / np.sum(exp_counts)
    act_pct = act_counts / np.sum(act_counts)

    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(max(0.0, psi_value))




class TelemetryDriftMonitor:
    """Tracks baseline feature distributions and detects concept drift over live metrics."""

    def __init__(self, baseline_samples: Optional[List[float]] = None) -> None:
        self.baseline = list(baseline_samples) if baseline_samples else []

    def set_baseline(self, samples: List[float]) -> None:
        self.baseline = list(samples)

    def assess_drift(self, recent_samples: List[float]) -> Dict[str, object]:
        if not self.baseline:
            return {
                "status": "NO_BASELINE",
                "psi": 0.0,
                "drift_detected": False,
                "message": "Baseline distribution not set.",
            }

        psi = calculate_psi(self.baseline, recent_samples)
        if psi >= 0.25:
            status = "CRITICAL_DRIFT"
            drift_detected = True
        elif psi >= 0.10:
            status = "MODERATE_DRIFT"
            drift_detected = True
        else:
            status = "HEALTHY"
            drift_detected = False

        return {
            "status": status,
            "psi": round(psi, 4),
            "drift_detected": drift_detected,
            "sample_count": len(recent_samples),
        }
