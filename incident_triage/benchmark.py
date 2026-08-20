from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import load_joblib, dump_json
from .config import DEFAULT_TELEMETRY_CONFIG
from .logs import HeuristicLogAnalyzer
from .tabular_model import TelemetryAnomalyModel
from .telemetry import TelemetryFeatureExtractor


def benchmark_tabular(model: TelemetryAnomalyModel, sample_values: list[float], runs: int = 200) -> dict:
    feature_row = model.extractor.extract_from_values(sample_values)
    aligned = model.extractor.align_features(feature_row)
    for _ in range(20):
        _ = model.model.predict_proba(aligned)

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.model.predict_proba(aligned)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the incident triage components")
    parser.add_argument("--model", type=Path, default=Path("artifacts/tabular_model.joblib"))
    parser.add_argument("--input", type=Path, default=Path("data/cpu_utilization_asg_misconfiguration.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()

    model = load_joblib(args.model)
    telemetry = pd.read_csv(args.input)
    sample_values = telemetry["value"].tail(60).astype(float).tolist()
    sample = HeuristicLogAnalyzer().analyze(
        "PacketResponder 2 for block blk_902318412891 terminating due to socket exception"
    )
    sample_dict = sample.model_dump() if hasattr(sample, "model_dump") else sample.dict()

    benchmark = {
        "tabular_latency": benchmark_tabular(model, sample_values),
        "log_analyzer": {"sample": sample_dict},
    }
    dump_json(args.output, benchmark)
    print(json.dumps(benchmark, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
