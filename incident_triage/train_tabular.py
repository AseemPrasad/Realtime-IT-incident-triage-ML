from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .artifacts import ensure_dir, dump_json
from .config import DEFAULT_TELEMETRY_CONFIG
from .tabular_model import TelemetryAnomalyModel


def load_telemetry_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[DEFAULT_TELEMETRY_CONFIG.timestamp_col] = pd.to_datetime(
        frame[DEFAULT_TELEMETRY_CONFIG.timestamp_col], errors="coerce"
    )
    frame = frame.sort_values(DEFAULT_TELEMETRY_CONFIG.timestamp_col).reset_index(drop=True)
    return frame


def label_telemetry(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    start = pd.Timestamp(DEFAULT_TELEMETRY_CONFIG.anomaly_window_start)
    end = pd.Timestamp(DEFAULT_TELEMETRY_CONFIG.anomaly_window_end)
    in_window = (df[DEFAULT_TELEMETRY_CONFIG.timestamp_col] >= start) & (
        df[DEFAULT_TELEMETRY_CONFIG.timestamp_col] <= end
    )
    high_values = df[DEFAULT_TELEMETRY_CONFIG.value_col] >= DEFAULT_TELEMETRY_CONFIG.value_floor
    df["is_anomaly"] = (in_window | high_values).astype(int)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the telemetry anomaly model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-name", type=str, default="tabular_model.joblib")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    frame = label_telemetry(load_telemetry_frame(args.input))
    model = TelemetryAnomalyModel.train(frame)

    model_path = args.output_dir / args.model_name
    metrics_path = args.output_dir / "tabular_metrics.json"
    features_path = args.output_dir / "tabular_feature_columns.json"
    model.save(model_path, metrics_path, features_path)
    print(f"Saved telemetry model to {model_path}")
    print(f"Threshold: {model.threshold:.3f}")
    print(f"Metrics written to {metrics_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

