from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import fft as scipy_fft

from .config import TelemetryConfig


def _safe_float_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("telemetry values cannot be empty")
    if not np.isfinite(arr).all():
        raise ValueError("telemetry values must be finite numbers")
    return arr


def _fft_energy(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    coeffs = scipy_fft.fft(values)
    return float(np.sum(np.abs(coeffs) ** 2) / max(1, values.size))


@dataclass
class TelemetryFeatureExtractor:
    config: TelemetryConfig = field(default_factory=TelemetryConfig)

    def feature_names(self) -> list[str]:
        names = [
            "value",
            "diff_1",
            "pct_change_1",
            "global_mean",
            "global_std",
            "global_min",
            "global_max",
        ]
        for window in self.config.window_sizes:
            names.extend(
                [
                    f"rolling_mean_{window}",
                    f"rolling_std_{window}",
                    f"rolling_min_{window}",
                    f"rolling_max_{window}",
                    f"rolling_range_{window}",
                    f"value_minus_mean_{window}",
                    f"zscore_{window}",
                ]
            )
        for span in self.config.ewma_spans:
            names.append(f"ewma_span_{span}")
        if len(self.config.ewma_spans) >= 2:
            names.append("ewma_delta")
        names.append(f"fft_energy_{self.config.fft_window}")
        return names

    def build_training_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.config.timestamp_col not in frame.columns or self.config.value_col not in frame.columns:
            raise KeyError(
                f"expected columns {self.config.timestamp_col!r} and {self.config.value_col!r}"
            )

        df = frame.copy()
        df[self.config.timestamp_col] = pd.to_datetime(df[self.config.timestamp_col], errors="coerce")
        df = df.sort_values(self.config.timestamp_col).reset_index(drop=True)
        series = df[self.config.value_col].astype(float)

        out = df[[self.config.timestamp_col, self.config.value_col]].copy()
        out["value"] = series
        out["diff_1"] = series.diff().fillna(0.0)
        out["pct_change_1"] = series.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out["global_mean"] = series.expanding(min_periods=1).mean()
        out["global_std"] = series.expanding(min_periods=1).std().fillna(0.0)
        out["global_min"] = series.expanding(min_periods=1).min()
        out["global_max"] = series.expanding(min_periods=1).max()

        for window in self.config.window_sizes:
            rolling = series.rolling(window=window, min_periods=1)
            mean = rolling.mean()
            std = rolling.std().fillna(0.0)
            min_ = rolling.min()
            max_ = rolling.max()
            out[f"rolling_mean_{window}"] = mean
            out[f"rolling_std_{window}"] = std
            out[f"rolling_min_{window}"] = min_
            out[f"rolling_max_{window}"] = max_
            out[f"rolling_range_{window}"] = max_ - min_
            out[f"value_minus_mean_{window}"] = series - mean
            out[f"zscore_{window}"] = np.where(std > 0, (series - mean) / std, 0.0)

        for span in self.config.ewma_spans:
            out[f"ewma_span_{span}"] = series.ewm(span=span, adjust=False).mean()
        if len(self.config.ewma_spans) >= 2:
            short_span, long_span = self.config.ewma_spans[:2]
            out["ewma_delta"] = out[f"ewma_span_{short_span}"] - out[f"ewma_span_{long_span}"]

        out[f"fft_energy_{self.config.fft_window}"] = [
            _fft_energy(series.iloc[max(0, idx - self.config.fft_window + 1) : idx + 1].to_numpy())
            for idx in range(len(series))
        ]
        return out

    def extract_from_values(self, values: Sequence[float]) -> pd.DataFrame:
        arr = _safe_float_array(values)
        series = pd.Series(arr)
        row: dict[str, float] = {
            "value": float(arr[-1]),
            "diff_1": float(arr[-1] - arr[-2]) if arr.size > 1 else 0.0,
            "pct_change_1": float((arr[-1] - arr[-2]) / abs(arr[-2])) if arr.size > 1 and arr[-2] != 0 else 0.0,
            "global_mean": float(series.mean()),
            "global_std": float(series.std(ddof=0)),
            "global_min": float(series.min()),
            "global_max": float(series.max()),
        }
        for window in self.config.window_sizes:
            slice_ = arr[-window:] if arr.size >= window else arr
            mean = float(np.mean(slice_))
            std = float(np.std(slice_))
            min_ = float(np.min(slice_))
            max_ = float(np.max(slice_))
            row[f"rolling_mean_{window}"] = mean
            row[f"rolling_std_{window}"] = std
            row[f"rolling_min_{window}"] = min_
            row[f"rolling_max_{window}"] = max_
            row[f"rolling_range_{window}"] = max_ - min_
            row[f"value_minus_mean_{window}"] = row["value"] - mean
            row[f"zscore_{window}"] = (row["value"] - mean) / std if std > 0 else 0.0
        for span in self.config.ewma_spans:
            row[f"ewma_span_{span}"] = float(series.ewm(span=span, adjust=False).mean().iloc[-1])
        if len(self.config.ewma_spans) >= 2:
            short_span, long_span = self.config.ewma_spans[:2]
            row["ewma_delta"] = row[f"ewma_span_{short_span}"] - row[f"ewma_span_{long_span}"]
        row[f"fft_energy_{self.config.fft_window}"] = _fft_energy(
            arr[-self.config.fft_window :] if arr.size >= self.config.fft_window else arr
        )
        return pd.DataFrame([row], columns=self.feature_names())

    def align_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame.copy()
        for name in self.feature_names():
            if name not in aligned.columns:
                aligned[name] = 0.0
        return aligned[self.feature_names()]

    def heuristic_risk(self, values: Sequence[float]) -> float:
        arr = _safe_float_array(values)
        last = float(arr[-1])
        baseline = float(np.mean(arr[:-1])) if arr.size > 1 else last
        spread = float(np.std(arr[:-1])) if arr.size > 2 else max(1.0, abs(baseline) * 0.05)
        z = abs(last - baseline) / max(spread, 1e-6)
        drift = abs(last - float(np.median(arr))) / max(spread, 1e-6)
        raw = 0.5 * z + 0.5 * drift
        return float(np.clip(raw / 6.0, 0.0, 1.0))
