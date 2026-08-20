from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class TelemetryConfig:
    value_col: str = "value"
    timestamp_col: str = "timestamp"
    window_sizes: tuple[int, ...] = (5, 15, 60)
    ewma_spans: tuple[int, ...] = (5, 20)
    fft_window: int = 30
    anomaly_threshold: float = 0.4
    value_floor: float = 95.0
    anomaly_window_start: str = "2014-07-12 00:00:00"
    anomaly_window_end: str = "2014-07-14 02:00:00"


@dataclass(frozen=True)
class FusionConfig:
    telemetry_weight: float = 0.7
    log_weight: float = 0.3
    p1_threshold: float = 0.85
    p2_threshold: float = 0.65
    p3_threshold: float = 0.45


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.15
    n_jobs: int = 4
    max_samples_for_slm: int = 2000
    output_dir: Path = field(default_factory=lambda: ARTIFACT_DIR)


DEFAULT_TELEMETRY_CONFIG = TelemetryConfig()
DEFAULT_FUSION_CONFIG = FusionConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()

