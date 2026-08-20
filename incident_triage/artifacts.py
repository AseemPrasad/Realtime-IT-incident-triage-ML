from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_joblib(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    joblib.dump(payload, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


@dataclass
class ArtifactManifest:
    tabular_model: str | None = None
    feature_columns: str | None = None
    telemetry_metrics: str | None = None
    slm_dataset: str | None = None
    slm_adapter: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

