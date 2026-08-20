from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from .artifacts import dump_joblib, dump_json, load_joblib
from .telemetry import TelemetryFeatureExtractor

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None


def _build_estimator(random_state: int, n_jobs: int):
    if lgb is not None:
        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )


def _make_calibrator(estimator, method: str = "sigmoid", cv: int = 3):
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:  # pragma: no cover - older sklearn
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 37)
    best = 0.5
    best_f1 = -1.0
    y_true = np.asarray(y_true).astype(int)
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best = float(threshold)
    return best


@dataclass
class TelemetryAnomalyModel:
    model: Any
    feature_columns: list[str]
    threshold: float
    extractor: TelemetryFeatureExtractor
    metrics: dict[str, Any]

    @classmethod
    def train(
        cls,
        frame: pd.DataFrame,
        label_col: str = "is_anomaly",
        random_state: int = 42,
        n_jobs: int = 4,
        calibration_method: str = "sigmoid",
    ) -> "TelemetryAnomalyModel":
        extractor = TelemetryFeatureExtractor()
        features = extractor.build_training_frame(frame)
        if label_col not in frame.columns:
            raise KeyError(f"missing label column {label_col!r}")
        y = frame.sort_values(extractor.config.timestamp_col).reset_index(drop=True)[label_col].astype(int)
        X = extractor.align_features(features)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        estimator = _build_estimator(random_state=random_state, n_jobs=n_jobs)
        calibrator = _make_calibrator(estimator, method=calibration_method, cv=3)
        calibrator.fit(X_train, y_train)

        valid_split = max(1, int(len(X_train) * 0.2))
        X_threshold = X_train.iloc[-valid_split:]
        y_threshold = y_train.iloc[-valid_split:]
        threshold_proba = calibrator.predict_proba(X_threshold)[:, 1]
        threshold = best_threshold(y_threshold.to_numpy(), threshold_proba)

        test_proba = calibrator.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, test_proba)) if len(np.unique(y_test)) > 1 else None,
            "threshold": float(threshold),
            "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        return cls(
            model=calibrator,
            feature_columns=extractor.feature_names(),
            threshold=threshold,
            extractor=extractor,
            metrics=metrics,
        )

    def _align(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.extractor.align_features(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self.model)
        X = self._align(frame)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(frame)
        return (proba >= self.threshold).astype(int)

    def predict_from_values(self, values: Iterable[float]) -> tuple[float, bool, dict[str, float]]:
        features = self.extractor.extract_from_values(list(values))
        aligned = self.extractor.align_features(features)
        proba = float(self.model.predict_proba(aligned)[:, 1][0])
        return proba, proba >= self.threshold, aligned.iloc[0].to_dict()

    def save(self, model_path, metadata_path=None, features_path=None) -> None:
        dump_joblib(model_path, self)
        if metadata_path is not None:
            dump_json(metadata_path, self.metrics)
        if features_path is not None:
            dump_json(features_path, self.feature_columns)

    @classmethod
    def load(cls, model_path):
        return load_joblib(model_path)

