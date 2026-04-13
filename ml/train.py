"""Simple training script for a house price model.

This downloads the California housing dataset, trains a small
RandomForestRegressor and saves the model with joblib.

Run from the `Fastapi` folder with: `python -m ml.train` or
`python ml/train.py`.
"""

import os
import json
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from app.core.config import settings


class ClipOutliers:
    """Simple transformer to clip values to the given quantile bounds.

    This is intentionally small and serializable with joblib.
    """

    def __init__(self, lower_q: float = 0.05, upper_q: float = 0.95):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        arr = np.asarray(X)
        self.lower_ = np.percentile(arr, self.lower_q * 100, axis=0)
        self.upper_ = np.percentile(arr, self.upper_q * 100, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X)
        return np.clip(arr, self.lower_, self.upper_)


def _registry_path(models_dir: str) -> str:
    return os.path.join(models_dir, "registry.json")


def _write_registry(models_dir: str, version_id: str, meta: dict) -> None:
    registry_file = _registry_path(models_dir)
    data = {"versions": {}, "latest": None}
    if os.path.exists(registry_file):
        try:
            with open(registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"versions": {}, "latest": None}

    data.setdefault("versions", {})
    data["versions"][version_id] = meta
    data["latest"] = version_id

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train_and_save(models_dir: Optional[str] = None, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> dict:
    """Train a model, fit a small preprocessor, evaluate against a simple
    baseline, and register the model if it improves on the baseline.

    When accepted the model and preprocessor are saved under
    `ml/models/<version>/model.joblib` and `preprocessor.joblib`, and
    `ml/models/registry.json` is updated with the `latest` pointer.
    """
    # If training data was provided (e.g., loaded from DB), use it.
    if X is None or y is None:
        print("Downloading California housing dataset...")
        data = fetch_california_housing()
        X = data.data
        y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build a small preprocessor pipeline: impute -> clip outliers -> scale
    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("clipper", ClipOutliers(lower_q=0.05, upper_q=0.95)),
            ("scaler", StandardScaler()),
        ]
    )

    print("Fitting preprocessor...")
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print("Training RandomForestRegressor (this may take a short while)...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train_proc, y_train)

    # Evaluate
    y_pred = model.predict(X_test_proc)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Baseline: median predictor
    baseline_pred = np.median(y_train)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, np.full_like(y_test, baseline_pred))))

    print(f"Model RMSE: {rmse:.4f}, Baseline RMSE: {baseline_rmse:.4f}")

    # Only register if the model beats baseline
    if rmse >= baseline_rmse:
        print("Model did not beat baseline; not registering new model.")
        return {"registered": False, "rmse": rmse, "baseline_rmse": baseline_rmse}

    # Save model + preprocessor under a new version
    models_dir = models_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    version_id = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    version_dir = os.path.join(models_dir, version_id)
    os.makedirs(version_dir, exist_ok=True)

    model_path = os.path.join(version_dir, "model.joblib")
    preproc_path = os.path.join(version_dir, "preprocessor.joblib")

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preproc_path)

    meta = {
        "model_path": os.path.relpath(model_path),
        "preprocessor_path": os.path.relpath(preproc_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "rmse": rmse,
    }

    _write_registry(models_dir, version_id, meta)

    print(f"Registered new model version: {version_id}")
    return {"registered": True, "version_id": version_id, "rmse": rmse, "baseline_rmse": baseline_rmse}


if __name__ == "__main__":
    train_and_save()
