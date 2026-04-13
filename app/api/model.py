import json
import os
from functools import lru_cache
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.exceptions import NotFittedError

from ..core.config import settings


class ModelService:
    """Model loader that supports a simple model registry.

    Behavior:
    - If a `ml/models/registry.json` exists, loads the `latest` model and its
      accompanying `preprocessor.joblib` (if present) from the version directory.
    - Falls back to `settings.model_path` or an unversioned `ml/models/house_price_model.joblib`.

    The service exposes `predict(X)` and `model_version` for callers to include
    the version identifier in responses.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._explicit_model_path = model_path
        self._model_path: Optional[str] = None
        self._preprocessor_path: Optional[str] = None
        self._model = None
        self._preprocessor = None
        self._version: Optional[str] = None
        self._resolved = False
        self._resolve_paths()

    def _resolve_paths(self) -> None:
        # If caller provided an explicit model path, use it
        if self._explicit_model_path:
            self._model_path = self._explicit_model_path
            self._version = os.path.basename(self._model_path)
            self._resolved = True
            return

        # Try to find a registry under the project's ml/models folder
        base = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "ml", "models")
        )
        registry = os.path.join(base, "registry.json")
        if os.path.exists(registry):
            try:
                with open(registry, "r", encoding="utf-8") as f:
                    data = json.load(f)
                latest = data.get("latest")
                if latest:
                    version_dir = os.path.join(base, latest)
                    candidate_model = os.path.join(version_dir, "model.joblib")
                    candidate_pre = os.path.join(version_dir, "preprocessor.joblib")
                    if os.path.exists(candidate_model):
                        self._model_path = candidate_model
                        self._preprocessor_path = (
                            candidate_pre if os.path.exists(candidate_pre) else None
                        )
                        self._version = latest
                        self._resolved = True
                        return
            except Exception:
                # fallthrough to other heuristics
                pass

        # Fallbacks: configured settings.model_path or a project-level ml/models file
        if getattr(settings, "model_path", None) and os.path.exists(
            settings.model_path
        ):
            self._model_path = settings.model_path
            self._preprocessor_path = None
            self._version = os.path.basename(self._model_path)
            self._resolved = True
            return

        candidate = os.path.join(base, "house_price_model.joblib")
        if os.path.exists(candidate):
            self._model_path = candidate
            self._preprocessor_path = None
            self._version = "unversioned"
            self._resolved = True
            return

        # Last resort: keep settings.model_path even if missing (load will raise)
        self._model_path = getattr(settings, "model_path", None)
        self._preprocessor_path = None
        self._version = None
        self._resolved = True

    def load(self) -> Any:
        if self._model is None:
            if not self._model_path or not os.path.exists(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")
            self._model = joblib.load(self._model_path)
            # load preprocessor when available (best-effort)
            if self._preprocessor_path and os.path.exists(self._preprocessor_path):
                try:
                    self._preprocessor = joblib.load(self._preprocessor_path)
                except Exception:
                    self._preprocessor = None
        return self._model

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def model_version(self) -> str:
        if self._version:
            return self._version
        try:
            return str(int(os.path.getmtime(self._model_path)))
        except Exception:
            return "unknown"

    def predict(self, X):
        model = self.load()
        X_arr = np.asarray(X)
        # Apply saved preprocessor if present to ensure consistent transforms
        if self._preprocessor is not None:
            try:
                X_trans = self._preprocessor.transform(X_arr)
            except Exception as exc:
                raise ValueError(f"Preprocessor transform failed: {exc}")
        else:
            X_trans = X_arr

        try:
            return model.predict(X_trans)
        except NotFittedError:
            raise NotFittedError("Model is not fitted")
        except Exception:
            raise


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()
