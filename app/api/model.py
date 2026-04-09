import os
from functools import lru_cache
from typing import Any, Optional

import joblib

from ..core.config import settings


class ModelService:
    """Loads the saved model lazily and exposes a predict method.

    Single responsibility: model loading and prediction. Designed to be used
    as a FastAPI dependency via `get_model_service()`.
    """

    def __init__(self, model_path: Optional[str] = None):
        # prefer explicit model_path, otherwise use settings value
        self._model_path = model_path or settings.model_path
        self._model = None

    def load(self) -> Any:
        if self._model is None:
            if not os.path.exists(self._model_path):
                raise FileNotFoundError(f"Model file not found: {self._model_path}")
            self._model = joblib.load(self._model_path)
        return self._model

    def predict(self, X):
        model = self.load()
        return model.predict(X)


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()
