import os
from typing import Any

import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "house_price_model.joblib")


class ModelService:
    """Loads the saved model lazily and exposes a predict method.

    Single responsibility: model loading and prediction. Designed to be used
    as a FastAPI dependency via `get_model_service()`.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self._model_path = model_path
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


_SERVICE: ModelService | None = None


def get_model_service() -> ModelService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = ModelService()
    return _SERVICE
