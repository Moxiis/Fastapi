"""Simple training script for a house price model.

This downloads the California housing dataset, trains a small
RandomForestRegressor and saves the model with joblib.

Run from the `Fastapi` folder with: `python -m ml.train` or
`python ml/train.py`.
"""
import os

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.joblib")


def train_and_save(model_path: str = MODEL_PATH) -> None:
    print("Downloading California housing dataset...")
    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestRegressor (this may take a short while)...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train_and_save()
