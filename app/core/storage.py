import json
from datetime import datetime

from .config import settings

DATA_DIR = settings.data_dir
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _write_jsonl(filename: str, obj: dict) -> str:
    path = DATA_DIR / filename
    payload = dict(obj)
    if "created_at" not in payload:
        payload["created_at"] = datetime.utcnow().isoformat() + "Z"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    return str(path)


def store_raw_input(obj: dict) -> str:
    """Append a raw input JSON object to `data/raw_inputs.jsonl`."""
    return _write_jsonl("raw_inputs.jsonl", obj)


def store_preprocessed(obj: dict) -> str:
    """Append a preprocessed features object to `data/preprocessed.jsonl`."""
    return _write_jsonl("preprocessed.jsonl", obj)


def store_prediction(obj: dict) -> str:
    """Append a prediction object to `data/predictions.jsonl`."""
    return _write_jsonl("predictions.jsonl", obj)
