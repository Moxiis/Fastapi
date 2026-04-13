import math
from typing import Any, List, Optional

import numpy as np

from ..core.storage import store_preprocessed


def preprocess_input(
    instance: Any, order: List[str], svc: Optional[object] = None
) -> List[List[float]]:
    """Convert a validated input instance into a numeric feature row.

    If a `svc` is provided and the service exposes a fitted `preprocessor`, it
    will be used to transform the raw numeric row. Otherwise a minimal, safe
    fallback is applied (NaN/infinite checks and numeric conversion).

    Returns a 2D list suitable for scikit-learn style `.predict([row])` calls.
    Also stores the preprocessed features (best-effort) for auditing / training.
    """
    # collect raw values in declared order
    raw_row = []
    for fname in order:
        value = getattr(instance, fname)
        raw_row.append(value)

    # If the model service exposes a preprocessor, prefer it (ensures parity with training)
    preprocessor = getattr(svc, "preprocessor", None) if svc is not None else None
    if preprocessor is not None:
        try:
            arr = np.asarray([raw_row], dtype=float)
            transformed = preprocessor.transform(arr)
            feature_row = transformed.tolist()[0]
            # persist transformed features
            try:
                store_preprocessed(
                    {
                        "features": feature_row,
                        "model_version": getattr(svc, "model_version", None),
                    }
                )
            except Exception:
                pass
            return [feature_row]
        except Exception:
            # if preprocessor fails, fall back to safe numeric conversion
            pass

    # Fallback: convert to floats, guard against NaN/Inf
    feature_row = []
    for v in raw_row:
        try:
            num = float(v)
        except (TypeError, ValueError):
            num = 0.0
        if math.isnan(num) or math.isinf(num):
            num = 0.0
        feature_row.append(num)

    # Persist the preprocessed features for traceability
    try:
        store_preprocessed(
            {
                "features": feature_row,
                "model_version": getattr(svc, "model_version", None),
            }
        )
    except Exception:
        pass

    return [feature_row]
