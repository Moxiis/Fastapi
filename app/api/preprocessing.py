from typing import List, Any

from ..core.storage import store_preprocessed


def preprocess_input(instance: Any, order: List[str]) -> List[List[float]]:
    """Convert a validated input instance into a numeric feature row

    - `instance` is expected to be a Pydantic model instance (e.g. HousePriceInput)
    - `order` is the list of feature names in the expected model order

    Returns a 2D list suitable for scikit-learn style `.predict([row])` calls.
    Also stores the preprocessed features (best-effort) for auditing / training.
    """
    feature_row = []
    for fname in order:
        value = getattr(instance, fname)
        feature_row.append(float(value))

    # Persist the preprocessed features for traceability
    try:
        store_preprocessed({"features": feature_row})
    except Exception:
        pass

    return [feature_row]
