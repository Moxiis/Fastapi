from fastapi import APIRouter, Depends, HTTPException

from .model import ModelService, get_model_service
from .schemas import HousePriceInput

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/predict", summary="Predict house price from features")
def predict_house_price(payload: HousePriceInput, svc: ModelService = Depends(get_model_service)):
    """Return a single numeric prediction for provided house features.

    The input fields correspond to the features used by the trained model.
    """
    features = [
        [
            payload.medinc,
            payload.houseage,
            payload.averooms,
            payload.avebedrms,
            payload.population,
            payload.aveoccup,
            payload.latitude,
            payload.longitude,
        ]
    ]
    try:
        pred = svc.predict(features)
        return {"predicted_price": float(pred[0])}
    except Exception as exc:  # pragma: no cover - surface errors to client
        raise HTTPException(status_code=500, detail=str(exc))
