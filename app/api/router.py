import logging

from pydantic import ValidationError

from fastapi import APIRouter, Depends, HTTPException

from ..core.storage import store_prediction
from ..validation.validation import validate_and_store
from .model import ModelService, get_model_service
from .preprocessing import preprocess_input
from .schemas import HousePriceInput, HousePriceOutput
from fastapi import BackgroundTasks
import traceback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post(
    "/predict",
    response_model=HousePriceOutput,
    summary="Predict house price from features",
)
def predict_house_price(
    payload: HousePriceInput, svc: ModelService = Depends(get_model_service)
):
    """Return a single numeric prediction for provided house features.

    The input fields correspond to the features used by the trained model.
    Feature order is derived from the `HousePriceInput` schema to avoid hardcoding.
    """
    # Determine feature order from the schema (preserves declared order)
    try:
        order = list(HousePriceInput.model_fields.keys())
    except Exception:
        # fallback to instance order
        order = list(payload.model_dump().keys())

    if not order:
        logger.error("Could not determine feature order from HousePriceInput schema")
        raise HTTPException(
            status_code=500,
            detail="Model feature order could not be determined from schema",
        )
    # Persist raw/validated input (user data validation step)
    try:
        validated, _ = validate_and_store(payload.model_dump())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Preprocessing: convert validated payload into numeric features
    try:
        features = preprocess_input(validated, order, svc)
    except Exception:
        logger.exception("Preprocessing failed")
        raise HTTPException(status_code=500, detail="Preprocessing failed")

    # Call model predict and handle common error cases explicitly
    try:
        pred = svc.predict(features)
    except ValueError as exc:
        # common when feature count/shape doesn't match
        logger.exception("Prediction failed due to invalid input or feature shape")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed (invalid input or feature shape): {exc}",
        )
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500, detail="Unexpected server error during prediction"
        )

    # Validate prediction output shape/value
    try:
        predicted_price = float(pred[0])
    except (IndexError, TypeError, ValueError) as exc:
        logger.exception("Model returned invalid prediction")
        raise HTTPException(
            status_code=500, detail=f"Model returned invalid prediction: {exc}"
        )

    # Include model version in response and persistence when available
    model_version = getattr(svc, "model_version", None)

    # Store prediction + model version (best-effort)
    try:
        store_prediction(
            {
                "input": payload.model_dump(),
                "features": features[0],
                "prediction": predicted_price,
                "model_version_id": model_version,
            }
        )
    except Exception:
        logger.exception("Failed to persist prediction")

    # Response model will perform final validation/serialization
    return HousePriceOutput(predicted_price=predicted_price, model_version_id=str(model_version))


@router.post("/train", summary="Trigger model training from DB historical data")
def trigger_training(background_tasks: BackgroundTasks):
    """Trigger model training using historical data stored in the database.

    This enqueues training into FastAPI's background tasks so the HTTP
    request returns quickly. Results are logged; model registry is updated
    if the trained model beats the baseline.
    """
    from ..core.storage import create_storage_tables, get_initial_training_data

    create_storage_tables()

    dataset = get_initial_training_data()
    if dataset is None:
        logger.info("No initial training data found in DB. Aborting training trigger.")
        return {"status": "no-data", "detail": "No initial training data found in DB"}

    X, y = dataset

    def _run_training(X_local, y_local):
        try:
            # import here to avoid circular imports at module load time
            import ml.train as ml_train

            result = ml_train.train_and_save(models_dir=None, X=X_local, y=y_local)
            logger.info("Training finished: %s", result)
        except Exception:
            logger.exception("Training failed: %s", traceback.format_exc())

    background_tasks.add_task(_run_training, X, y)
    return {"status": "started", "detail": "Training started in background"}
