# FastAPI House Price Prediction Example

This example shows a small FastAPI app that exposes a machine-learning
prediction endpoint for a simple house-price model trained on the
California Housing dataset.

## Quick start

1: Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2: Train the model (this will download the dataset and save a model file):

```bash
python -m ml.train
```

3: Run the app:

```bash
uvicorn main:app --reload --port 8000
```

4: Try a prediction (example using `http` or `curl`):

```bash
curl -X POST "http://127.0.0.1:8000/ml/predict" -H "Content-Type: application/json" -d \
'{"medinc":8.3252,"houseage":41.0,"averooms":6.984127,"avebedrms":1.0238,"population":322.0,"aveoccup":2.555556,"latitude":37.88,"longitude":-122.23}'
```

Notes

- The model output is the target value from scikit-learn's dataset.
- The code is intentionally small and follows SRP and separation of concerns:
  - `ml/train.py` trains and saves the model
  - `ml/model.py` handles model loading and prediction
  - `ml/router.py` exposes the FastAPI endpoint

## TODO

- Validation:
  - add data validation before prediction
- Asynchronus:
  - add asynchronus inference for our code
  - check how batch inference is working
  - compare single file vs batch time
- Training:
  - add our training data to our sqlalchemy database
  - add our predictions data to sqlalchemy database
  - automaticly run training with our sqlalchemy database
