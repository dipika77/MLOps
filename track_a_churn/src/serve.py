"""
Thin FastAPI wrapper that loads the CHAMPION model straight from the MLflow
Model Registry (not from a local file), so promoting a new version in the
registry is all it takes to change what this API serves - no redeploy of new
model bytes.

    uv run uvicorn src.serve:app --port 8080

Then:
    curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
         -d '{"gender": "Female", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "No",
              "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
              "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
              "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
              "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
              "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}'
"""
from pathlib import Path
from typing import Optional

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

MODEL_NAME = "telco_churn_classifier"
MODEL_STAGE = "Production"  # loaded by registry stage, same pattern as W16's provider fallback

app = FastAPI(title="Telco Churn Model Server", version="1.0")
_model = None


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    model_name: str
    model_version: Optional[str] = None
    model_stage: str


def get_model():
    global _model
    if _model is None:
        # loaded with the sklearn flavor (not the generic pyfunc wrapper) so we can call
        # predict_proba directly and return a real probability, not just the 0/1 class.
        _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    return _model


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    model = get_model()
    row = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction="Yes" if proba >= 0.5 else "No",
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )
