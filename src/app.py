import os
import time
import logging
import joblib
import pandas as pd
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app
from pythonjsonlogger import jsonlogger

# Structured JSON Logging
logger = logging.getLogger("heart_disease_api")
log_handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)

# Prometheus Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_duration_seconds", "Inference Latency in Seconds", ["endpoint"])
PREDICTION_COUNTER = Counter("model_predictions_total", "Model Predictions Count", ["predicted_class"])

app = FastAPI(
    title="Heart Disease Risk Inference API",
    version="1.0.0",
    description="Production API with input validation, confidence scoring, and Prometheus telemetry."
)

# Expose Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

MODEL_PATH = os.getenv("MODEL_PATH", "models/heart_disease_pipeline.pkl")
model = None

@app.on_event("startup")
def load_production_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded production model from {MODEL_PATH}")
    else:
        logger.error(f"Model artifact not found at {MODEL_PATH}")

class PatientInput(BaseModel):
    age: float = Field(..., ge=1.0, le=120.0, example=58.0)
    sex: float = Field(..., ge=0, le=1, example=1.0)
    cp: float = Field(..., ge=0, le=4, example=2.0)
    trestbps: float = Field(..., ge=50.0, le=250.0, example=120.0)
    chol: float = Field(..., ge=100.0, le=600.0, example=240.0)
    fbs: float = Field(..., ge=0, le=1, example=0.0)
    restecg: float = Field(..., ge=0, le=2, example=1.0)
    thalach: float = Field(..., ge=40.0, le=250.0, example=160.0)
    exang: float = Field(..., ge=0, le=1, example=0.0)
    oldpeak: float = Field(..., ge=0.0, le=10.0, example=1.2)
    slope: float = Field(..., ge=0, le=3, example=1.0)
    ca: float = Field(..., ge=0.0, le=3.0, example=0.0)
    thal: float = Field(..., ge=0.0, le=7.0, example=3.0)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Healthy, 1 = Risk Detected")
    risk_score: float = Field(..., description="Probability of heart disease")
    status: Literal["healthy", "risk_detected"]
    latency_ms: float

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact uninitialized.")
    return {"status": "HEALTHY", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    start = time.time()
    endpoint = "/predict"

    if model is None:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=503).inc()
        raise HTTPException(status_code=503, detail="Model is uninitialized.")

    try:
        input_df = pd.DataFrame([payload.model_dump()])
        pred = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        risk_prob = float(probabilities[1])

        elapsed = time.time() - start
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=200).inc()
        PREDICTION_COUNTER.labels(predicted_class=str(pred)).inc()

        logger.info("Inference executed", extra={
            "input": payload.model_dump(),
            "prediction": pred,
            "risk_score": risk_prob,
            "latency_ms": elapsed * 1000
        })

        return PredictionResponse(
            prediction=pred,
            risk_score=round(risk_prob, 4),
            status="risk_detected" if pred == 1 else "healthy",
            latency_ms=round(elapsed * 1000, 2)
        )
    except Exception as exc:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status=500).inc()
        logger.error(f"Inference error: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Inference pipeline failure.")
