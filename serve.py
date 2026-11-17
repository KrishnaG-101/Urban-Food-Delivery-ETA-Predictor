"""Urban Food Delivery ETA Predictor - FastAPI Inference Service."""

from fastapi import FastAPI

app = FastAPI(
    title="Urban Food Delivery ETA Predictor",
    description="Production API for predicting food delivery times (ETA).",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}

