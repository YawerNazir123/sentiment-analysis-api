"""FastAPI service for the sentiment classifier.

Run with::

    uvicorn src.app:app --reload

Endpoints
---------
GET  /            Service info
GET  /health      Liveness check + model status
GET  /model-info  Model metadata (labels, features)
POST /predict     {"text": "..."} -> sentiment + probabilities
POST /predict-batch {"texts": ["...", "..."]} -> list of predictions
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import LABELS, load_or_train, predict_batch, predict_text

_model = None


def get_model():
    """Return the loaded model, loading (or training) it on first use."""
    global _model
    if _model is None:
        _model = load_or_train()
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # preload so the first request is fast
    yield
    global _model
    _model = None


app = FastAPI(
    title="Sentiment Analysis API",
    description="Classify short review text as positive, negative or neutral.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=100, description="Texts to classify"
    )


@app.get("/")
def root() -> dict:
    return {
        "service": "sentiment-analysis-api",
        "version": "1.0.0",
        "labels": list(LABELS),
        "endpoints": ["/health", "/model-info", "/predict", "/predict-batch", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    model = get_model()
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info() -> dict:
    model = get_model()
    vectorizer = model.named_steps["tfidf"]
    return {
        "model": "TF-IDF (1-2 grams) + LogisticRegression",
        "labels": list(LABELS),
        "n_features": len(vectorizer.get_feature_names_out()),
        "classes": model.classes_.tolist(),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        return predict_text(get_model(), request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict-batch")
def predict_many(request: PredictBatchRequest) -> dict:
    return {"predictions": predict_batch(get_model(), request.texts)}
