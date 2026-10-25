"""Model loading and prediction helpers shared by the API and the CLI."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.data import generate_dataset
from src.train import DEFAULT_MODEL_PATH, build_pipeline

LABELS = ("positive", "negative", "neutral")


def load_or_train(model_path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a saved model, or train a fresh one if none exists yet.

    Training on the built-in dataset takes a couple of seconds, so a
    fresh clone is usable immediately without a separate training step.
    """
    if model_path.exists():
        return joblib.load(model_path)
    print(f"No saved model at {model_path}; training a fresh one ...")
    df = generate_dataset()
    model = build_pipeline()
    model.fit(df["text"], df["label"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def predict_text(model: Pipeline, text: str) -> dict:
    """Predict the sentiment of one text, with per-class probabilities."""
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    label = str(model.predict([text])[0])
    probabilities = dict(
        zip(model.classes_.tolist(), (float(p) for p in model.predict_proba([text])[0]))
    )
    return {
        "text": text,
        "label": label,
        "confidence": round(probabilities[label], 4),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
    }


def predict_batch(model: Pipeline, texts: list[str]) -> list[dict]:
    """Predict the sentiment of several texts at once."""
    return [predict_text(model, text) for text in texts]
