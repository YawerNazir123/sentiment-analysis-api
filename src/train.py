"""Model training, evaluation and persistence.

Trains a TF-IDF + Logistic Regression sentiment classifier, evaluates
it on a held-out split, and saves the fitted pipeline with joblib.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import generate_dataset, load_dataset

DEFAULT_MODEL_PATH = Path("models/sentiment_model.joblib")
DEFAULT_DATA_PATH = Path("data/reviews.csv")
LABELS = ("positive", "negative", "neutral")


def build_pipeline() -> Pipeline:
    """TF-IDF (word uni/bi-grams) + logistic regression classifier."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(data_path: Path | None = None, model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Train on a CSV dataset (or generate the built-in one) and save the model.

    Returns a dict with the evaluation metrics.
    """
    if data_path is None or not data_path.exists():
        if data_path is not None:
            print(f"Dataset not found at {data_path}; generating the built-in dataset.")
        df = generate_dataset()
    else:
        df = load_dataset(str(data_path))

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "labels": sorted(df["label"].unique().tolist()),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Trained on {metrics['n_train']} examples, tested on {metrics['n_test']}.")
    print(f"Hold-out accuracy: {metrics['accuracy']:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Model saved to {model_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sentiment classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV with 'text' and 'label' columns (default: data/reviews.csv). "
        "If the file does not exist, the built-in dataset is generated.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model (default: models/sentiment_model.joblib).",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional path to write evaluation metrics as JSON.",
    )
    args = parser.parse_args()

    metrics = train(data_path=args.data, model_path=args.model)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2))
        print(f"Metrics written to {args.metrics}")


if __name__ == "__main__":
    main()
