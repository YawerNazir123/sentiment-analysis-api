"""Tests for the training pipeline and prediction helpers."""

from src.data import generate_dataset
from src.model import load_or_train, predict_batch, predict_text
from src.train import build_pipeline


def test_dataset_is_balanced_and_labelled():
    df = generate_dataset(n_per_class=50, seed=1)
    assert set(df["label"].unique()) == {"positive", "negative", "neutral"}
    assert (df.groupby("label").size() == 56).all()  # 50 generated + 6 handwritten
    assert df["text"].str.len().min() > 5


def test_pipeline_trains_and_predicts_known_sentiment():
    df = generate_dataset(n_per_class=100, seed=42)
    model = build_pipeline()
    model.fit(df["text"], df["label"])
    assert predict_text(model, "I love this phone, it works perfectly")["label"] == "positive"
    assert predict_text(model, "This laptop is terrible, a complete waste of money")["label"] == "negative"


def test_predict_text_returns_probabilities():
    model = load_or_train()
    result = predict_text(model, "Absolutely fantastic product, highly recommended")
    assert result["label"] in {"positive", "negative", "neutral"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01


def test_predict_batch_matches_single_predictions():
    model = load_or_train()
    texts = ["Great product, I love it", "Awful, broke on day one"]
    batch = predict_batch(model, texts)
    assert [r["label"] for r in batch] == [
        predict_text(model, t)["label"] for t in texts
    ]
