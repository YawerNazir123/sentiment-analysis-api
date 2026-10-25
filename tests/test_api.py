"""Tests for the FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "sentiment-analysis-api"


def test_health_reports_model_loaded():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_reports_labels_and_features():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["labels"] == ["positive", "negative", "neutral"]
    assert body["n_features"] > 0


def test_predict_positive_text():
    response = client.post("/predict", json={"text": "I love this phone, it works perfectly"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert body["confidence"] > 0.5


def test_predict_negative_text():
    response = client.post(
        "/predict", json={"text": "This laptop is terrible, a complete waste of money"}
    )
    assert response.status_code == 200
    assert response.json()["label"] == "negative"


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_batch_returns_one_result_per_text():
    response = client.post(
        "/predict-batch",
        json={"texts": ["Great product, I love it", "Awful, broke on day one"]},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["label"] == "positive"
    assert predictions[1]["label"] == "negative"
