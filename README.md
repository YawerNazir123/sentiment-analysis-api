# Sentiment Analysis API

A production-style **NLP sentiment classifier** served as a REST API: classify short
review text as **positive**, **negative** or **neutral**, with per-class probabilities.

Built as an end-to-end ML project: data → training → evaluation → serving, with tests,
Docker and CI — the kind of pipeline an AI/ML Engineer is expected to ship.

## Results

| Metric | Value |
| --- | --- |
| Model | TF-IDF (word 1–2 grams) + Logistic Regression |
| Dataset | 1,068 labelled review sentences (`data/reviews.csv`) |
| Train / test split | 854 / 214 (stratified, 80/20) |
| Hold-out accuracy | **99.07%** |
| Precision / recall / F1 | 0.99 across all three classes |

> **Honest note:** the built-in dataset is generated from curated phrase templates
> (seeded and reproducible, see `src/data.py`), so the 99% figure is accuracy *on that
> dataset*, not on real-world reviews. The point of this repo is the complete,
> working pipeline — and `src/train.py` accepts any CSV with `text` and `label`
> columns, so you can drop in a real dataset (e.g. IMDB, Amazon Polarity) with one
> flag: `python -m src.train --data your_reviews.csv`.

## Quickstart

```bash
git clone https://github.com/YawerNazir123/sentiment-analysis-api.git
cd sentiment-analysis-api

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (optional — the API trains itself on first run if no model exists)
python -m src.train

# Serve the API
uvicorn src.app:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

## Using the API

```bash
# Single prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this phone, the battery lasts for days"}'
```

```json
{
  "text": "I love this phone, the battery lasts for days",
  "label": "positive",
  "confidence": 0.69,
  "probabilities": {"negative": 0.15, "neutral": 0.16, "positive": 0.69}
}
```

```bash
# Batch prediction (up to 100 texts)
curl -X POST http://127.0.0.1:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product, highly recommended", "Terrible, broke after one day"]}'
```

| Endpoint | Description |
| --- | --- |
| `GET /` | Service info |
| `GET /health` | Liveness check + model status |
| `GET /model-info` | Model metadata (labels, feature count) |
| `POST /predict` | Classify one text |
| `POST /predict-batch` | Classify up to 100 texts |
| `GET /docs` | Interactive API docs (Swagger UI) |

## Command-line use

```bash
python -m src.predict "I absolutely love this product, it works perfectly"
python -m src.predict --batch "Great product!" "Terrible, broke in a day"
```

## Docker

```bash
docker build -t sentiment-analysis-api .
docker run -p 8000:8000 sentiment-analysis-api

# or with Compose
docker compose up
```

## Project structure

```
sentiment-analysis-api/
├── src/
│   ├── data.py      # Built-in labelled dataset (reproducible, seeded)
│   ├── train.py     # Training + evaluation, saves the model with joblib
│   ├── model.py     # Load-or-train logic and prediction helpers
│   ├── app.py       # FastAPI service (single + batch prediction)
│   └── predict.py   # Command-line prediction
├── data/
│   └── reviews.csv  # 1,068 labelled review sentences
├── tests/
│   ├── test_model.py
│   └── test_api.py
├── .github/workflows/ci.yml   # Train + test on every push
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-dev.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

11 tests cover dataset balance, model training, prediction output, and every API
endpoint, including validation errors.

## What this demonstrates

- **NLP / text classification** — TF-IDF vectorisation, n-grams, logistic regression
- **Model evaluation** — stratified train/test split, classification report, per-class metrics
- **ML serving** — FastAPI, Pydantic validation, single + batch inference, lazy model loading
- **MLOps basics** — Docker image that trains at build time, GitHub Actions CI that
  retrains and tests on every push, reproducible seeded data

## License

MIT
