# 📦 Demand Forecasting & Anomaly Detection Dashboard

A full-stack MLOps system that forecasts short-term product demand and automatically flags unusual sales spikes/drops for operations teams to review.

> Built end-to-end: data pipeline → ensemble forecasting model → anomaly detection → explainable predictions (SHAP) → REST API → interactive dashboard → scheduled retraining with experiment tracking.

<!-- Replace this with an actual screenshot or GIF of your Streamlit dashboard once it's running -->
![Dashboard Screenshot](docs/dashboard_forecast.png)
![Dashboard Screenshot](docs/dashboard_forecast2.png)
![Dashboard Screenshot](docs/mlflow_ss.png)
---

## 🧩 Problem Statement

Operations and inventory teams need to know two things every day: *how much demand is coming*, and *is anything happening right now that doesn't match the pattern*. Manually eyeballing sales spreadsheets doesn't scale across SKUs and doesn't catch anomalies fast enough. This project automates both: a 7-day rolling forecast per SKU, plus automatic flagging of days where actual demand deviates sharply from expected — with an explanation of *why* the model predicted what it did, not just a black-box number.

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   Data Source          │
                    │ (synthetic generator /  │
                    │  CSV / real sales data)  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PostgreSQL           │
                    │  sales_data table       │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                             ▼
      ┌─────────────────────┐      ┌─────────────────────┐
      │  Prefect Flow          │      │  FastAPI Backend       │
      │  (scheduled retrain)    │      │  /predict                │
      │  - feature engineering  │      │  /explain (SHAP)          │
      │  - train XGBoost        │      │  /anomalies                │
      │  - train Prophet        │      │  /model-metrics             │
      │  - train IsolationForest│      └──────────┬─────────────┘
      │  - log to MLflow          │                 │
      └──────────┬──────────────┘                 │
                  │                                 │
                  ▼                                 ▼
          ┌───────────────┐               ┌─────────────────┐
          │  MLflow           │               │  Streamlit          │
          │  Tracking / Registry│◄──────────────┤  Dashboard            │
          └───────────────┘               └─────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend | Streamlit + Plotly | Fast to build, interactive charts, good enough for an internal ops tool |
| Backend API | FastAPI | Async, auto-generated OpenAPI docs, standard for ML model serving |
| Forecasting | XGBoost + Prophet (ensemble) | Prophet captures seasonality/trend; XGBoost captures non-linear effects from engineered lag features. Combining both covers each model's individual weak spots |
| Anomaly Detection | Isolation Forest (scikit-learn) | Unsupervised, no labeled anomaly data required, works well on the engineered feature set |
| Explainability | SHAP | TreeExplainer over the XGBoost model — shows which features drove each individual prediction |
| Database | PostgreSQL | Reliable, well-indexed on `(sku_id, date)`; TimescaleDB noted below as the production-scale upgrade |
| Orchestration | Prefect | Lightweight scheduled retraining flow, minimal infra vs. running a full Airflow cluster locally |
| Experiment Tracking | MLflow | Logs RMSE, parameters, and feature sets for every training run; enables comparing model versions over time |

---

## 📁 Project Structure

```
demand-forecast-dashboard/
├── data/
│   └── generate_synthetic_data.py   # synthetic sales data w/ injected anomalies
├── db/
│   └── models.py                    # SQLAlchemy schema + CSV → Postgres loader
├── ml/
│   ├── features.py                  # lag/rolling/calendar feature engineering
│   ├── train.py                     # Prophet + XGBoost training, MLflow logging
│   └── anomaly.py                   # Isolation Forest anomaly detection
├── api/
│   └── main.py                      # FastAPI: /predict /explain /anomalies
├── dashboard/
│   └── app.py                       # Streamlit dashboard
├── flows/
│   └── retrain_flow.py              # Prefect scheduled retraining flow
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ (running locally or remotely)

### Setup

```bash
git clone <your-repo-url>
cd demand-forecast-dashboard
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create the database:
```sql
CREATE DATABASE demand_forecast;
```

Update the connection string in `db/models.py` with your credentials, then:

```bash
python data/generate_synthetic_data.py   # generate sample data
python db/models.py                      # load into Postgres
python ml/train.py                       # train models, logs to MLflow
uvicorn api.main:app --reload            # start the API (localhost:8000)
streamlit run dashboard/app.py           # start the dashboard (localhost:8501)
```

View experiment tracking history:
```bash
mlflow ui   # localhost:5000
```

---

## 🔑 Key Technical Decisions

<!-- Fill these in with YOUR actual findings once you've run training a few times — 
     this section is what makes a portfolio README stand out over a generic one -->

- **Why an ensemble instead of one model?** [e.g. "Prophet alone underfit demand spikes around promotional periods; XGBoost alone couldn't extrapolate long-term trend beyond the training window. Combining both reduced RMSE by X% over either model alone."]
- **Why Isolation Forest over a simpler threshold rule?** Unsupervised detection adapts per-SKU without needing hand-tuned thresholds for each product's normal variance.
- **Why SHAP for explainability?** Ops teams are more likely to trust and act on a flagged anomaly or forecast if they can see *which features* drove it (e.g. "this spike prediction is 80% driven by the day-of-week and rolling average, not noise").

---

## 📊 Results

<!-- Replace with your actual numbers after training -->

| Metric | Value |
|---|---|
| XGBoost RMSE (avg across SKUs) | (24.76 + 22.90 + 10.98 + 22.38 + [SKU-005's number]) / 5 |
| Anomalies flagged (test window) | 47 (across 5 SKUs, ~1.9% of days)|
| Retraining frequency | Weekly (Prefect scheduled) |

<!-- Add your MLflow UI screenshot here showing RMSE trend across multiple retraining runs — 
     this is one of the most convincing artifacts you can include -->
![MLflow Experiment Tracking](docs/mlflow_screenshot.png)

---

## 🔭 What I'd Do at Scale

This project intentionally uses lighter-weight tools for local development. In a production environment, the natural upgrades are:

- **PostgreSQL → TimescaleDB**: purpose-built time-series compression and query performance at millions+ of rows
- **Prefect → Airflow**: distributed task orchestration across multiple workers/teams
- **Local MLflow → hosted MLflow / model registry with CI/CD**: automated promotion of models from staging to production based on validation metrics
- **Streamlit → React + D3.js**: fully custom interactive time-series brushing/zooming for power users
- **Isolation Forest → PyOD ensemble**: combine multiple anomaly detection algorithms (e.g. Isolation Forest + LOF + Autoencoder) and flag by consensus

---

## 📄 License

<!-- e.g. MIT — add a LICENSE file if you want this public -->

