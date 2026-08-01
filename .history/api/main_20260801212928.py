from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.xgboost
import shap
from ml.features import build_features
import xgboost as xgb

app = FastAPI(title="Demand Forecasting API")

df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])

FEATURE_COLS = ["lag_1", "lag_7", "lag_14", "rolling_mean_7",
                 "rolling_std_7", "day_of_week", "month", "is_weekend"]

@app.get("/")
def root():
    return {"status": "ok", "message": "Demand Forecasting API"}

@app.get("/predict/{sku_id}")
def predict(sku_id: str, days: int = 7):
    sku_df = df[df["sku_id"] == sku_id]
    if sku_df.empty:
        raise HTTPException(404, f"No data for {sku_id}")

    features_df = build_features(sku_df)
    
    model = xgb.XGBRegressor()
    model.load_model(f"models/{sku_id}_xgb.json")
    # Note: requires registering the model under this name during training —
    # see MLflow Model Registry docs. For a simpler v1, load from a local
    # models/ folder saved with model.save_model() instead.

    latest_features = features_df[FEATURE_COLS].iloc[[-1]]
    prediction = model.predict(latest_features)[0]

    return {"sku_id": sku_id, "forecast_next_day": float(prediction)}

@app.get("/explain/{sku_id}")
def explain(sku_id: str):
    sku_df = df[df["sku_id"] == sku_id]
    features_df = build_features(sku_df)
    model = xgb.XGBRegressor()
    model.load_model(f"models/{sku_id}_xgb.json")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df[FEATURE_COLS].iloc[[-1]])

    return {
        "sku_id": sku_id,
        "feature_contributions": dict(zip(FEATURE_COLS, shap_values[0].tolist()))
    }

@app.get("/anomalies/{sku_id}")
def anomalies(sku_id: str):
    from ml.anomaly import detect_anomalies
    sku_df = df[df["sku_id"] == sku_id]
    result = detect_anomalies(sku_df)
    flagged = result[result["is_anomaly"]]
    return flagged.to_dict(orient="records")