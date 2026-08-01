import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.xgboost
import numpy as np

from ml.features import build_features

FEATURE_COLS = ["lag_1", "lag_7", "lag_14", "rolling_mean_7",
                 "rolling_std_7", "day_of_week", "month", "is_weekend"]

def train_prophet(df: pd.DataFrame) -> Prophet:
    prophet_df = df[["date", "units_sold"]].rename(
        columns={"date": "ds", "units_sold": "y"}
    )
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    return model

def train_xgboost(df: pd.DataFrame):
    features_df = build_features(df)
    X = features_df[FEATURE_COLS]
    y = features_df["units_sold"]

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, rmse, features_df

def train_for_sku(sku_id: str, df: pd.DataFrame):
    sku_df = df[df["sku_id"] == sku_id].copy()

    mlflow.set_experiment("demand-forecasting")
    with mlflow.start_run(run_name=f"train_{sku_id}"):
        prophet_model = train_prophet(sku_df)
        xgb_model, rmse, features_df = train_xgboost(sku_df)

        mlflow.log_param("sku_id", sku_id)
        mlflow.log_param("features_used", FEATURE_COLS)
        mlflow.log_param("n_training_rows", len(sku_df))
        mlflow.log_metric("xgb_rmse", rmse)
        mlflow.xgboost.log_model(xgb_model, "xgb_model")

        print(f"[{sku_id}] XGBoost RMSE: {rmse:.2f}")

        import os
        os.makedirs("models", exist_ok=True)
        xgb_model.save_model(f"models/{sku_id}_xgb.json")

    return prophet_model, xgb_model

if __name__ == "__main__":
    df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
    for sku in df["sku_id"].unique():
        train_for_sku(sku, df)