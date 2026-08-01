import pandas as pd
from sklearn.ensemble import IsolationForest
from ml.features import build_features

def detect_anomalies(df: pd.DataFrame, contamination=0.02) -> pd.DataFrame:
    features_df = build_features(df)
    feature_cols = ["units_sold", "rolling_mean_7", "rolling_std_7"]

    model = IsolationForest(contamination=contamination, random_state=42)
    features_df["anomaly_score"] = model.fit_predict(features_df[feature_cols])
    features_df["is_anomaly"] = features_df["anomaly_score"] == -1

    return features_df[["date", "sku_id", "units_sold", "is_anomaly"]]