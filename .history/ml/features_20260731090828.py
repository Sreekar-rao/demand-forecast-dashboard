import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: date, sku_id, units_sold — one SKU at a time works best."""
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    # lag features — yesterday, last week, two weeks ago
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["units_sold"].shift(lag)

    # rolling stats
    df["rolling_mean_7"] = df["units_sold"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["units_sold"].shift(1).rolling(7).std()

    # calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df.dropna().reset_index(drop=True)