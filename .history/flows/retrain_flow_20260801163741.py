from prefect import flow, task
import pandas as pd
from ml.train import train_for_sku

@task
def load_data():
    return pd.read_csv("data/sales_data.csv", parse_dates=["date"])

@task
def retrain_sku(sku_id: str, df: pd.DataFrame):
    return train_for_sku(sku_id, df)

@flow(name="weekly-demand-retrain")
def retrain_all_models():
    df = load_data()
    for sku in df["sku_id"].unique():
        retrain_sku(sku, df)

if __name__ == "__main__":
    retrain_all_models()