import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sales_data(n_skus=5, n_days=730, seed=42):
    np.random.seed(seed)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    rows = []
    for sku_id in range(1, n_skus + 1):
        base_demand = np.random.randint(50, 200)
        trend = np.linspace(0, np.random.uniform(-20, 40), n_days)

        for i, date in enumerate(dates):
            # weekly seasonality (weekends dip for B2B-style demand)
            weekly = 15 * np.sin(2 * np.pi * date.weekday() / 7)
            # yearly seasonality (holiday bump in Nov/Dec)
            yearly = 30 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            noise = np.random.normal(0, 8)

            demand = base_demand + trend[i] + weekly + yearly + noise

            # inject anomalies ~1.5% of the time — spikes or drops
            is_injected_anomaly = np.random.random() < 0.015
            if is_injected_anomaly:
                demand *= np.random.choice([2.5, 0.2])  # spike or crash

            rows.append({
                "date": date,
                "sku_id": f"SKU-{sku_id:03d}",
                "units_sold": max(0, round(demand)),
                "injected_anomaly": is_injected_anomaly  # ground truth, for evaluation only
            })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv("data/sales_data.csv", index=False)
    print(df.head(15))
    print(f"\nGenerated {len(df)} rows, {df['injected_anomaly'].sum()} injected anomalies")