from sqlalchemy import create_engine, Column, String, Float, Date, Boolean, Integer
from sqlalchemy.orm import declarative_base
import pandas as pd

Base = declarative_base()

class SalesRecord(Base):
    __tablename__ = "sales_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, index=True)
    sku_id = Column(String, index=True)
    units_sold = Column(Float)

DATABASE_URL = "postgresql://postgres:yourpassword@localhost:5432/demand_forecast"
engine = create_engine(DATABASE_URL)

def load_csv_to_db(csv_path="data/sales_data.csv"):
    Base.metadata.create_all(engine)
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df[["date", "sku_id", "units_sold"]].to_sql(
        "sales_data", engine, if_exists="replace", index=False
    )
    print(f"Loaded {len(df)} rows into Postgres")

if __name__ == "__main__":
    load_csv_to_db()