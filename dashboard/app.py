import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")
st.title("📦 Demand Forecasting & Anomaly Detection")

API_URL = "http://localhost:8000"

df = pd.read_csv("data/sales_data.csv", parse_dates=["date"])
sku_list = sorted(df["sku_id"].unique())
selected_sku = st.sidebar.selectbox("Select SKU", sku_list)

sku_df = df[df["sku_id"] == selected_sku].sort_values("date")

col1, col2 = st.columns([3, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sku_df["date"], y=sku_df["units_sold"],
                              mode="lines", name="Actual Demand"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("7-Day Forecast")
    try:
        pred = requests.get(f"{API_URL}/predict/{selected_sku}").json()
        st.metric("Next Day Forecast", f"{pred['forecast_next_day']:.0f} units")
    except Exception as e:
        st.warning(f"API not reachable: {e}")

st.subheader("🚨 Flagged Anomalies")
try:
    anomalies = requests.get(f"{API_URL}/anomalies/{selected_sku}").json()
    if anomalies:
        st.dataframe(pd.DataFrame(anomalies))
    else:
        st.success("No anomalies detected in this window.")
except Exception as e:
    st.warning(f"API not reachable: {e}")

st.subheader("🔍 Why This Forecast? (SHAP Explanation)")
try:
    explanation = requests.get(f"{API_URL}/explain/{selected_sku}").json()
    contrib_df = pd.DataFrame(
        explanation["feature_contributions"].items(),
        columns=["Feature", "Contribution"]
    ).sort_values("Contribution")
    fig2 = go.Figure(go.Bar(x=contrib_df["Contribution"], y=contrib_df["Feature"],
                             orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.warning(f"API not reachable: {e}")