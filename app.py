# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv("driver_delay_weather.csv", parse_dates=["date"], index_col="date")
    return df.asfreq("D").ffill()

df = load_data()
st.title("Smart Logistics — Delay & Traffic Predictor")
st.subheader("Recent data snapshot")
st.dataframe(df.tail(7))

horizon = st.slider("Forecast horizon (days)", 7, 60, 14)

if st.button("Run Delay Forecast"):
    train = df[:-horizon]
    exog_forecast = df[["temp","precip","traffic_index"]].iloc[-horizon:]
    sar = SARIMAX(train["delays"], order=(1,1,1), exog=train[["temp","precip","traffic_index"]], enforce_stationarity=False)
    res = sar.fit(disp=False)
    fc = res.get_forecast(steps=horizon, exog=exog_forecast)
    fig, ax = plt.subplots()
    ax.plot(df.index[-30:], df["delays"][-30:], label="Actual")
    ax.plot(fc.predicted_mean.index, fc.predicted_mean, label="Forecast", linestyle="--")
    st.pyplot(fig)

if st.button("Train Traffic Model"):
    df_ml = df.copy()
    df_ml["lag1"] = df_ml["traffic_index"].shift(1)
    df_ml = df_ml.dropna()
    rf = RandomForestRegressor(n_estimators=50).fit(df_ml[["lag1"]], df_ml["traffic_index"])
    joblib.dump(rf, "rf_traffic.joblib")
    st.success("Trained and saved rf_traffic.joblib")