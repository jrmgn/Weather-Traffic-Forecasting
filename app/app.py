import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Delay & Weather Dashboard", layout="wide")
st.title("Delay Prediction & Weather Trends")

# --- STRICT PATH CHECK ---
# Since you run from 'delay-weather-dashboard', these are the paths:
data_path = "data/merged_hourly.csv"
forecast_path = "data/forecast.csv"

# Check if files exist to give you a helpful message
if not os.path.exists(data_path):
    st.error(f"Missing file: {data_path}. Check your 'data' folder!")
elif not os.path.exists(forecast_path):
    st.error(f"Missing file: {forecast_path}. Run your Prophet code first!")
else:
    # 1. Load Data
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    forecast = pd.read_csv(forecast_path, parse_dates=['ds'])
    
    # 2. Controls
    st.sidebar.header("Controls")
    # Convert timestamp to date for the slider
    min_date = df.timestamp.min().date()
    max_date = df.timestamp.max().date()
    start = st.sidebar.date_input("Start date", min_date)
    end = st.sidebar.date_input("End date", max_date)
    
    # 3. Filter data
    mask = (df.timestamp.dt.date >= start) & (df.timestamp.dt.date <= end)
    vis = df.loc[mask].copy()
    
    fc_mask = (forecast.ds.dt.date >= start) & (forecast.ds.dt.date <= end)
    fc = forecast.loc[fc_mask]
    
    # 4. Plot (Following your strict instructions)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vis.timestamp, y=vis.delay_minutes, name='Actual Delays', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=fc.ds, y=fc.yhat, name='Forecast Delays', line=dict(color='red', dash='dash')))
    
    # Note: Using 'rainfall_mm' from your actual dataset
    fig.add_trace(go.Bar(x=vis.timestamp, y=vis.rainfall_mm, name='Precipitation', yaxis='y2', opacity=0.35, marker_color='lightblue'))
    
    fig.update_layout(
        yaxis=dict(title="Delay (Minutes)"),
        yaxis2=dict(overlaying='y', side='right', title='Precipitation (mm)'),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. Summary
    st.success("Dashboard Loaded Successfully!")