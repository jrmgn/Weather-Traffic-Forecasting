# data_gen.py
import numpy as np
import pandas as pd

def generate_synthetic_data(n_days=730, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    base = 5 + 0.01 * np.arange(n_days) + 3*np.sin(2 * np.pi * dates.dayofweek / 7)
    temp = 25 + 5*np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0,1,n_days)
    precip = np.clip(np.random.gamma(0.5, 1.0, n_days) - 0.6, 0, None)
    wind = np.abs(np.random.normal(3,1,n_days))
    traffic_index = np.clip(40 + 10*np.sin(2*np.pi*np.arange(n_days)/24) + np.random.normal(0,8,n_days), 5, 100)
    delays = np.maximum(0, base + 0.2*precip + 0.02*(traffic_index) + np.random.normal(0,1.5,n_days))
    df = pd.DataFrame({
        "date": dates,
        "delays": delays,
        "temp": temp,
        "precip": precip,
        "wind": wind,
        "traffic_index": traffic_index
    }).set_index("date")
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("driver_delay_weather.csv")
    print("Saved driver_delay_weather.csv")