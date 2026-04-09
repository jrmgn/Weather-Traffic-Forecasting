"""
forecast_compare.py
Train baseline Linear Regression vs RandomForestRegressor
on delivery delay data and visualize prediction vs actual.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data(path=r"data/merged_hourly.csv"):
    """Load your real dataset."""
    if os.path.exists(path):
        # Using your actual file: merged_hourly.csv
        df = pd.read_csv(path, parse_dates=['timestamp'])
        # Rename to match the script's internal logic
        df = df.rename(columns={'timestamp': 'date', 'delay_minutes': 'delays_minutes'})
        df = df.sort_values('date').reset_index(drop=True)
        return df
    else:
        print(f"Error: {path} not found. Ensure Step 2 was completed.")
        return None

def create_features(df):
    """Create time-based, lag, and weather features."""
    df = df.copy()
    df.set_index('date', inplace=True)

    # Time features
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Lag features (previous hour, previous day)
    df['lag_1'] = df['delays_minutes'].shift(1)
    df['lag_24'] = df['delays_minutes'].shift(24)

    # Rolling features
    df['rolling_6_mean'] = df['delays_minutes'].rolling(window=6).mean().shift(1)

    # Fill NA from shifts
    df = df.fillna(method='bfill').fillna(df.mean())
    return df

def train_test_split_time_series(df, test_size=0.2):
    n = len(df)
    split_at = int(n * (1 - test_size))
    train = df.iloc[:split_at]
    test = df.iloc[split_at:]
    return train, test

def evaluate_model(y_true, y_pred, prefix="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    # Fix: Calculate MSE first, then take the square root for RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) 
    
    r2 = r2_score(y_true, y_pred)
    print(f"{prefix} — MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def main():
    # 1) Load data
    df_raw = load_data()
    if df_raw is None: return

    # 2) Feature engineering
    df = create_features(df_raw)
    target_col = 'delays_minutes'
    # Use all columns (including weather) except target
    feature_cols = [c for c in df.columns if c != target_col]

    # 3) Train/test split
    train_df, test_df = train_test_split_time_series(df, test_size=0.2)
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # 4) Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    evaluate_model(y_test, lr_pred, prefix="Linear Regression")

    # 5) Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    evaluate_model(y_test, rf_pred, prefix="Random Forest")

    # 6) Visualize Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, y_test, label='Actual', color='black', alpha=0.5)
    plt.plot(test_df.index, rf_pred, label='Random Forest', color='red', linestyle='--')
    plt.title("Day 4 Forecast Comparison")
    plt.legend()
    plt.show()

    # 7) Feature importance
    importances = rf.feature_importances_
    plt.figure(figsize=(8,5))
    plt.barh(feature_cols, importances)
    plt.title("What drives delays? (RF Feature Importance)")
    plt.show()

    # 8) Save models
    if not os.path.exists('models'): os.makedirs('models')
    dump(lr, 'delay-weather-dashboard/models/linear_regression.joblib')
    dump(rf, 'delay-weather-dashboard/models/rf_delay.pkl')
    print("Models saved successfully in 'models/' folder.")

if __name__ == "__main__":
    main()