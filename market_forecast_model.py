"""
Market Forecast Model — 30-day leather market demand forecasting.
Trains an LSTM model and saves it to logs/market_lstm.keras
Generates synthetic dataset if CSV doesn't exist.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import logging

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "logs", "market_lstm.keras")

PRODUCT_TYPES = ['Belt', 'Bag', 'Shoe Leather', 'Jacket Leather', 'Wallet', 'Custom']
SEASONAL = {1: 0.85, 2: 0.80, 3: 0.92, 4: 0.97, 5: 1.00, 6: 1.05,
            7: 0.95, 8: 0.90, 9: 1.00, 10: 1.12, 11: 1.28, 12: 1.45}


def generate_synthetic_market_data(days=500):
    """
    Generate synthetic daily demand for each product.
    Returns a DataFrame with columns: date, product_type, demand.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    records = []
    for product in PRODUCT_TYPES:
        base = np.random.uniform(80, 200)
        # Add a gentle upward trend over the period
        trend = np.linspace(0, 0.3, days)
        # Monthly seasonality
        seasonality = 20 * np.sin(2 * np.pi * np.arange(days) / 30)
        noise = np.random.normal(0, 10, days)
        demand = base + trend * days + seasonality + noise
        demand = np.maximum(10, demand).astype(int)
        for i, d in enumerate(dates):
            records.append({
                'date': d,
                'product_type': product,
                'demand': demand[i]
            })
    df = pd.DataFrame(records)
    # Ensure no duplicates by grouping (should be none, but safe)
    df = df.groupby(['date', 'product_type'], as_index=False)['demand'].first()
    os.makedirs(os.path.join(PROJECT_ROOT, 'dataset'), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, 'dataset', 'leather_market_scenarios.csv')
    df.to_csv(out_path, index=False)
    logger.info(f"Generated {len(df)} rows → {out_path}")
    return df


def train_lstm_model(force_retrain=False):
    """Train LSTM model on historical demand data."""
    if os.path.exists(MODEL_PATH) and not force_retrain:
        logger.info("Loading existing LSTM model")
        return tf.keras.models.load_model(MODEL_PATH)

    # Load or generate dataset
    csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'leather_market_scenarios.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
    else:
        df = generate_synthetic_market_data(500)

    # Pivot to get time series per product
    pivot = df.pivot(index='date', columns='product_type', values='demand').fillna(0)
    # Ensure all product columns exist
    for prod in PRODUCT_TYPES:
        if prod not in pivot.columns:
            pivot[prod] = 0
    pivot = pivot[PRODUCT_TYPES]  # keep consistent order

    data = pivot.values
    # Normalize
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data_norm = (data - mean) / (std + 1e-8)

    seq_len = 30
    X, y = [], []
    for i in range(len(data_norm) - seq_len):
        X.append(data_norm[i:i+seq_len])
        y.append(data_norm[i+seq_len])
    X = np.array(X)
    y = np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(y.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    logger.info(f"Saved LSTM model to {MODEL_PATH}")
    return model


def forecast_30_days():
    """Generate 30-day demand forecast using the trained LSTM model."""
    model = train_lstm_model()
    # Load the most recent data for the last 30 days
    csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'leather_market_scenarios.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        pivot = df.pivot(index='date', columns='product_type', values='demand').fillna(0)
    else:
        df = generate_synthetic_market_data(500)
        pivot = df.pivot(index='date', columns='product_type', values='demand').fillna(0)

    for prod in PRODUCT_TYPES:
        if prod not in pivot.columns:
            pivot[prod] = 0
    pivot = pivot[PRODUCT_TYPES]

    # Use last 30 days as input sequence
    last_30 = pivot.iloc[-30:].values
    mean = last_30.mean(axis=0)
    std = last_30.std(axis=0)
    last_30_norm = (last_30 - mean) / (std + 1e-8)
    X_input = last_30_norm.reshape(1, 30, -1)

    pred_norm = model.predict(X_input, verbose=0)[0]
    pred = pred_norm * std + mean

    # Confidence intervals ±15%
    lower = pred * 0.85
    upper = pred * 1.15

    result = []
    for i, prod in enumerate(PRODUCT_TYPES):
        result.append({
            'product_type': prod,
            'predicted_demand': int(pred[i]),
            'lower_bound': int(lower[i]),
            'upper_bound': int(upper[i])
        })
    return result


if __name__ == "__main__":
    train_lstm_model()
    print("Market LSTM model trained and saved.")
    # Test forecast
    fcast = forecast_30_days()
    print("Sample forecast:", fcast[:2])