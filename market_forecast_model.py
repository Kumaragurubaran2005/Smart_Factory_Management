import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "logs", "market_lstm.keras")

PRODUCT_TYPES = ['Belt', 'Bag', 'Shoe Leather', 'Jacket Leather', 'Wallet', 'Custom']

# ─── Global cached model and normalization stats ─────────────────────────────
_cached_model = None
_cached_mean = None
_cached_std = None




def train_lstm_model(force_retrain=False):
    """Train the LSTM model only if missing or forced, then cache it."""
    global _cached_model, _cached_mean, _cached_std

    if not force_retrain and _cached_model is not None:
        return _cached_model

    csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'leather_market_scenarios.csv')
    if not os.path.exists(csv_path):
        df = generate_synthetic_market_data()
    else:
        df = pd.read_csv(csv_path, parse_dates=['date'])

    pivot = df.pivot(index='date', columns='product_type', values='demand').fillna(0)
    pivot = pivot[PRODUCT_TYPES]
    data = pivot.values

    _cached_mean = data.mean(axis=0)
    _cached_std = data.std(axis=0)
    data_norm = (data - _cached_mean) / (_cached_std + 1e-8)

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
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(seq_len, X.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(y.shape[1])
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)

    model.fit(X_train, y_train, epochs=80, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[early_stop, reduce_lr], verbose=1)

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_denorm = y_pred * _cached_std + _cached_mean
    y_test_denorm = y_test * _cached_std + _cached_mean
    mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
    mape = np.mean(np.abs((y_test_denorm - y_pred_denorm) / (y_test_denorm + 1e-8))) * 100
    print("\n📊 MARKET MODEL EVALUATION")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("MAPE:", round(mape, 2), "%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    _cached_model = model
    return model


def load_cached_model():
    """Load the model from disk once and cache it along with normalization stats."""
    global _cached_model, _cached_mean, _cached_std
    if _cached_model is not None:
        return

    if os.path.exists(MODEL_PATH):
        logger.info("Loading pre-trained market LSTM model from disk")
        _cached_model = tf.keras.models.load_model(MODEL_PATH)
        # Recompute normalization stats from the dataset
        csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'leather_market_scenarios.csv')
        if not os.path.exists(csv_path):
            df = generate_synthetic_market_data()
        else:
            df = pd.read_csv(csv_path, parse_dates=['date'])
        pivot = df.pivot(index='date', columns='product_type', values='demand').fillna(0)
        pivot = pivot[PRODUCT_TYPES]
        data = pivot.values
        _cached_mean = data.mean(axis=0)
        _cached_std = data.std(axis=0)
    else:
        train_lstm_model()


def forecast_30_days():
    """Return 30‑day forecast using the cached LSTM model."""
    load_cached_model()  # ensures model is loaded once

    csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'leather_market_scenarios.csv')
    df = pd.read_csv(csv_path, parse_dates=['date'])
    pivot = df.pivot(index='date', columns='product_type', values='demand').fillna(0)
    pivot = pivot[PRODUCT_TYPES]

    last_30 = pivot.iloc[-30:].values
    last_30_norm = (last_30 - _cached_mean) / (_cached_std + 1e-8)
    X_input = last_30_norm.reshape(1, 30, -1)

    pred_norm = _cached_model.predict(X_input, verbose=0)[0]
    pred = pred_norm * _cached_std + _cached_mean

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
    print("Sample forecast:", forecast_30_days()[:2])