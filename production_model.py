# =============================
# production_model.py (UPDATED)
# =============================

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import tensorflow as tf
from keras.layers import Dense

# Patch Dense to ignore unsupported args
original_init = Dense.__init__

def new_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return original_init(self, *args, **kwargs)

Dense.__init__ = new_init
# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEMAND_CSV = os.path.join(PROJECT_ROOT, "dataset/leather_orders.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "logs/lstm_model.keras")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# 1. LSTM DEMAND MODEL
# =============================
class DemandLSTM:

    def __init__(self, seq_len=7):
        self.seq_len = seq_len
        self.mean = 0
        self.std = 1
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.seq_len, 1)),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_len):
            X.append(data[i:i+self.seq_len])
            y.append(data[i+self.seq_len])
        return np.array(X), np.array(y)

    def normalize(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data) + 1e-8
        return (data - self.mean) / self.std

    def denormalize(self, value):
        return value * self.std + self.mean

    def train(self, series, epochs=10):
        if len(series) <= self.seq_len:
            return

        series = np.array(series)
        norm = self.normalize(series)

        X, y = self.prepare_data(norm)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, seq):
        if len(seq) < self.seq_len:
            seq = [seq[-1]] * (self.seq_len - len(seq)) + seq

        seq = np.array(seq[-self.seq_len:])
        seq = (seq - self.mean) / (self.std + 1e-8)
        seq = seq.reshape((1, self.seq_len, 1))

        pred = self.model.predict(seq, verbose=0)[0][0]
        pred = float(self.denormalize(pred))

        return float(np.clip(pred, 50, 1500))

    def save(self):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        self.model.save(MODEL_SAVE_PATH)
        np.save(MODEL_SAVE_PATH + "_stats.npy", [self.mean, self.std])
        logger.info("LSTM saved")

    def load(self):
        if os.path.exists(MODEL_SAVE_PATH):
            self.model = tf.keras.models.load_model(MODEL_SAVE_PATH,compile=False)

            stats = MODEL_SAVE_PATH + "_stats.npy"
            if os.path.exists(stats):
                self.mean, self.std = np.load(stats)

            logger.info("LSTM loaded")
            return True
        return False


# =============================
# =============================
# 2. ML CAPACITY PREDICTOR
# =============================
class CapacityPredictor:

    def __init__(self):
        self.model = self._build_model()
        self.trained = False

    def _build_model(self):
        # Features: tanning_health, drying_health, finishing_health, tanning_base, drying_base, finishing_base
        model = Sequential([
            Input(shape=(6,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_synthetic(self, epochs=5):
        # Generate synthetic data based on linear production limits to pre-train the DNN
        X, y = [], []
        for _ in range(2000):
            bases = [np.random.randint(80, 150) for _ in range(3)]
            healths = [np.random.rand() * 0.8 + 0.2 for _ in range(3)]
            
            cap = min([b * h for b, h in zip(bases, healths)])
            
            X.append(healths + bases)
            y.append(cap)
            
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        self.trained = True

    def predict_capacity(self, machines):
        if not self.trained:
            logger.info("Training capacity ML model on synthetic production data...")
            self.train_synthetic()

        healths_dict = {"tanning": 0.0, "drying": 0.0, "finishing": 0.0}
        bases_dict = {"tanning": 0.0, "drying": 0.0, "finishing": 0.0}

        for m in machines:
            m_type = m.get("type", "").lower()
            h = m.get("health", 0.0) / 100.0
            base = float(m.get("baseProductivity", 100.0))
            
            for k in healths_dict.keys():
                if k in m_type:
                    healths_dict[k] = max(healths_dict[k], h)  # Max health for parallel machines
                    bases_dict[k] += base

        features = [
            healths_dict["tanning"], healths_dict["drying"], healths_dict["finishing"],
            bases_dict["tanning"], bases_dict["drying"], bases_dict["finishing"]
        ]
        
        pred = self.model.predict(np.array([features]), verbose=0)[0][0]
        
        # Estimate the bottleneck stage purely for logging
        b_stage = "none"
        b_val = float('inf')
        for k in healths_dict.keys():
            v = healths_dict[k] * bases_dict[k]
            if v < b_val and v > 0:
                b_val = v
                b_stage = k
                
        return b_stage, max(0.0, float(pred))


# =============================
# 3. REWARD SYSTEM
# =============================
class RewardSystem:

    def compute(self, produced, demand):
        profit = produced * 5
        penalty = max(0, demand-produced) * 2
        return (profit - penalty)/100


# =============================
# 4. MAIN SYSTEM
# =============================
class ProductionSystem:

    def __init__(self):
        self.lstm = DemandLSTM()
        self.capacity_model = CapacityPredictor()
        self.reward = RewardSystem()

        self._init_data()

    def _init_data(self):

        if os.path.exists(DEMAND_CSV):
            try:
                df = pd.read_csv(DEMAND_CSV)
                series = df["quantity"].values
            except:
                series = np.random.randint(300,600,100)
        else:
            series = np.random.randint(300,600,100)

        self.demand_series = series.tolist()

        if not self.lstm.load():
            self.lstm.train(self.demand_series)
            self.lstm.save()

    def predict_demand(self, seq=None):
        if seq is None:
            seq = self.demand_series[-7:]
        return self.lstm.predict(seq)

    def compute_capacity(self, machines):
        return self.capacity_model.predict_capacity(machines)

    def final_output(self, demand, raw_limit, machines):
        _, cap = self.compute_capacity(machines)
        return min(demand, raw_limit, cap), _

    def evaluate(self, produced, demand):
        return self.reward.compute(produced, demand)


# =============================
# TEST
# =============================
if __name__ == "__main__":
    ps = ProductionSystem()

    machines = [
        {"type":"tanning","baseProductivity":120,"health":80},
        {"type":"drying","baseProductivity":100,"health":70},
        {"type":"finishing","baseProductivity":90,"health":85},
    ]

    d = ps.predict_demand()
    print("Demand:", d)

    out, b = ps.final_output(d, 3000, machines)
    print("Output:", out, "Bottleneck:", b)