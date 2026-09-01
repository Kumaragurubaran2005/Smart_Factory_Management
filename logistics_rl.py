# =============================
# logistics_rl.py (UPDATED)
# =============================

import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
from keras.layers import Dense

# Patch Dense to ignore unsupported args
original_init = Dense.__init__

def new_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return original_init(self, *args, **kwargs)

Dense.__init__ = new_init
# =============================
# ENVIRONMENT
# =============================
class TruckEnv:

    N_TRUCKS = 3
    MAX_LOAD = 2000.0
    MAX_CAP = 600.0
    MAX_DIST = 150.0

    def __init__(self):
        import database
        try:
            df_t = database.get_trucks()
            self.trucks = df_t["truck_id"].tolist()[:self.N_TRUCKS]
            while len(self.trucks) < self.N_TRUCKS:
                self.trucks.append(f"FAKE_T{len(self.trucks)}")
        except Exception:
            self.trucks = [f"T{i+1}" for i in range(self.N_TRUCKS)]
            
        self.state_size = 1 + self.N_TRUCKS * 3
        self.action_size = self.N_TRUCKS

    def reset(self, production_output):
        self.remaining_load = float(production_output)

        import database
        try:
            df_t = database.get_trucks()
            cap_map = {row["truck_id"]: row["capacity"] for _, row in df_t.iterrows()}
        except Exception:
            cap_map = {}

        self.truck_status = {}
        for t in self.trucks:
            real_cap = float(cap_map.get(t, 500.0))
            self.truck_status[t] = {
                "available": True,
                "capacity": real_cap,
                "distance": 60.0, # fixed static standard route distance
                "fuel_eff": 1.0 # standard fuel efficiency baseline
            }

        return self._state()

    def _state(self):
        load_norm = self.remaining_load / self.MAX_LOAD

        features = []
        for t in self.trucks:
            ts = self.truck_status[t]
            features.extend([
                float(ts["available"]),
                ts["capacity"] / self.MAX_CAP,
                ts["distance"] / self.MAX_DIST,
            ])

        return np.array([load_norm] + features, dtype=np.float32)

    def get_valid_actions(self):
        return [
            i for i,t in enumerate(self.trucks)
            if self.truck_status[t]["available"]
        ]

    def step(self, action):

        truck_id = self.trucks[action]
        truck = self.truck_status[truck_id]

        if not truck["available"]:
            return self._state(), -10.0, True, {}

        delivered = min(truck["capacity"], self.remaining_load)
        self.remaining_load -= delivered

        fuel_cost = truck["distance"] * truck["fuel_eff"]
        backlog_penalty = self.remaining_load * 0.05

        reward = delivered - fuel_cost - backlog_penalty
        reward = reward / 100.0   # normalize

        truck["available"] = False

        done = (
            self.remaining_load <= 0 or
            all(not t["available"] for t in self.truck_status.values())
        )

        if self.remaining_load <= 0:
            reward += 5.0   # completion bonus

        # 🔥 Explainable AI
        info = {
            "truck": truck_id,
            "delivered": delivered,
            "remaining": self.remaining_load,
            "distance": truck["distance"],
            "fuel_eff": truck["fuel_eff"],
            "reward": reward
        }

        return self._state(), reward, done, info


# =============================
# AGENT
# =============================
class TruckAgent:

    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        memory_size=5000,
        batch_size=64,
        target_sync_freq=100
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_freq = target_sync_freq

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=memory_size)
        self.step_count = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size)
        ])

        model.compile(
            optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
            loss='mse'
        )
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, valid_actions, eval_mode=False):

        if not valid_actions:
            return 0

        if not eval_mode and np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        q = self.model.predict(state[np.newaxis,:], verbose=0)[0]

        masked = np.full(self.action_size, -1e9)
        masked[valid_actions] = q[valid_actions]

        return int(np.argmax(masked))

    def remember(self, s,a,r,ns,d):
        self.memory.append((s,a,r,ns,d))

    def replay(self):

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch]) / 100.0
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        target_q = self.model.predict(states, verbose=0)

        next_q_main = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_main, axis=1)

        next_q_target = self.target_model.predict(next_states, verbose=0)
        max_next_q = next_q_target[np.arange(self.batch_size), next_actions]

        target_q[np.arange(self.batch_size), actions] = (
            rewards + (1 - dones) * self.gamma * max_next_q
        )

        self.model.fit(states, target_q, epochs=1, verbose=0)

        # epsilon decay (stable)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.step_count += 1
        if self.step_count % self.target_sync_freq == 0:
            self.update_target()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        self.update_target()