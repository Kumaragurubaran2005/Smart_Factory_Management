# =============================
# production_rl.py (FINAL FIXED)
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
class DQNAgent:

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.999,   # 🔥 FIXED
        memory_size: int = 10000,
        batch_size: int = 128,
        target_sync_freq: int = 100,
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
        self._step = 0

        self.model = self._build_model(lr)
        self.target_model = self._build_model(lr)
        self.update_target()

        # 🔥 DEBUG CHECK
        print(f"[DQN INIT] State size: {self.state_size}")
        print(f"[DQN INIT] Model input: {self.model.input_shape}")

    # =============================
    # MODEL
    # =============================
    def _build_model(self, lr):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(self.action_size, activation="linear"),
        ])

        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),  # 🔥 stable
            loss=tf.keras.losses.Huber()
        )
        return model

    # =============================
    # TARGET NETWORK
    # =============================
    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    # =============================
    # ACTION
    # =============================
    def act(self, state, valid_actions=None, eval_mode=False):

        if valid_actions is None or len(valid_actions) == 0:
            return 0   # 🔥 safety

        if not eval_mode and np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        # 🔥 STATE SHAPE SAFETY
        if state.shape[0] != self.state_size:
            raise ValueError(
                f"State size mismatch: expected {self.state_size}, got {state.shape[0]}"
            )

        q = self.model.predict(state[np.newaxis, :], verbose=0)[0]

        masked = np.full(self.action_size, -1e9)
        masked[valid_actions] = q[valid_actions]

        return int(np.argmax(masked))

    # =============================
    # MEMORY
    # =============================
    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    # =============================
    # TRAINING
    # =============================
    def replay(self):

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # 🔥 normalize rewards
        rewards = np.clip(rewards / 100.0, -5, 5)

        target_q = self.model.predict(states, verbose=0)

        # Double DQN
        next_q_main = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_main, axis=1)

        next_q_target = self.target_model.predict(next_states, verbose=0)
        max_next_q = next_q_target[np.arange(self.batch_size), next_actions]

        target_q[np.arange(self.batch_size), actions] = (
            rewards + (1 - dones) * self.gamma * max_next_q
        )

        self.model.fit(states, target_q, epochs=1, verbose=0)

        # 🔥 FIXED epsilon decay (multiplicative)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._step += 1
        if self._step % self.target_sync_freq == 0:
            self.update_target()

    # =============================
    # SAVE / LOAD
    # =============================
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        loaded_model = tf.keras.models.load_model(path)
        if loaded_model.input_shape[-1] != self.state_size:
            raise ValueError(f"Model dimension mismatch. Expected {self.state_size}, got {loaded_model.input_shape[-1]}. Model will be retrained from scratch.")
        self.model = loaded_model
        self.update_target()