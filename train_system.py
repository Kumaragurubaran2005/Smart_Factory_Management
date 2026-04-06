from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense

# Patch Dense to ignore unsupported args
original_init = Dense.__init__

def new_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return original_init(self, *args, **kwargs)

Dense.__init__ = new_init

from baselines import (
    greedy_production_action,
    random_production_action,
    greedy_truck_action,
    random_truck_action
)

# ── Local modules ─────────────────────────────────────────────────────────────
from production_model import ProductionSystem
from logistics_rl import TruckEnv, TruckAgent
from production_rl import DQNAgent
# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CFG = {
    # Paths
    "log_dir": "./logs",
    "prod_best_path": "./logs/prod_best.keras",
    "logistics_best_path": "./logs/logistics_best.keras",
    "train_log_path": "./logs/train_log.csv",
    "curve_path": "./logs/learning_curves.png",

    # Reproducibility
    "seed": 42,

    # Production environment
    "prod_workers": ["W1", "W2", "W3", "W4"],
    "prod_machines": ["tanning", "drying", "finishing"],
    "prod_max_steps": 30,
    "prod_episodes": 3000,

    # Logistics environment
    "log_episodes": 2000,
    "log_max_steps": 10,

    # Shared agent hypers
    "gamma": 0.95,
    "lr": 1e-3,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "target_sync_freq": 100,

    # Production agent memory / batch
    "prod_memory": 10_000,
    "prod_batch": 64,

    # Logistics agent memory / batch
    "log_memory": 5_000,
    "log_batch": 32,

    # Evaluation
    "eval_freq": 100,
    "eval_episodes": 10,

    # Early stopping
    "patience": 500,

    # Reward shaping tweaks
    "step_bonus": 0.5,          # small positive reward per step to encourage longer episodes
}


# ──────────────────────────────────────────────────────────────────────────────
# PRODUCTION ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerProfile:
    age: int
    experience: float   # 0..1
    fatigue: float      # 0..1
    skill: float        # 0..1


class ProductionEnv:
    """
    RL environment for worker → machine scheduling.

    State  (9-dim):  [demand_norm, tanning_health, drying_health,
                      finishing_health, w1_avail, w2_avail, w3_avail,
                      w4_avail, avg_skill]

    Action (12-dim): worker × machine (4 workers × 3 machine types)
    """

    MAX_DEMAND = 1500.0
    MIN_MACHINE_HEALTH = 0.5
    MAX_FATIGUE = 1.0

    def __init__(self, system: ProductionSystem):
        import database
        try:
            # Pull exactly 4 present workers based on real skill
            df_w = database.get_worker_skills(database.get_available_workers())
            if df_w is not None and not df_w.empty:
                df_w = df_w.sort_values(by="overall_skill", ascending=False).head(4)
                self.workers = df_w["worker_id"].tolist()
                while len(self.workers) < 4:
                    self.workers.append(f"FAKE_W{len(self.workers)}")
            else:
                self.workers = CFG["prod_workers"]

            # Pull exactly 3 real machines types
            df_m = database.get_machines()
            if df_m is not None and not df_m.empty:
                self.machines = df_m["type"].tolist()[:3]
                while len(self.machines) < 3:
                    self.machines.append(f"FAKE_M{len(self.machines)}")
            else:
                self.machines = CFG["prod_machines"]
        except Exception:
            self.workers = CFG["prod_workers"]
            self.machines = CFG["prod_machines"]

        self.system = system
        self.n_w = len(self.workers)
        self.n_m = len(self.machines)
        self.use_fatigue = True
        self.use_health_penalty = True
        self.use_reward_shaping = True

        # EXPANDED RESEARCH-GRADE STATE TENSOR:
        # [demand_norm] + [healths]*M + [wears]*M + [productivities]*M + [avail]*W + [fatigues]*W + [skills]*W + [shifts]*W + [supply_lim, prev_out, downtime, avg_skill]
        self.state_size = 1 + (self.n_m * 3) + (self.n_w * 4) + 4
        
        self.action_size = self.n_w * self.n_m
        self.action_map = [(w, m) for w in self.workers for m in self.machines]

        # Cache demand series for episode resets
        self._series = list(getattr(self.system, "demand_series", []) or np.random.randint(300, 600, 200))

        self._machine_health: dict[str, float] = {}
        self._avail_workers: list[str] = []
        self._worker_profiles: dict[str, WorkerProfile] = {}
        self._demand: float = 0.0
        self._last_explanation: dict[str, Any] = {}

    def get_valid_actions(self) -> list[int]:
        return [i for i, (w, _) in enumerate(self.action_map) if w in self._avail_workers]

    def reset(self):
        series = self._series
        if len(series) < 8:
            series = list(np.random.randint(300, 600, 200))

        idx = random.randint(0, len(series) - 8)
        seq = series[idx: idx + 7]
        pred_dem = float(np.clip(self.system.predict_demand(seq), 50, self.MAX_DEMAND))

        import database
        from database import get_worker_fatigue   # FIXED: use database function, not app
            
        real_health_base = database.get_machine_health()
        df_m = database.get_machines()
        type_health = {}
        if df_m is not None and not df_m.empty:
            for _, row in df_m.iterrows():
                h = real_health_base.get(row["machine_id"], 0.8)
                h = max(0.1, min(1.0, float(h)))   # FIXED: clamp health to [0.1, 1.0]
                type_health[row["type"]] = h
        else:
            for m in self.machines:
                type_health[m] = 0.8

        self._machine_health = {m: float(type_health.get(m, 0.8)) for m in self.machines}

        raw_limit = random.randint(300, 800)
        self._supply_limit = float(raw_limit)
        self._previous_output = getattr(self, "_previous_output", 0.0) # mock prev
        self._downtime = float(np.random.rand() * 10.0)
        
        mach_list = [
            {"type": m, "baseProductivity": 100, "health": self._machine_health[m] * 100}
            for m in self.machines
        ]
        output, _ = self.system.final_output(pred_dem, raw_limit, mach_list)
        self._demand = float(max(0.0, output))

        n_avail = random.randint(2, self.n_w)
        self._avail_workers = random.sample(self.workers, n_avail)

        df_w = database.get_worker_skills(self.workers)
        skill_map = {}
        if df_w is not None and not df_w.empty:
            for _, w_row in df_w.iterrows():
                skill_map[w_row["worker_id"]] = w_row["overall_skill"] / 100.0

        self._worker_profiles = {}
        for w in self.workers:
            base_skill = float(skill_map.get(w, 0.8))
            try:
                base_fatigue = float(get_worker_fatigue(w))
            except Exception:
                base_fatigue = 0.0
            self._worker_profiles[w] = WorkerProfile(
                age=35,
                experience=base_skill,
                fatigue=base_fatigue,
                skill=base_skill,
            )

        self._last_explanation = {}
        return self._state()

    def _state(self) -> np.ndarray:
        d_norm = np.clip(self._demand / self.MAX_DEMAND, 0.0, 1.0)
        healths = [self._machine_health[m] for m in self.machines]
        wears = [1.0 - h for h in healths]
        productivities = [(h * 100.0)/100.0 for h in healths]  # normalized generic proxy
        
        avail = [1.0 if w in self._avail_workers else 0.0 for w in self.workers]
        fatigues = [self._worker_profiles[w].fatigue for w in self.workers]
        skills = [self._worker_profiles[w].skill for w in self.workers]
        shifts = [(self._worker_profiles[w].fatigue * 12.0)/12.0 for w in self.workers] # simulated shift load
        
        avg_skill = float(np.mean(skills))
        
        state_list = [d_norm] + healths + wears + productivities + avail + fatigues + skills + shifts + [
            self._supply_limit / 1000.0, 
            np.clip(self._previous_output / 1000.0, 0, 1), 
            self._downtime / 24.0, 
            avg_skill
        ]
        return np.array(state_list, dtype=np.float32)

    def step(self, action: int):
        worker, machine = self.action_map[action]

        if worker not in self._avail_workers:
            done = (len(self._avail_workers) == 0) or (self._demand <= 0)
            info = {
                "reason": "worker_not_available",
                "worker": worker,
                "machine": machine,
            }
            return self._state(), -50.0, done, info

        profile = self._worker_profiles[worker]
        health = self._machine_health[machine]

        # Productivity model influenced by skill, experience, fatigue, and machine health.
        productivity = (
            100.0
            * profile.skill
            * profile.experience
            * health
            * (1.0 - profile.fatigue)
        )

        prev_demand = self._demand
        self._demand = max(0.0, self._demand - productivity)
        demand_reduction = prev_demand - self._demand

        reward = 0.0

        if self.use_reward_shaping:
            reward += 0.20 * productivity
            reward += 0.05 * demand_reduction
        else:
            reward += productivity * 0.1

        # ======================= FIXED PENALTIES =======================
        if self.use_fatigue:
            reward -= 2.0 * profile.fatigue           # was 20.0
            if profile.fatigue > 0.8:
                reward -= 5.0                         # was 50.0

        if self.use_health_penalty:
            reward -= 2.0 * (1.0 - health)            # was 5.0

        # Optional small survival bonus to encourage longer episodes
        reward += CFG.get("step_bonus", 0.5)

        # Physical Machine Degradation Model
        wear = (productivity * 0.0002) + (profile.fatigue * 0.005)
        self._machine_health[machine] = max(self.MIN_MACHINE_HEALTH, health - wear)
        profile.fatigue = min(self.MAX_FATIGUE, profile.fatigue + 0.05)   # was 0.15
        self._avail_workers.remove(worker)

        done = (self._demand <= 0) or (len(self._avail_workers) == 0)
        if len(self._avail_workers) == 0 and self._demand > 0:
            reward -= 10.0

        explanation = {
            "selected_worker": worker,
            "selected_machine": machine,
            "worker_age": profile.age,
            "worker_experience": round(profile.experience, 4),
            "worker_fatigue": round(profile.fatigue, 4),
            "worker_skill": round(profile.skill, 4),
            "machine_health": round(health, 4),
            "estimated_productivity": round(productivity, 4),
            "demand_before": round(prev_demand, 4),
            "demand_after": round(self._demand, 4),
            "reward": round(float(reward), 4),
            "reasoning": (
                "Selected this worker-machine pair because the worker was available, "
                "the machine was healthy enough, and the estimated productivity was high."
            ),
        }
        self._last_explanation = explanation

        return self._state(), float(reward), done, explanation


# ──────────────────────────────────────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self):
        os.makedirs(CFG["log_dir"], exist_ok=True)

        print("[*] Initialising ProductionSystem ...")
        self.prod_system = ProductionSystem()

        self.prod_env = ProductionEnv(self.prod_system)
        self.truck_env = TruckEnv()

        self.prod_agent = DQNAgent(
            state_size=self.prod_env.state_size,
            action_size=self.prod_env.action_size,
            lr=CFG["lr"],
            gamma=CFG["gamma"],
            epsilon_start=CFG["epsilon_start"],
            epsilon_min=CFG["epsilon_min"],
            epsilon_decay=CFG["epsilon_decay"],
            memory_size=CFG["prod_memory"],
            batch_size=CFG["prod_batch"],
            target_sync_freq=CFG["target_sync_freq"],
        )

        self.truck_agent = TruckAgent(
            state_size=self.truck_env.state_size,
            action_size=self.truck_env.action_size,
            lr=CFG["lr"],
            gamma=CFG["gamma"],
            epsilon_start=CFG["epsilon_start"],
            epsilon_min=CFG["epsilon_min"],
            epsilon_decay=CFG["epsilon_decay"],
            memory_size=CFG["log_memory"],
            batch_size=CFG["log_batch"],
            target_sync_freq=CFG["target_sync_freq"],
        )

        # Resume from saved checkpoints when available.
        if os.path.exists(CFG["prod_best_path"]):
            print(f"[INFO] Loading production model: {CFG['prod_best_path']}")
            try:
                self.prod_agent.load(CFG["prod_best_path"])
            except Exception as exc:
                print(f"[WARN] Could not load production model: {exc}")

        if os.path.exists(CFG["logistics_best_path"]):
            print(f"[INFO] Loading logistics model: {CFG['logistics_best_path']}")
            try:
                self.truck_agent.load(CFG["logistics_best_path"])
            except Exception as exc:
                print(f"[WARN] Could not load logistics model: {exc}")

        self.best_prod_eval = -np.inf
        self.best_log_eval = -np.inf
        self.best_joint_eval = -np.inf

        self.prod_rewards: list[float] = []
        self.log_rewards: list[float] = []
        self.joint_evals: list[dict[str, float]] = []
        self._log_rows: list[dict[str, Any]] = []

    def evaluate_production_policy(self, policy="rl", episodes=50):
        results = []

        for _ in range(episodes):
            state = self.prod_env.reset()
            total_reward = 0.0

            for _ in range(CFG["prod_max_steps"]):
                valid = self.prod_env.get_valid_actions()
                if not valid:
                    break

                if policy == "rl":
                    action = self.prod_agent.act(state, valid, eval_mode=True)

                elif policy == "greedy":
                    action = greedy_production_action(self.prod_env)

                elif policy == "random":
                    action = random_production_action(self.prod_env)

                else:
                    raise ValueError("Unknown policy")

                state, r, done, _ = self.prod_env.step(action)
                total_reward += r

                if done:
                    break

            results.append(total_reward)

        return float(np.mean(results))
    def evaluate_logistics_policy(self, policy="rl", episodes=50):
        rewards = []

        for _ in range(episodes):
            load = float(np.random.randint(200, 1500))
            state = self.truck_env.reset(load)
            ep_r = 0.0

            for _ in range(CFG["log_max_steps"]):
                valid = self.truck_env.get_valid_actions()
                if not valid:
                    break

                if policy == "rl":
                    action = self.truck_agent.act(state, valid, eval_mode=True)

                elif policy == "greedy":
                    action = greedy_truck_action(self.truck_env)

                elif policy == "random":
                    action = random_truck_action(self.truck_env)

                state, r, done, _ = self.truck_env.step(action)
                ep_r += r

                if done:
                    break

            rewards.append(ep_r)

        return float(np.mean(rewards))
    def run_ablation(self):
        print("\n==============================")
        print(" ABLATION STUDY")
        print("==============================")

        configs = [
            ("Full Model", True, True, True),
            ("No Fatigue", False, True, True),
            ("No Health Penalty", True, False, True),
            ("No Reward Shaping", True, True, False),
        ]

        results = []

        for name, fatigue, health, shaping in configs:
            print(f"\nRunning: {name}")

            self.prod_env.use_fatigue = fatigue
            self.prod_env.use_health_penalty = health
            self.prod_env.use_reward_shaping = shaping

            score = self.evaluate_production_policy("rl", episodes=30)

            results.append((name, score))
            print(f"Score: {score:.2f}")

        print("\n=== ABLATION RESULTS ===")
        for r in results:
            print(f"{r[0]:25s} : {r[1]:.2f}")
        # ── helpers ──────────────────────────────────────────────────────────────

    def _safe_act(self, agent, state, valid, eval_mode: bool = False):
        if not valid:
            return None
        return agent.act(state, valid, eval_mode=eval_mode)

    # ── Production training ───────────────────────────────────────────────────

    def _run_prod_episode(self) -> float:
        state = self.prod_env.reset()
        ep_reward = 0.0

        for _ in range(CFG["prod_max_steps"]):
            valid = self.prod_env.get_valid_actions()
            if not valid:
                break

            action = self.prod_agent.act(state, valid)
            next_s, reward, done, _ = self.prod_env.step(action)
            self.prod_agent.remember(state, action, reward, next_s, done)

            ep_reward += reward
            state = next_s
            if done:
                break

        self.prod_agent.replay()
        return ep_reward

    def _eval_production(self, n: int | None = None) -> float:
        n = n or CFG["eval_episodes"]
        rewards = []
        for _ in range(n):
            state = self.prod_env.reset()
            ep_r = 0.0
            for _ in range(CFG["prod_max_steps"]):
                valid = self.prod_env.get_valid_actions()
                if not valid:
                    break
                action = self.prod_agent.act(state, valid, eval_mode=True)
                state, r, done, _ = self.prod_env.step(action)
                ep_r += r
                if done:
                    break
            rewards.append(ep_r)
        return float(np.mean(rewards))

    def train_production(self, n_episodes: int) -> None:
        print(f"\n{'=' * 60}")
        print(f"  PHASE 1 - Production DQN   ({n_episodes} episodes)")
        print(f"{'=' * 60}")

        no_improve = 0
        t0 = time.time()

        for ep in range(n_episodes):
            ep_reward = self._run_prod_episode()
            self.prod_rewards.append(ep_reward)

            if (ep + 1) % CFG["eval_freq"] == 0:
                eval_r = self._eval_production()
                elapsed = time.time() - t0
                eps_per_sec = (ep + 1) / max(1e-6, elapsed)
                eta = (n_episodes - ep - 1) / max(1e-6, eps_per_sec)

                print(
                    f"  [Prod] Ep {ep + 1:5d}/{n_episodes} | "
                    f"Train {ep_reward:8.1f} | Eval {eval_r:8.1f} | "
                    f"eps {self.prod_agent.epsilon:.3f} | ETA {eta / 60:.1f} min"
                )

                if eval_r > self.best_prod_eval:
                    self.best_prod_eval = eval_r
                    self.prod_agent.save(CFG["prod_best_path"])
                    print(f"    [BEST] New best production model saved (eval={eval_r:.1f})")
                    no_improve = 0
                else:
                    no_improve += CFG["eval_freq"]

                if no_improve >= CFG["patience"]:
                    print(f"  [STOP] Early stopping (no improvement for {CFG['patience']} episodes)")
                    break

    # ── Logistics training ────────────────────────────────────────────────────

    def _run_log_episode(self, load: float) -> float:
        state = self.truck_env.reset(load)
        ep_reward = 0.0

        for _ in range(CFG["log_max_steps"]):
            valid = self.truck_env.get_valid_actions()
            if not valid:
                break
            action = self.truck_agent.act(state, valid)
            next_s, reward, done ,info= self.truck_env.step(action)
            self.truck_agent.remember(state, action, reward, next_s, done)
            ep_reward += reward
            state = next_s
            if done:
                break

        self.truck_agent.replay()
        return ep_reward

    def _eval_logistics(self, n: int | None = None) -> float:
        n = n or CFG["eval_episodes"]
        rewards = []
        for _ in range(n):
            load = float(np.random.randint(200, 1500))
            state = self.truck_env.reset(load)
            ep_r = 0.0
            for _ in range(CFG["log_max_steps"]):
                valid = self.truck_env.get_valid_actions()
                if not valid:
                    break
                action = self.truck_agent.act(state, valid, eval_mode=True)
                state, r, done ,info= self.truck_env.step(action)
                ep_r += r
                if done:
                    break
            rewards.append(ep_r)
        return float(np.mean(rewards))

    # ── Joint evaluation ──────────────────────────────────────────────────────

    def joint_eval(self, n: int | None = None) -> dict[str, float]:
        """
        Full pipeline: Production env → total_output → Logistics env
        Returns combined reward and fill rate.
        """
        n = n or CFG["eval_episodes"]

        prod_rewards, log_rewards, fill_rates = [], [], []

        for _ in range(n):
            # Production
            state = self.prod_env.reset()
            prod_r = 0.0
            total_prod = 0.0

            for _ in range(CFG["prod_max_steps"]):
                valid = self.prod_env.get_valid_actions()
                if not valid:
                    break
                action = self.prod_agent.act(state, valid, eval_mode=True)
                state, r, done, info = self.prod_env.step(action)
                prod_r += r
                total_prod += float(info.get("estimated_productivity", 0.0))
                if done:
                    break

            # Logistics
            load = max(50.0, total_prod)
            state = self.truck_env.reset(load)
            log_r = 0.0

            for _ in range(CFG["log_max_steps"]):
                valid = self.truck_env.get_valid_actions()
                if not valid:
                    break
                action = self.truck_agent.act(state, valid, eval_mode=True)
                state, r, done, info = self.truck_env.step(action)
                log_r += r
                if done:
                    break

            delivered = load - getattr(self.truck_env, "remaining_load", load)
            fill_rate = (delivered / load) if load > 0 else 0.0

            prod_rewards.append(prod_r)
            log_rewards.append(log_r)
            fill_rates.append(fill_rate)

        prod_mean = float(np.mean(prod_rewards))
        log_mean = float(np.mean(log_rewards))
        joint = (prod_mean + log_mean) / 100.0

        return {
            "prod_reward": prod_mean,
            "log_reward": log_mean,
            "joint_reward": float(joint),
            "fill_rate": float(np.mean(fill_rates)),
        }

    # ── CSV logger ────────────────────────────────────────────────────────────

    def _write_log(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row["timestamp"] = int(time.time())
        self._log_rows.append(row)
        path = CFG["train_log_path"]
        write_header = not Path(path).exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ── Learning curves ───────────────────────────────────────────────────────

    def _plot_curves(self) -> None:
        if not self._log_rows:
            return
        def moving_avg(x, k=50):
            if len(x) < k:
                return x
            return np.convolve(x, np.ones(k)/k, mode='valid')
        episodes = [r["episode"] for r in self._log_rows]
        joint_r = [r["joint_reward"] for r in self._log_rows]
        fill_rates = [r["fill_rate"] for r in self._log_rows]
        prod_eps = [r["prod_epsilon"] for r in self._log_rows]
        log_eps = [r["log_epsilon"] for r in self._log_rows]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Smart Factory RL — Training Curves", fontsize=14, fontweight="bold")

        axes[0].plot(moving_avg(self.prod_rewards), linewidth=2)
        axes[0].set_title("Production Reward (Smoothed)")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].grid(alpha=0.3)

        axes[1].plot(episodes, fill_rates, linewidth=2)
        axes[1].set_title("Delivery Fill Rate")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Fill Rate (0–1)")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(alpha=0.3)

        axes[2].plot(episodes, prod_eps, label="Production eps", linewidth=2)
        axes[2].plot(episodes, log_eps, label="Logistics eps", linewidth=2)
        axes[2].set_title("Epsilon Decay")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Epsilon")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(CFG["curve_path"])
        plt.close()

        print(f"\n[INFO] Saved training curves....")


    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, prod_episodes: int, log_episodes: int) -> None:
        t_start = time.time()

        # Phase 1: train production
        self.train_production(prod_episodes)

        # Phase 2: train logistics with periodic joint evals
        print(f"\n{'=' * 60}")
        print(f"  PHASE 2 - Logistics DQN    ({log_episodes} episodes)")
        print(f"{'=' * 60}")

        no_improve = 0
        t0 = time.time()

        for ep in range(log_episodes):
            load = float(np.random.randint(200, 1500))
            ep_reward = self._run_log_episode(load)
            self.log_rewards.append(ep_reward)

            if (ep + 1) % CFG["eval_freq"] == 0:
                eval_log = self._eval_logistics()
                j = self.joint_eval()
                self.joint_evals.append(j)

                elapsed = time.time() - t0
                eta = (log_episodes - ep - 1) / max(1e-6, (ep + 1) / max(1e-6, elapsed))

                print(
                    f"  [Log]  Ep {ep + 1:5d}/{log_episodes} | "
                    f"Train {ep_reward:7.1f} | Eval {eval_log:7.1f} | "
                    f"Joint {j['joint_reward']:7.1f} | Fill {j['fill_rate']:.2%} | "
                    f"eps {self.truck_agent.epsilon:.3f} | ETA {eta / 60:.1f} min"
                )

                prod_eval = self._eval_production()
                self._write_log({
                    "episode": prod_episodes + ep + 1,
                    "prod_epsilon": round(self.prod_agent.epsilon, 4),
                    "log_epsilon": round(self.truck_agent.epsilon, 4),
                    "prod_eval": round(prod_eval, 2),
                    "log_eval": round(eval_log, 2),
                    "joint_reward": round(j["joint_reward"], 2),
                    "fill_rate": round(j["fill_rate"], 4),
                })

                if j["joint_reward"] > self.best_joint_eval:
                    self.best_joint_eval = j["joint_reward"]
                    self.prod_agent.save(CFG["prod_best_path"])
                    self.truck_agent.save(CFG["logistics_best_path"])
                    print(f"    [BEST] New best JOINT models saved (joint={j['joint_reward']:.1f})")
                    no_improve = 0
                else:
                    no_improve += CFG["eval_freq"]

                if no_improve >= CFG["patience"]:
                    print(f"  [STOP] Early stopping after {ep + 1} logistics episodes.")
                    break

        # Phase 3: Joint fine-tuning
        print(f"\n{'=' * 60}")
        print("  PHASE 3 - Joint Fine-Tuning (Production + Logistics)")
        print(f"{'=' * 60}")

        for ep in range(500):
            # Production rollout
            state = self.prod_env.reset()
            total_prod = 0.0

            for _ in range(10):
                valid = self.prod_env.get_valid_actions()
                if not valid:
                    break
                action = self.prod_agent.act(state, valid)
                next_s, r, done, info = self.prod_env.step(action)
                self.prod_agent.remember(state, action, r, next_s, done)
                total_prod += float(info.get("estimated_productivity", 0.0))
                state = next_s
                if done:
                    break

            # Logistics rollout
            load = max(50.0, total_prod)
            state = self.truck_env.reset(load)

            for _ in range(5):
                valid = self.truck_env.get_valid_actions()
                if not valid:
                    break
                action = self.truck_agent.act(state, valid)
                next_s, r, done,info = self.truck_env.step(action)
                self.truck_agent.remember(state, action, r, next_s, done)
                state = next_s
                if done:
                    break

            self.prod_agent.replay()
            self.truck_agent.replay()

            if (ep + 1) % 100 == 0:
                print(f"  [Joint] Episode {ep + 1}/500 completed")

        # Final summary
        total_time = time.time() - t_start
        final_j = self.joint_eval(n=20)

        print(f"\n{'=' * 60}")
        print("  TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Total time          : {total_time / 60:.1f} min")
        print(f"  Production episodes : {len(self.prod_rewards)}")
        print(f"  Logistics episodes  : {len(self.log_rewards)}")
        print(f"  Best joint reward   : {self.best_joint_eval:.1f}")
        print(f"  Final joint reward  : {final_j['joint_reward']:.1f}")
        print(f"  Final fill rate     : {final_j['fill_rate']:.2%}")
        print("  Saved models:")
        print(f"    {CFG['prod_best_path']}")
        print(f"    {CFG['logistics_best_path']}")
        print(f"  CSV log: {CFG['train_log_path']}")

        self._plot_curves()
        print("\n==============================")
        print(" BENCHMARK RESULTS")
        print("==============================")

        print("\nProduction:")
        print("RL     :", self.evaluate_production_policy("rl"))
        print("Greedy :", self.evaluate_production_policy("greedy"))
        print("Random :", self.evaluate_production_policy("random"))

        print("\nLogistics:")
        print("RL     :", self.evaluate_logistics_policy("rl"))
        print("Greedy :", self.evaluate_logistics_policy("greedy"))
        print("Random :", self.evaluate_logistics_policy("random"))
        self.run_ablation()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Smart Factory RL — Production-Grade Trainer")
    p.add_argument("--prod", type=int, default=CFG["prod_episodes"], help=f"Production training episodes (default {CFG['prod_episodes']})")
    p.add_argument("--log", type=int, default=CFG["log_episodes"], help=f"Logistics training episodes (default {CFG['log_episodes']})")
    p.add_argument("--eval-freq", type=int, default=CFG["eval_freq"], help=f"Joint eval frequency (default {CFG['eval_freq']})")
    p.add_argument("--seed", type=int, default=CFG["seed"], help=f"Random seed (default {CFG['seed']})")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    CFG["prod_episodes"] = args.prod
    CFG["log_episodes"] = args.log
    CFG["eval_freq"] = args.eval_freq
    CFG["seed"] = args.seed

    trainer = Trainer()
    trainer.run(args.prod, args.log)