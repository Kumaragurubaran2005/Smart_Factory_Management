# =============================
# baselines.py
# =============================
import random
import numpy as np


# -----------------------------
# PRODUCTION BASELINES
# -----------------------------
def greedy_production_action(env):
    best_score = -1e9
    best_action = None

    for a in env.get_valid_actions():
        worker, machine = env.action_map[a]

        profile = env._worker_profiles[worker]
        health = env._machine_health[machine]

        score = (
            profile.skill
            * profile.experience
            * health
            * (1.0 - profile.fatigue)
        )

        if score > best_score:
            best_score = score
            best_action = a

    return best_action


def random_production_action(env):
    valid = env.get_valid_actions()
    return random.choice(valid) if valid else None


# -----------------------------
# LOGISTICS BASELINES
# -----------------------------
def greedy_truck_action(env):
    best = None
    best_cap = -1

    for i, t in enumerate(env.trucks):
        if env.truck_status[t]["available"]:
            cap = env.truck_status[t]["capacity"]
            if cap > best_cap:
                best_cap = cap
                best = i

    return best


def random_truck_action(env):
    valid = env.get_valid_actions()
    return random.choice(valid) if valid else None