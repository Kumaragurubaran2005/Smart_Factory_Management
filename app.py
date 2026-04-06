import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
import cv2
import sqlite3
import tensorflow as tf
from flask import Flask, render_template, jsonify, Response, request, session, redirect, url_for
from flask_sock import Sock
from production_rl import DQNAgent as ProductionAgent
from logistics_rl import TruckAgent as LogisticsAgent, TruckEnv
from raw_material_model import calculate_material_needs, get_7_30_90_predictions
from market_forecast_model import forecast_30_days

# -----------------------------
# APP INFRASTRUCTURE
# -----------------------------
app = Flask(__name__)
app.secret_key = "factory_secret_key"
sock = Sock(app)
connected_clients = set()

@sock.route('/ws')
def websocket_endpoint(ws):
    connected_clients.add(ws)
    try:
        while True:
            msg = ws.receive()
            if not msg:
                break
    except Exception:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)

# -----------------------------
# PATH SETUP
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------
# CONFIGURATION
# -----------------------------
class Config:
    """Application configuration."""
    DB_NAME = os.environ.get('DB_NAME', 'factory.db')
    PROD_MODEL_PATH = os.environ.get('PROD_MODEL_PATH', os.path.join(PROJECT_ROOT, "logs", "prod_best.keras"))
    LOGISTICS_MODEL_PATH = os.environ.get('LOGISTICS_MODEL_PATH', os.path.join(PROJECT_ROOT, "logs", "logistics_best.keras"))
    CSV_PATH = os.environ.get('CSV_PATH', os.path.join(PROJECT_ROOT, "dataset/machines.csv"))
    STAFF_PATH = os.environ.get('STAFF_PATH', os.path.join(PROJECT_ROOT, "dataset/employees.csv"))
    DEMAND_CSV = os.environ.get('DEMAND_CSV', os.path.join(PROJECT_ROOT, "dataset/leather_orders.csv"))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))

# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------
# GLOBAL STATE WITH LOCKS
# -----------------------------
last_pulse = time.time()
last_pulse_lock = threading.Lock()
MANUAL_OVERRIDES = {}          # machine_type -> worker_id
overrides_lock = threading.Lock()
physics_lock = threading.Lock()  # for CSV I/O

GLOBAL_RAW_STOCK = 5000
GLOBAL_FINISHED_GOODS = 0
GLOBAL_DISPATCHED = 0
stock_lock = threading.Lock()

# -----------------------------
# APP INITIALIZATION
# -----------------------------
app.config.from_object(Config)

# Initialize database
try:
    from database import DB_NAME, init_db
    init_db()
    logger.info("Database initialized")
except ImportError as e:
    logger.error(f"Database module import error: {e}")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def safe_read_csv(filepath, **kwargs):
    """Read CSV safely, returning empty DataFrame on error."""
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Failed to read CSV {filepath}: {e}")
        return pd.DataFrame()

def safe_write_csv(df, filepath, **kwargs):
    """Write CSV safely, logging errors."""
    try:
        df.to_csv(filepath, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Failed to write CSV {filepath}: {e}")
        return False

def seed_attendance_if_empty():
    """If no attendance for today, insert records for all workers from employees.csv."""
    try:
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM Attendance WHERE date(timestamp) = ?", (today,))
        count = cursor.fetchone()[0]
        if count == 0:
            staff_df = safe_read_csv(Config.STAFF_PATH)
            if not staff_df.empty:
                now = pd.Timestamp.now().isoformat()
                for _, row in staff_df.iterrows():
                    worker_id = row['worker_id']
                    cursor.execute("INSERT INTO Attendance (worker_id, timestamp) VALUES (?, ?)", (worker_id, now))
                conn.commit()
                logger.info(f"Seeded attendance for {len(staff_df)} workers for today.")
            else:
                logger.warning("No employees.csv found; cannot seed attendance.")
                # Create a default employees.csv if it doesn't exist
                default_employees = pd.DataFrame({
                    'worker_id': ['W001', 'W002', 'W003', 'W004'],
                    'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Lee'],
                    'overall_skill': [85, 92, 78, 88]
                })
                default_employees.to_csv(Config.STAFF_PATH, index=False)
                logger.info(f"Created default employees.csv at {Config.STAFF_PATH}")
                # Retry seeding
                staff_df = safe_read_csv(Config.STAFF_PATH)
                if not staff_df.empty:
                    now = pd.Timestamp.now().isoformat()
                    for _, row in staff_df.iterrows():
                        worker_id = row['worker_id']
                        cursor.execute("INSERT INTO Attendance (worker_id, timestamp) VALUES (?, ?)", (worker_id, now))
                    conn.commit()
                    logger.info(f"Seeded attendance for {len(staff_df)} workers from default CSV.")
        conn.close()
    except Exception as e:
        logger.error(f"Error seeding attendance: {e}")

def get_real_machines():
    """Fetch current machine list with updated wear based on elapsed time."""
    if not os.path.exists(Config.CSV_PATH):
        logger.warning("Machines CSV not found")
        return []

    with physics_lock:
        df = safe_read_csv(Config.CSV_PATH)
        if df.empty:
            return []
        
        required = ["machine_id", "type", "baseProductivity", "initialWear"]
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing column {col} in machines CSV")
                return []
        
        df.dropna(subset=["machine_id", "type", "baseProductivity"], inplace=True)
        
        now = time.time()
        with last_pulse_lock:
            global last_pulse
            elapsed = now - last_pulse
            if elapsed >= 1.0:
                last_pulse = now
            else:
                elapsed = 0.0
        
        machines = []
        for idx, row in df.iterrows():
            try:
                wear = float(row["initialWear"]) + (elapsed * 0.1)
                wear = min(95.0, wear)
                df.at[idx, "initialWear"] = wear
                
                health = max(10, 100 - wear)
                machines.append({
                    "id": row["machine_id"],
                    "type": row["type"],
                    "baseProductivity": row["baseProductivity"],
                    "wear": wear,
                    "health": health,
                    "name": row["type"]
                })
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing machine row {idx}: {e}")
                continue
        
        safe_write_csv(df, Config.CSV_PATH, index=False)
        
    return machines

def get_present_staff():
    """Return DataFrame of staff currently present (attendance today)."""
    staff_df = safe_read_csv(Config.STAFF_PATH)
    if staff_df.empty:
        return staff_df
    
    present_ids = []
    try:
        conn = sqlite3.connect(Config.DB_NAME)
        query = "SELECT DISTINCT worker_id FROM Attendance WHERE date(timestamp) = date('now', 'localtime')"
        present_ids = pd.read_sql(query, conn)["worker_id"].tolist()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to get attendance: {e}")
    
    present_staff = staff_df[staff_df["worker_id"].isin(present_ids)].copy()
    present_staff = present_staff.sort_values(by="overall_skill", ascending=False)
    return present_staff

def calculate_worker_fatigue(worker_id):
    try:
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT start_time FROM WorkerHistory
            WHERE worker_id=? AND end_time IS NULL
            ORDER BY start_time DESC LIMIT 1
        """, (worker_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return 0.0
        start_time = pd.to_datetime(row[0])
        now = pd.Timestamp.now()
        hours_worked = (now - start_time).total_seconds() / 3600.0
        fatigue = max(0.0, min(1.0, hours_worked / 8.0))
        return fatigue
    except Exception as e:
        logger.error(f"Error calculating fatigue: {e}")
        return 0.0

def calculate_rl_decision(machines, present_staff, all_staff, demand_val):
    """
    Reinforcement learning based action recommendation.
    Returns a textual recommendation, Q-value, and confidence.
    """
    global prod_agent, current_forecast
    if prod_agent is None:
        return {"action": "⚠️ RL model not loaded", "q_value": 0, "confidence": 0}
    
    valid_machines = [m for m in machines if m["health"] >= 30.0]
    if not valid_machines:
        return {"action": "Hardware completely offline. Maintenance required.", "q_value": 0, "confidence": 0}
    
    if present_staff.empty:
        return {"action": "No workers present", "q_value": 0, "confidence": 0}
    
    # Build state vector
    health_tanning = health_drying = health_finishing = 1.0
    for m in machines:
        m_type = str(m.get("type", "")).lower()
        h_val = float(m.get("health", 100)) / 100.0
        if "tanning" in m_type:
            health_tanning = h_val
        elif "drying" in m_type:
            health_drying = h_val
        elif "finishing" in m_type:
            health_finishing = h_val
    
    top_workers = present_staff.head(4)
    num_workers = len(top_workers)
    healths = [health_tanning, health_drying, health_finishing]
    wears = [1.0 - h for h in healths]
    productivities = [h for h in healths]
    avail = [1.0 if i < num_workers else 0.0 for i in range(4)]
    
    fatigues_lst = []
    skills_lst = []
    shifts_lst = []
    for i in range(4):
        if i < num_workers:
            w_id = top_workers.iloc[i]["worker_id"]
            f_val = calculate_worker_fatigue(w_id)
            s_val = top_workers.iloc[i]["overall_skill"] / 100.0
            fatigues_lst.append(f_val)
            skills_lst.append(s_val)
            shifts_lst.append(f_val)
        else:
            fatigues_lst.append(0.0)
            skills_lst.append(0.0)
            shifts_lst.append(0.0)
    
    avg_skill = float(np.mean([s for s in skills_lst if s > 0])) if any(s > 0 for s in skills_lst) else 0.0
    norm_demand = demand_val / 1000.0
    sup_lim = current_forecast.get("supply_limit", 600)
    prev_out = current_forecast.get("finished_goods", 0)
    
    state_list = [norm_demand] + healths + wears + productivities + avail + fatigues_lst + skills_lst + shifts_lst + [
        sup_lim / 1000.0,
        np.clip(prev_out / 1000.0, 0.0, 1.0),
        0.0,
        avg_skill
    ]
    state = np.array(state_list, dtype=np.float32)
    
    machine_types = ["tanning", "drying", "finishing"]
    valid_actions = []
    for w_idx in range(4):
        if w_idx >= num_workers:
            continue
        for m_idx in range(3):
            if any(machine_types[m_idx] in m["type"].lower() for m in valid_machines):
                valid_actions.append(w_idx * 3 + m_idx)
    
    try:
        best_action_idx = prod_agent.act(state, valid_actions=valid_actions, eval_mode=True)
        q_values = prod_agent.model.predict(state.reshape(1, -1), verbose=0)[0]
        best_q = q_values[best_action_idx]
    except Exception as e:
        logger.error(f"RL recommendation failed: {e}")
        return {"action": "⚠️ RL error", "q_value": 0, "confidence": 0}
    
    if best_action_idx is None:
        return {"action": "No operational machines for available skills", "q_value": 0, "confidence": 0}
    
    w_idx = best_action_idx // 3
    m_idx = best_action_idx % 3
    recommended_worker = top_workers.iloc[w_idx]
    recommended_machine_type = machine_types[m_idx]
    
    candidate_machines = [m for m in valid_machines if recommended_machine_type in m["type"].lower()]
    if candidate_machines:
        worst_machine = sorted(candidate_machines, key=lambda x: x["health"])[0]
    else:
        worst_machine = sorted(valid_machines, key=lambda x: x["health"])[0]
    
    action_text = f"Assign {recommended_worker['name']} to {worst_machine['type']}"
    return {
        "action": action_text,
        "q_value": float(best_q),
        "confidence": float(best_q)
    }

def compute_allocation(machines, present_staff):
    allocations = []
    conn = sqlite3.connect(Config.DB_NAME)
    cursor = conn.cursor()
    current_time = pd.Timestamp.now()
    used_workers = set()

    with overrides_lock:
        manual_overrides_copy = MANUAL_OVERRIDES.copy()

    sorted_machines = sorted(machines, key=lambda x: x["health"])
    unassigned_machines = []

    for m in sorted_machines:
        if m["health"] < 30:
            allocations.append({
                "machine_id": m["id"],
                "machine_name": m["type"],
                "operator": "UNDER MAINTENANCE",
                "operator_id": "None",
                "reason": "Machine health critically low",
                "match_percent": 0.0,
                "machine_health": m["health"]/100
            })
            continue

        manual_id = manual_overrides_copy.get(m["type"])
        if manual_id and manual_id not in used_workers:
            w_match = present_staff[present_staff['worker_id'] == manual_id]
            if not w_match.empty:
                best_worker = w_match.iloc[0]
                worker_id = best_worker["worker_id"]
                cursor.execute("SELECT machine, start_time FROM WorkerHistory WHERE worker_id=? ORDER BY start_time DESC LIMIT 1", (worker_id,))
                prev = cursor.fetchone()
                if prev:
                    prev_machine, start_time = prev
                    if prev_machine != m["type"]:
                        duration = (current_time - pd.to_datetime(start_time)).total_seconds()
                        cursor.execute("UPDATE WorkerHistory SET end_time=?, duration=? WHERE worker_id=? AND end_time IS NULL", (current_time.isoformat(), duration, worker_id))
                        cursor.execute("INSERT INTO WorkerHistory (worker_id, machine, start_time) VALUES (?, ?, ?)", (worker_id, m["type"], current_time.isoformat()))
                else:
                    cursor.execute("INSERT INTO WorkerHistory (worker_id, machine, start_time) VALUES (?, ?, ?)", (worker_id, m["type"], current_time.isoformat()))
                conn.commit()
                allocations.append({
                    "machine_id": m["id"],
                    "machine_name": m["type"],
                    "operator": best_worker["name"],
                    "operator_id": worker_id,
                    "reason": "Assigned via Administrator Manual Override.",
                    "match_percent": 1.0,
                    "machine_health": m["health"]/100
                })
                used_workers.add(worker_id)
                continue

        unassigned_machines.append(m)

    # DRL autonomous allocation
    machine_types = ["tanning", "drying", "finishing"]
    while unassigned_machines:
        if 'role' in present_staff.columns:
            op_staff = present_staff[(present_staff['role'].isna()) | (present_staff['role'] == 'Operator')]
        else:
            op_staff = present_staff
        available_staff = op_staff[~op_staff['worker_id'].isin(used_workers)]
        if available_staff.empty:
            break
        top_workers = available_staff.head(4)
        num_workers = len(top_workers)
        try:
            demand_val = current_forecast.get("predicted_demand", 500)
        except Exception:
            demand_val = 500
        norm_demand = min(1.0, demand_val / 1000.0)
        health_tanning = health_drying = health_finishing = 1.0
        for m in machines:
            mt = str(m.get("type", "")).lower()
            hv = float(m.get("health", 100)) / 100.0
            if "tanning" in mt: health_tanning = hv
            elif "drying" in mt: health_drying = hv
            elif "finishing" in mt: health_finishing = hv
        healths = [health_tanning, health_drying, health_finishing]
        wears = [1.0 - h for h in healths]
        productivities = [h for h in healths]
        avail = [1.0 if i < num_workers else 0.0 for i in range(4)]
        fatigues_lst = []
        skills_lst = []
        shifts_lst = []
        for i in range(4):
            if i < num_workers:
                w_id = top_workers.iloc[i]["worker_id"]
                f_val = calculate_worker_fatigue(w_id)
                s_val = top_workers.iloc[i]["overall_skill"] / 100.0
                fatigues_lst.append(f_val)
                skills_lst.append(s_val)
                shifts_lst.append(f_val)
            else:
                fatigues_lst.append(0.0)
                skills_lst.append(0.0)
                shifts_lst.append(0.0)
        avg_skill = float(np.mean([s for s in skills_lst if s > 0])) if any(s > 0 for s in skills_lst) else 0.0
        sup_lim = current_forecast.get("supply_limit", 600)
        prev_out = current_forecast.get("finished_goods", 0)
        state_list = [norm_demand] + healths + wears + productivities + avail + fatigues_lst + skills_lst + shifts_lst + [
            sup_lim / 1000.0,
            np.clip(prev_out / 1000.0, 0.0, 1.0),
            0.0,
            avg_skill
        ]
        state = np.array(state_list, dtype=np.float32)
        valid_actions = []
        for w_idx in range(num_workers):
            for m_idx in range(3):
                m_type = machine_types[m_idx]
                if any(m_type in str(m["type"]).lower() for m in unassigned_machines):
                    valid_actions.append(w_idx * 3 + m_idx)
        if not valid_actions:
            break
        if prod_agent is None:
            best_action_idx = valid_actions[0]
            best_q = 0.0
        else:
            try:
                best_action_idx = prod_agent.act(state, valid_actions=valid_actions, eval_mode=True)
                q_values = prod_agent.model.predict(state.reshape(1, -1), verbose=0)[0]
                best_q = q_values[best_action_idx]
            except Exception as e:
                logger.error(f"RL loop err: {e}")
                best_action_idx = valid_actions[0]
                best_q = 0.0
        w_idx = best_action_idx // 3
        m_idx = best_action_idx % 3
        best_worker = top_workers.iloc[w_idx]
        recommended_machine_type = machine_types[m_idx]
        candidates = [m for m in unassigned_machines if recommended_machine_type in str(m["type"]).lower()]
        target_machine = candidates[0]
        worker_id = best_worker["worker_id"]
        cursor.execute("SELECT machine, start_time FROM WorkerHistory WHERE worker_id=? ORDER BY start_time DESC LIMIT 1", (worker_id,))
        prev = cursor.fetchone()
        if prev:
            prev_machine, start_time = prev
            if prev_machine != target_machine["type"]:
                duration = (current_time - pd.to_datetime(start_time)).total_seconds()
                cursor.execute("UPDATE WorkerHistory SET end_time=?, duration=? WHERE worker_id=? AND end_time IS NULL", (current_time.isoformat(), duration, worker_id))
                cursor.execute("INSERT INTO WorkerHistory (worker_id, machine, start_time) VALUES (?, ?, ?)", (worker_id, target_machine["type"], current_time.isoformat()))
        else:
            cursor.execute("INSERT INTO WorkerHistory (worker_id, machine, start_time) VALUES (?, ?, ?)", (worker_id, target_machine["type"], current_time.isoformat()))
        conn.commit()
        fatigue_val = calculate_worker_fatigue(worker_id)
        reason = f"""[RL AI Optimal Pathing]
- Q-Value Confidence: {best_q:.2f}
- Selected Profile >> Skill: {best_worker['overall_skill']} | Fatigue Evaluator: {fatigue_val:.2f}
- Target: Optimal wear dispersion identified."""
        allocations.append({
            "machine_id": target_machine["id"],
            "machine_name": target_machine["type"],
            "operator": best_worker["name"],
            "operator_id": worker_id,
            "reason": reason,
            "match_percent": (best_worker["overall_skill"] / 100.0) * (target_machine["health"] / 100.0),
            "machine_health": target_machine["health"]/100
        })
        used_workers.add(worker_id)
        unassigned_machines.remove(target_machine)

    for m in unassigned_machines:
        allocations.append({
            "machine_id": m["id"],
            "machine_name": m["type"],
            "operator": "Unassigned",
            "operator_id": "None",
            "reason": "No available matched workers",
            "match_percent": 0.0,
            "machine_health": m["health"]/100
        })

    conn.close()
    return allocations

def compute_stats(machines, allocations=None):
    total = len(machines)
    avg_health = np.mean([m["health"] for m in machines]) / 100 if machines else 0
    critical = len([m for m in machines if m["health"] < 40])
    stage_capacities = {}
    for m in machines:
        stage = m.get("type", "Unknown")
        if stage not in stage_capacities:
            stage_capacities[stage] = 0
        if allocations:
            assigned_mids = {a["machine_id"] for a in allocations if a["operator"] not in ["Unassigned", "UNDER MAINTENANCE"]}
            if m["id"] in assigned_mids:
                stage_capacities[stage] += int(m["baseProductivity"] * (m["health"] / 100))
        else:
            stage_capacities[stage] += int(m["baseProductivity"] * (m["health"] / 100))
    output = min(stage_capacities.values()) if stage_capacities else 0
    return {
        "total_machines": total,
        "avg_health": avg_health,
        "critical_count": critical,
        "production_output": output
    }

def get_workers_list():
    present_staff = get_present_staff()
    workers = []
    for _, w in present_staff.iterrows():
        workers.append({"id": w["worker_id"], "name": w["name"]})
    return workers

def generate_advanced_forecast():
    global lstm_recent_seq, current_forecast, pipeline
    if not pipeline:
        return current_forecast
    try:
        predicted_demand = pipeline.predict_demand(lstm_recent_seq)
        machines = get_real_machines()
        if not machines:
            machines = [
                {"type": "tanning", "baseProductivity": 120, "health": 80},
                {"type": "drying", "baseProductivity": 100, "health": 70},
                {"type": "finishing", "baseProductivity": 90, "health": 85}
            ]
        machine_list = [
            {"type": m.get("name") or m.get("type"), "baseProductivity": m["baseProductivity"], "health": m["health"]}
            for m in machines
        ]
        with stock_lock:
            raw_stock = GLOBAL_RAW_STOCK
        rejection_rate = np.random.uniform(0.05, 0.15)
        usable_raw = raw_stock * (1 - rejection_rate)
        output, bottleneck = pipeline.final_output(predicted_demand, usable_raw, machine_list)
        caps = {}
        for m in machines:
            m_type = m.get("type", "").lower()
            cap = float(m.get("baseProductivity", 100.0)) * (float(m.get("health", 0.0))/100.0)
            for key in ["tanning", "drying", "finishing"]:
                if key in m_type:
                    caps[key] = caps.get(key, 0.0) + cap
        lstm_recent_seq.append(output)
        lstm_recent_seq.pop(0)
        with stock_lock:
            result = {
                "date": pd.Timestamp.now().strftime('%Y-%m-%d'),
                "predicted_demand": predicted_demand,
                "supply_limit": GLOBAL_RAW_STOCK,
                "stage_capacities": caps,
                "bottleneck_stage": bottleneck,
                "feasible_output": output,
                "process_capacity": min(caps.values()) if caps else output,
                "finished_goods": GLOBAL_FINISHED_GOODS,
                "dispatched_goods": GLOBAL_DISPATCHED
            }
        return result
    except Exception as e:
        logger.error(f"Forecast generation failed: {e}")
        return current_forecast

def compute_truck_allocation(load):
    if logistics_agent is None:
        return {"error": "Logistics RL not loaded", "assignments": []}
    state = logistics_env.reset(load)
    allocations = []
    for _ in range(10):
        valid = logistics_env.get_valid_actions()
        if not valid:
            break
        try:
            action = logistics_agent.act(state, valid, eval_mode=True)
        except TypeError:
            action = logistics_agent.act(state, valid)
        truck_id = logistics_env.trucks[action]
        truck = logistics_env.truck_status[truck_id]
        delivered = min(truck["capacity"], logistics_env.remaining_load)
        next_state, reward, done = logistics_env.step(action)
        allocations.append({"truck_id": truck_id, "assigned_load": delivered})
        state = next_state
        if done:
            break
    return {"assignments": allocations, "remaining": logistics_env.remaining_load}

# -----------------------------
# LOAD MODELS AND DATA
# -----------------------------
prod_agent = None
if os.path.exists(Config.PROD_MODEL_PATH):
    try:
        prod_agent = ProductionAgent(30, 12)
        prod_agent.load(Config.PROD_MODEL_PATH)
        logger.info("Production RL model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load production RL model: {e}")
else:
    logger.warning("Production RL model not found at %s", Config.PROD_MODEL_PATH)

logistics_agent = None
logistics_env = TruckEnv()
if os.path.exists(Config.LOGISTICS_MODEL_PATH):
    try:
        logistics_agent = LogisticsAgent(logistics_env.state_size, logistics_env.action_size)
        logistics_agent.load(Config.LOGISTICS_MODEL_PATH)
        logger.info("Logistics RL model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load logistics RL model: {e}")
else:
    logger.warning("Logistics RL model not found at %s", Config.LOGISTICS_MODEL_PATH)

pipeline = None
lstm_recent_seq = []
current_forecast = {
    "feasible_output": 400,
    "predicted_demand": 400,
    "bottleneck_stage": "Unknown",
    "process_capacity": 400
}

try:
    from production_model import ProductionSystem
    logger.info("Initializing Advanced Production Intelligence AI...")
    pipeline = ProductionSystem()
    if os.path.exists(Config.DEMAND_CSV):
        try:
            demand_series = pd.read_csv(Config.DEMAND_CSV).groupby('date')['quantity'].sum().values
            lstm_recent_seq = list(demand_series[-7:]) if len(demand_series) >= 7 else [400] * 7 
            logger.info("Demand model trained on historical data")
        except Exception as e:
            logger.error(f"Failed to train demand model: {e}")
            lstm_recent_seq = [400] * 7
    else:
        logger.warning("Demand CSV not found; using random data for training")
        demand_series = np.random.randint(300, 600, 100)
        lstm_recent_seq = list(demand_series[-7:]) if len(demand_series) >= 7 else [400] * 7
    logger.info("Advanced AI ready")
except ImportError as e:
    logger.error(f"ProductionSystem import failed: {e}")
except Exception as e:
    logger.error(f"Error loading forecasting model: {e}")

seed_attendance_if_empty()

# -----------------------------
# AUTONOMOUS FACTORY LOOP
# -----------------------------
def autonomous_factory_loop():
    global GLOBAL_RAW_STOCK, GLOBAL_FINISHED_GOODS, GLOBAL_DISPATCHED, current_forecast
    logger.info("Autonomous Factory System initiated. Waiting 5s before first cycle...")
    time.sleep(5)
    while True:
        try:
            time.sleep(5)
            machines = get_real_machines()
            present_staff = get_present_staff()
            if machines:
                allocations = compute_allocation(machines, present_staff)
                assigned_mc = {a["machine_id"] for a in allocations if a["operator"] not in ["Unassigned", "UNDER MAINTENANCE"]}
                stage_capacities = {}
                for m in machines:
                    stage = m["type"]
                    if stage not in stage_capacities:
                        stage_capacities[stage] = 0
                    if m["id"] in assigned_mc:
                        val = m["baseProductivity"] * (min(100, max(0, float(m["health"]))) / 100)
                        stage_capacities[stage] += val
                raw_prod = min(stage_capacities.values()) if stage_capacities else 0
                tick_prod = int(raw_prod * 0.1)
                tick_prod = max(0, tick_prod)
            else:
                tick_prod = 0
            pending_orders_global = max(0, current_forecast.get("predicted_demand", 0) - GLOBAL_DISPATCHED)
            with stock_lock:
                GLOBAL_RAW_STOCK = max(0, GLOBAL_RAW_STOCK)
                GLOBAL_FINISHED_GOODS = max(0, GLOBAL_FINISHED_GOODS)
                target_to_produce = max(0, pending_orders_global - GLOBAL_FINISHED_GOODS)
                desired_production = min(tick_prod, target_to_produce)
                if GLOBAL_RAW_STOCK >= desired_production:
                    actual_production = desired_production
                else:
                    actual_production = max(0, GLOBAL_RAW_STOCK)
                GLOBAL_RAW_STOCK -= actual_production
                GLOBAL_FINISHED_GOODS += actual_production
                current_finished = GLOBAL_FINISHED_GOODS
            if actual_production > 0:
                logger.info(f"⚙️ Autoloop Output: {actual_production} units. Raw Remaining: {GLOBAL_RAW_STOCK} | Ready for Delivery: {current_finished}")
            if connected_clients:
                state_bundle = {
                    "type": "stats",
                    "rawStock": GLOBAL_RAW_STOCK,
                    "finishedGoods": GLOBAL_FINISHED_GOODS,
                    "dispatched": GLOBAL_DISPATCHED,
                    "allocations": allocations
                }
                dead_clients = set()
                bundle_str = json.dumps(state_bundle)
                for ws in connected_clients:
                    try:
                        ws.send(bundle_str)
                    except Exception:
                        dead_clients.add(ws)
                for ws in dead_clients:
                    connected_clients.remove(ws)
            if pending_orders_global > 0 and current_finished > 0:
                dispatch_amount = min(current_finished, pending_orders_global)
                dispatch_ready = (current_finished >= pending_orders_global) or (actual_production == 0)
                if dispatch_ready:
                    result = compute_truck_allocation(dispatch_amount)
                    truck_assignments = result.get("assignments", [])
                    if truck_assignments:
                        total_dispatched_now = sum(alloc["assigned_load"] for alloc in truck_assignments)
                        logger.info(f"🚚 🤖 AUTO-DISPATCHED (RL): {total_dispatched_now} units across {len(truck_assignments)} trucks.")
                        with stock_lock:
                            GLOBAL_FINISHED_GOODS -= total_dispatched_now
                            GLOBAL_DISPATCHED += total_dispatched_now
        except Exception as e:
            logger.error(f"Autonomous Factory Loop Error: {e}")

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route("/favicon.ico")
def favicon():
    return Response(status=204)

@app.route("/")
def index():
    if session.get("role") != "admin":
        return redirect(url_for("login_page"))
    return render_template("index.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/api/auth", methods=["POST"])
def auth():
    data = request.get_json()
    user_id = data.get("id", "").strip().upper()
    role = data.get("role", "worker")
    if role == "admin":
        if user_id == "ADMIN":
            session["user_id"] = "ADMIN"
            session["role"] = "admin"
            session["name"] = "Operations Director"
            return jsonify({"success": True, "redirect": "/"})
        return jsonify({"success": False, "message": "Invalid administrator clearance!"}), 403
    conn = sqlite3.connect(Config.DB_NAME)
    cursor = conn.cursor()
    if role == "mechanic":
        cursor.execute("SELECT name FROM Mechanics WHERE mechanic_id=?", (user_id,))
        rv = cursor.fetchone()
        if rv:
            session["user_id"] = user_id
            session["role"] = role
            session["name"] = rv[0]
            conn.close()
            return jsonify({"success": True, "redirect": "/mechanic"})
    else:
        cursor.execute("SELECT name FROM Workers WHERE worker_id=?", (user_id,))
        rv = cursor.fetchone()
        if rv:
            session["user_id"] = user_id
            session["role"] = role
            session["name"] = rv[0]
            conn.close()
            return jsonify({"success": True, "redirect": "/worker"})
    conn.close()
    return jsonify({"success": False, "message": "Invalid credentials!"}), 401

@app.route("/mechanic")
def mechanic_portal():
    if session.get("role") != "mechanic":
        return redirect(url_for("login_page"))
    return render_template("mechanic.html", name=session.get("name"))

@app.route("/worker")
def worker_portal():
    if session.get("role") != "worker":
        return redirect(url_for("login_page"))
    return render_template("worker.html", name=session.get("name"), worker_id=session.get("user_id"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/api/repair", methods=["POST"])
def repair_machine():
    if session.get("role") != "mechanic":
        return jsonify({"message": "Unauthorized"}), 403
    data = request.get_json()
    machine_id = data.get("machine_id")
    with physics_lock:
        df = pd.read_csv(Config.CSV_PATH)
        if "machine_id" in df.columns:
            df.loc[df["machine_id"] == machine_id, "initialWear"] = 0.0
            df.to_csv(Config.CSV_PATH, index=False)
            logger.info(f"Mechanic {session.get('name')} repaired {machine_id}")
            return jsonify({"success": True})
    return jsonify({"success": False, "message": "Machine not found"}), 404

@app.route("/api/overtime", methods=["POST"])
def apply_overtime():
    worker_id = session.get("user_id")
    if not worker_id:
        return jsonify({"message": "Unauthorized"}), 403
    conn = sqlite3.connect(Config.DB_NAME)
    cursor = conn.cursor()
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    try:
        cursor.execute("INSERT INTO Overtime_Requests (worker_id, date, status) VALUES (?, ?, ?)", (worker_id, today, "Approved"))
        conn.commit()
        logger.info(f"Worker {worker_id} applied for approved overtime.")
        success = True
    except Exception as e:
        logger.error(f"Overtime API Error: {e}")
        success = False
    finally:
        conn.close()
    return jsonify({"success": success})

@app.route("/video_feed")
def video_feed():
    def gen_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("Could not open camera")
            return
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/stats")
def stats():
    machines = get_real_machines()
    present = get_present_staff()
    alloc = compute_allocation(machines, present)
    stats_data = compute_stats(machines, alloc)
    stats_data["machines"] = [{
        "id": m["id"],
        "name": m["type"],
        "health": m["health"] / 100,
        "productivity": m["baseProductivity"] / 100,
        "wear": m["wear"]
    } for m in machines]
    return jsonify(stats_data)

@app.route("/api/rl_output")
def rl_output():
    machines = get_real_machines()
    present_staff = get_present_staff()
    staff_df = safe_read_csv(Config.STAFF_PATH)
    decision = calculate_rl_decision(machines, present_staff, staff_df, current_forecast["feasible_output"])
    return jsonify(decision)

@app.route("/api/alloc")
def allocations():
    machines = get_real_machines()
    present_staff = get_present_staff()
    alloc = compute_allocation(machines, present_staff)
    return jsonify(alloc)

@app.route("/api/workers")
def workers():
    workers_list = get_workers_list()
    return jsonify(workers_list)

@app.route("/api/voice_command", methods=["POST"])
def voice_command():
    data = request.get_json()
    command = data.get("command", "").lower()
    response = {"message": "Command received but not processed."}
    if "assign" in command:
        words = command.split()
        if "to" in words:
            try:
                to_idx = words.index("to")
                worker_parts = words[words.index("assign")+1:to_idx]
                machine_parts = words[to_idx+1:]
                machine_name = " ".join(machine_parts).strip()
                machines = get_real_machines()
                target_machine = None
                for m in machines:
                    if machine_name in m["type"].lower():
                        target_machine = m
                        break
                if not target_machine:
                    response["message"] = f"Machine '{machine_name}' not found."
                    return jsonify(response), 404
                present = get_present_staff()
                found_worker = None
                for _, w in present.iterrows():
                    w_name = w["name"].lower()
                    if any(part in w_name for part in worker_parts):
                        found_worker = w
                        break
                if found_worker is not None:
                    with overrides_lock:
                        MANUAL_OVERRIDES[target_machine["type"]] = found_worker["worker_id"]
                    response["message"] = f"Override accepted: {found_worker['name']} assigned to {target_machine['type']}."
                else:
                    response["message"] = "Worker not present."
            except Exception as e:
                logger.error(f"Voice command error: {e}")
                response["message"] = f"Syntax error: {str(e)}"
        else:
            response["message"] = "Format: 'Assign [Name] to [Machine]'"
    elif "status" in command:
        response["message"] = "Factory sensors nominal."
    else:
        response["message"] = "Unknown command."
    return jsonify(response)

@app.route("/api/assign", methods=["POST"])
def assign():
    data = request.get_json()
    worker_id = data.get("worker_id")
    machine_identifier = data.get("machine") or data.get("machine_id")
    if not machine_identifier:
        return jsonify({"message": "No machine specified"}), 400
    machines = get_real_machines()
    target_machine = None
    for m in machines:
        if m["id"] == machine_identifier:
            target_machine = m
            break
    if not target_machine:
        for m in machines:
            if machine_identifier.lower() in m["type"].lower():
                target_machine = m
                break
    if not target_machine:
        return jsonify({"message": f"Machine '{machine_identifier}' not found"}), 404
    with overrides_lock:
        MANUAL_OVERRIDES[target_machine["type"]] = worker_id
    logger.info(f"Manual override: {worker_id} → {target_machine['type']}")
    return jsonify({"message": f"Successfully locked operator to {target_machine['type']}."})

@app.route("/api/truck/login", methods=["POST"])
def truck_login():
    data = request.get_json()
    truck_id = data.get("truck_id")
    driver_name = data.get("driver_name")
    capacity = data.get("capacity")
    try:
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO Trucks (truck_id, driver_name, capacity, status) VALUES (?, ?, ?, 'available')", (truck_id, driver_name, capacity))
        conn.commit()
        conn.close()
        return jsonify({"message": "Truck logged in successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/assign_trucks", methods=["POST"])
def assign_trucks():
    data = request.get_json()
    load = data.get("load")
    result = compute_truck_allocation(load)
    return jsonify(result)

@app.route("/api/forecast")
def get_forecast():
    global current_forecast
    if not current_forecast.get("date"):
        try:
            current_forecast = generate_advanced_forecast()
        except Exception as e:
            logger.error(f"Forecast init failed: {e}")
            return jsonify({"error": "Forecast system initializing..."}), 503
    with stock_lock:
        current_forecast["supply_limit"] = GLOBAL_RAW_STOCK
        current_forecast["finished_goods"] = GLOBAL_FINISHED_GOODS
        current_forecast["dispatched_goods"] = GLOBAL_DISPATCHED
    return jsonify(current_forecast)

@app.route("/api/forecast/next_day", methods=["POST"])
def next_day_forecast():
    global current_forecast
    try:
        current_forecast = generate_advanced_forecast()
        return jsonify(current_forecast)
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({"error": "Forecast failed", "message": str(e)}), 500

@app.route("/place_order")
def place_order_page():
    return render_template("place_order.html")

@app.route("/orders")
def orders_history_page():
    return render_template("orders_history.html")

@app.route("/manual_override")
def manual_override_page():
    return render_template("manual_override.html")

@app.route("/api/orders_history")
def api_orders_history():
    from database import get_orders
    try:
        df = get_orders()
        return jsonify(df.to_dict('records'))
    except Exception as e:
        logger.error(f"Error fetching orders: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/place_order", methods=["POST"])
def api_place_order():
    try:
        data = request.json
        from database import add_order
        order_id = add_order(data['customer_name'], data['quantity'], data['deadline'])
        return jsonify({"success": True, "order_id": order_id})
    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/dashboard")
def dashboard():
    machines = get_real_machines()
    present_staff = get_present_staff()
    staff_df = safe_read_csv(Config.STAFF_PATH)
    rl_decision = calculate_rl_decision(machines, present_staff, staff_df, current_forecast["feasible_output"])
    allocations = compute_allocation(machines, present_staff)
    stats_data = compute_stats(machines, allocations)
    stats_data["forecast"] = current_forecast
    delivery_plan = {"message": "Active dispatching handled by autonomous loop."}
    return jsonify({
        "machines": machines,
        "attendance": present_staff.to_dict("records"),
        "rl_decision": rl_decision,
        "allocations": allocations,
        "stats": stats_data,
        "delivery": delivery_plan
    })

@app.route("/api/worker_info/<worker_id>")
def worker_info(worker_id):
    conn = sqlite3.connect(Config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT machine, start_time FROM WorkerHistory WHERE worker_id=? ORDER BY start_time DESC LIMIT 1", (worker_id,))
    current = cursor.fetchone()
    cursor.execute("SELECT machine, duration FROM WorkerHistory WHERE worker_id=? AND duration IS NOT NULL ORDER BY start_time DESC LIMIT 1 OFFSET 1", (worker_id,))
    prev = cursor.fetchone()
    machines = get_real_machines()
    present_staff = get_present_staff()
    allocs = compute_allocation(machines, present_staff)
    reason = "Standby"
    match_percent = 0.0
    for a in allocs:
        if a.get("operator_id") == worker_id:
            reason = a.get("reason", "Standby")
            match_percent = a.get("match_percent", 0.0)
            break
    conn.close()
    current_machine = current[0] if current else "None"
    start_time = current[1] if current else None
    time_spent = 0
    if start_time:
        try:
            time_spent = (pd.Timestamp.now() - pd.to_datetime(start_time)).total_seconds()
        except:
            time_spent = 0
    return jsonify({
        "current_machine": current_machine,
        "time_spent_seconds": time_spent,
        "previous_machine": prev[0] if prev else "None",
        "previous_duration": prev[1] if prev else 0,
        "reason": reason,
        "match_percent": match_percent
    })

# ===================== NEW ORDER & SUPPLY CHAIN ROUTES =====================
@app.route("/orders_portal")
def orders_portal():
    if session.get("role") != "admin":
        return redirect(url_for("login_page"))
    return render_template("orders_portal.html")

@app.route("/order_detail/<int:order_id>")
def order_detail(order_id):
    if session.get("role") != "admin":
        return redirect(url_for("login_page"))
    return render_template("order_detail.html", order_id=order_id)

@app.route("/warehouse_dashboard")
def warehouse_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login_page"))
    return render_template("warehouse_dashboard.html")

@app.route("/raw_material_portal")
def raw_material_portal():
    if session.get("role") != "admin":
        return redirect(url_for("login_page"))
    return render_template("raw_material_portal.html")

@app.route("/api/orders/place", methods=["POST"])
def api_place_customer_order():
    try:
        data = request.json
        from database import add_customer_order
        order_id, order_ref = add_customer_order(data)
        return jsonify({"success": True, "order_id": order_id, "order_ref": order_ref})
    except Exception as e:
        logger.error(f"Place order error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/orders/list")
def api_orders_list():
    from database import get_all_customer_orders
    df = get_all_customer_orders()
    return jsonify(df.to_dict('records'))

@app.route("/api/orders/<int:order_id>")
def api_order_detail(order_id):
    from database import get_order_by_id
    order = get_order_by_id(order_id)
    if order is None:
        return jsonify({"error": "Order not found"}), 404
    return jsonify(order.to_dict())

@app.route("/api/orders/<int:order_id>/process", methods=["POST"])
def api_process_order(order_id):
    from database import get_order_by_id, update_order_status, add_order_allocation
    order = get_order_by_id(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404
    machines = get_real_machines()
    present_staff = get_present_staff()
    if present_staff.empty:
        return jsonify({"error": "No workers present"}), 400
    allocations = compute_allocation(machines, present_staff)
    assigned = [a for a in allocations if a["operator"] not in ["Unassigned", "UNDER MAINTENANCE"]]
    if not assigned:
        return jsonify({"error": "No available machines/workers"}), 400
    alloc = assigned[0]
    add_order_allocation(order_id, alloc["operator_id"], alloc["operator"], alloc["machine_id"], alloc["machine_name"])
    update_order_status(order_id, "allocated")
    update_order_status(order_id, "in_production")
    return jsonify({"success": True, "message": f"Order {order_id} allocated to {alloc['operator']} on {alloc['machine_name']}"})

@app.route("/api/orders/<int:order_id>/ship", methods=["POST"])
def api_ship_order(order_id):
    from database import get_order_by_id, update_order_status, add_shipment, update_warehouse_stock
    order = get_order_by_id(order_id)
    if not order:
        return jsonify({"error": "Order not found"}), 404
    if order['status'] != 'in_production':
        return jsonify({"error": "Order not ready for shipping"}), 400
    add_shipment(order_id, order['quantity'])
    update_warehouse_stock(order['product_type'], order['quantity'])
    from datetime import datetime
    update_order_status(order_id, "shipped", actual_completion=datetime.now().isoformat())
    return jsonify({"success": True, "message": f"Order {order_id} shipped to warehouse"})

@app.route("/api/warehouse")
def api_warehouse():
    from database import get_warehouse_stock
    df = get_warehouse_stock()
    return jsonify(df.to_dict('records'))

@app.route("/api/raw_material/inventory")
def api_raw_material_inventory():
    from database import get_raw_material_inventory
    df = get_raw_material_inventory()
    return jsonify(df.to_dict('records'))

@app.route("/api/raw_material/needs", methods=["GET"])
def api_raw_material_needs():
    import sqlite3
    conn = sqlite3.connect(Config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT product_type, quantity FROM CustomerOrders WHERE status NOT IN ('shipped', 'cancelled')")
    rows = cursor.fetchall()
    conn.close()
    orders = [{"product_type": r[0], "quantity": r[1]} for r in rows]
    needs = calculate_material_needs(orders)
    return jsonify(needs)

@app.route("/api/raw_material/predict", methods=["GET"])
def api_raw_material_predict():
    import sqlite3
    conn = sqlite3.connect(Config.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT material_type, quantity FROM RawMaterialInventory")
    inv = {r[0]: r[1] for r in cursor.fetchall()}
    cursor.execute("""
        SELECT product_type, quantity FROM CustomerOrders
        WHERE created_at >= datetime('now', '-30 days') AND status NOT IN ('cancelled')
    """)
    recent = [{"product_type": r[0], "quantity": r[1]} for r in cursor.fetchall()]
    conn.close()
    if not recent:
        recent = [{"product_type": "Belt", "quantity": 100}]
    predictions = get_7_30_90_predictions(inv, recent)
    return jsonify(predictions)

@app.route("/api/raw_material/order", methods=["POST"])
def api_place_raw_material_order():
    data = request.json
    from database import add_raw_material_order
    add_raw_material_order(
        material_type=data['material_type'],
        quantity=data['quantity'],
        supplier=data.get('supplier', 'Unknown'),
        urgency=data.get('urgency', 'Normal'),
        cost=data.get('cost')
    )
    return jsonify({"success": True, "message": "Raw material order placed"})

@app.route("/api/raw_material/orders", methods=["GET"])
def api_raw_material_orders():
    from database import get_raw_material_orders
    df = get_raw_material_orders()
    return jsonify(df.to_dict('records'))

@app.route("/api/market/forecast", methods=["GET"])
def api_market_forecast():
    forecast = forecast_30_days()
    return jsonify(forecast)

if __name__ == "__main__":
    factory_thread = threading.Thread(target=autonomous_factory_loop, daemon=True)
    factory_thread.start()

    print("🚀 Running on http://127.0.0.1:5000")

    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000,
        threaded=True
    )