import pandas as pd
import random
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
NUM_EMPLOYEES = 100
DAYS_HISTORY = 7

machine_types = ["Tanning", "Drying", "Finishing", "Packaging"]

first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]

# -----------------------------
# EMPLOYEES
# -----------------------------
employees = []

for i in range(1, NUM_EMPLOYEES + 1):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    
    # Assign specialization (IMPORTANT for RL realism)
    primary_skill = random.choice(machine_types)

    employees.append({
        "worker_id": f"W{i:03d}",
        "name": name,
        "overall_skill": random.randint(60, 99),

        "skill_tanning": 1 if primary_skill == "Tanning" or random.random() > 0.7 else 0,
        "skill_drying": 1 if primary_skill == "Drying" or random.random() > 0.7 else 0,
        "skill_finishing": 1 if primary_skill == "Finishing" or random.random() > 0.7 else 0,
        "skill_packaging": 1 if primary_skill == "Packaging" or random.random() > 0.7 else 0
    })

df_employees = pd.DataFrame(employees)
df_employees.to_csv("employees.csv", index=False)

# -----------------------------
# ATTENDANCE (SHIFT BASED)
# -----------------------------
attendance = []
base_date = datetime.now()

for days_ago in range(DAYS_HISTORY, -1, -1):
    current_date = base_date - timedelta(days=days_ago)

    for emp in employees:
        if random.random() < 0.85:
            shift = random.choice(["Morning", "Evening"])

            if shift == "Morning":
                hour = random.randint(7, 9)
            else:
                hour = random.randint(14, 16)

            timestamp = current_date.replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )

            attendance.append({
                "worker_id": emp["worker_id"],
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Present",
                "confidence": round(random.uniform(0.85, 0.99), 2),
                "shift": shift
            })

df_attendance = pd.DataFrame(attendance).sort_values(by="timestamp")
df_attendance.to_csv("attendance.csv", index=False)

# -----------------------------
# MACHINES (20 MACHINES)
# -----------------------------
machines = []

for i in range(1, 21):
    m_type = random.choice(machine_types)

    machines.append({
        "machine_id": f"M-{100+i}",
        "type": m_type,
        "baseProductivity": random.randint(70, 100),
        "initialWear": round(random.uniform(10, 70), 2)
    })

df_machines = pd.DataFrame(machines)
df_machines.to_csv("machines.csv", index=False)

# -----------------------------
# DONE
# -----------------------------
print("✅ Generated:")
print(" - employees.csv (100 workers)")
print(" - attendance.csv (7 days + shifts)")
print(" - machines.csv (20 machines)")