import pandas as pd
import random
import os
from datetime import datetime
from database import init_db, mark_attendance

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "dataset/employees.csv"
ABSENT_RANGE = (15, 25)

# -----------------------------
# LOAD EMPLOYEES
# -----------------------------
if not os.path.exists(CSV_PATH):
    print(f"❌ Missing file: {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)

if "worker_id" not in df.columns:
    raise ValueError("CSV must contain 'worker_id' column")

staff_data = df.to_dict("records")
num_staff = len(staff_data)

# -----------------------------
# SIMULATE ATTENDANCE
# -----------------------------
def simulate_attendance():
    absent_count = random.randint(*ABSENT_RANGE)
    present_count = num_staff - absent_count

    present_workers = random.sample(staff_data, present_count)

    return present_workers, absent_count, present_count

# -----------------------------
# INSERT INTO DB
# -----------------------------
def insert_attendance(present_workers):
    today = datetime.now().date()
    now = datetime.now()

    inserted = 0

    for w in present_workers:
        worker_id = w["worker_id"]

        # Let the SQLite database assign its own timestamp natively
        mark_attendance(worker_id=worker_id, confidence=1.0)
        inserted += 1

    return inserted

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running Attendance Simulation...\n")

    init_db()

    present_workers, absent_count, present_count = simulate_attendance()

    inserted = insert_attendance(present_workers)

    print("📊 Attendance Summary:")
    print(f"Total Workers : {num_staff}")
    print(f"Absent        : {absent_count}")
    print(f"Present       : {present_count}")
    print(f"Inserted Rows : {inserted}")

    print("\n✅ Attendance successfully stored in database!")