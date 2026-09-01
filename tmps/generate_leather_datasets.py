import pandas as pd
import numpy as np
import os

def generate_datasets():
    print("Generating Synthetic Leather Industry Datasets...")
    np.random.seed(42)

    # 120 days from Jan 1, 2026
    dates = pd.date_range(start="2026-01-01", periods=120)

    # ---------------------------------------------------------
    # Dataset 1: Orders (Objective 1)
    # ---------------------------------------------------------
    orders = []
    for d in dates:
        for _ in range(np.random.randint(5, 15)):
            orders.append([
                d,
                f"O{np.random.randint(1000,9999)}",
                np.random.choice(["Cow", "Goat", "Sheep"]),
                np.random.choice(["A", "B", "C"]),
                np.random.randint(100, 1000),
                np.random.randint(1, 10)
            ])
            
    orders_df = pd.DataFrame(orders, columns=[
        "date", "order_id", "leather_type", "grade", "quantity", "due_days"
    ])
    orders_df.to_csv("leather_orders.csv", index=False)
    print("- Created leather_orders.csv")

    # ---------------------------------------------------------
    # Dataset 2: Raw Materials (Objective 2)
    # ---------------------------------------------------------
    raw_df = pd.DataFrame({
        "date": dates,
        "stock": np.random.randint(1000, 5000, len(dates)),
        "lead_time": np.random.randint(2, 7, len(dates)),
        "rejection_rate": np.random.uniform(0.05, 0.20, len(dates))
    })
    raw_df.to_csv("leather_raw_materials.csv", index=False)
    print("- Created leather_raw_materials.csv")

    # ---------------------------------------------------------
    # Dataset 3: Production Process (Objective 3)
    # ---------------------------------------------------------
    stages = ["Tanning", "Drying", "Finishing"]
    proc = []
    for d in dates:
        for s in stages:
            proc.append([
                d, s,
                np.random.randint(3, 10),  # machine_available
                np.random.randint(1, 4),   # labor_shifts
                np.random.randint(50, 150) # throughput_per_hour
            ])
            
    proc_df = pd.DataFrame(proc, columns=[
        "date", "stage", "machine_available", "labor_shifts", "throughput"
    ])
    proc_df.to_csv("leather_process.csv", index=False)
    print("- Created leather_process.csv")

    # ---------------------------------------------------------
    # Dataset 4: Historical Output
    # ---------------------------------------------------------
    hist_df = pd.DataFrame({
        "date": dates,
        "produced_quantity": [np.random.randint(300, 1200) for _ in dates]
    })
    hist_df.to_csv("leather_historical.csv", index=False)
    print("- Created leather_historical.csv")
    
    print("\n✅ Data generation complete.")

if __name__ == "__main__":
    generate_datasets()
