import pandas as pd
import os

CSV_PATH = "dataset/employees.csv"

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    
    # Assign roles (60% operators, 40% handlers)
    df['role'] = 'operator'
    df.loc[60:, 'role'] = 'handler'
    
    df.to_csv(CSV_PATH, index=False)
    print(f"Updated {CSV_PATH} with roles (60 operators, 40 handlers)")
else:
    print(f"Error: {CSV_PATH} not found")
