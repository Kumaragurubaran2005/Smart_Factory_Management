import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "logs", "raw_material_model.pkl")

MATERIAL_PROFILE = {
    'Belt': {'Hides_kg': 0.35, 'Chrome_Chemicals_L': 0.12, 'Dyes_kg': 0.06, 'Salt_kg': 0.09},
    'Bag': {'Hides_kg': 0.92, 'Chrome_Chemicals_L': 0.35, 'Dyes_kg': 0.14, 'Salt_kg': 0.23},
    'Shoe Leather': {'Hides_kg': 0.58, 'Chrome_Chemicals_L': 0.23, 'Dyes_kg': 0.09, 'Salt_kg': 0.18},
    'Jacket Leather': {'Hides_kg': 1.73, 'Chrome_Chemicals_L': 0.58, 'Dyes_kg': 0.23, 'Salt_kg': 0.46},
    'Wallet': {'Hides_kg': 0.29, 'Chrome_Chemicals_L': 0.09, 'Dyes_kg': 0.05, 'Salt_kg': 0.07},
    'Custom': {'Hides_kg': 1.15, 'Chrome_Chemicals_L': 0.40, 'Dyes_kg': 0.17, 'Salt_kg': 0.29},
}

SEASONAL_FACTOR = {1:0.85,2:0.80,3:0.90,4:0.95,5:1.00,6:1.05,7:0.95,8:0.90,9:1.00,10:1.10,11:1.25,12:1.40}


def generate_synthetic_usage_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    records = []

    for date in dates:
        s_mul = SEASONAL_FACTOR.get(date.month, 1.0)

        for ptype, profile in MATERIAL_PROFILE.items():
            qty = np.random.randint(30, 220) * s_mul

            row = {
                'date': date,
                'product_type': ptype,
                'quantity_produced': round(qty, 0)
            }

            for mat, per_unit in profile.items():
                row[mat] = round(qty * per_unit * np.random.uniform(1.05, 1.30), 2)

            records.append(row)

    df = pd.DataFrame(records)
    os.makedirs(os.path.join(PROJECT_ROOT, 'dataset'), exist_ok=True)
    df.to_csv(os.path.join(PROJECT_ROOT, 'dataset', 'raw_material_usage.csv'), index=False)

    return df


def train_model(force_retrain=False):
    csv_path = os.path.join(PROJECT_ROOT, 'dataset', 'raw_material_usage.csv')

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
    else:
        df = generate_synthetic_usage_data()

    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    for lag in [1, 7, 30]:
        df[f'orders_lag_{lag}'] = df['quantity_produced'].shift(lag)

    df.dropna(inplace=True)

    features = ['dayofweek', 'month', 'orders_lag_1', 'orders_lag_7', 'orders_lag_30']
    targets = ['Hides_kg', 'Chrome_Chemicals_L', 'Dyes_kg', 'Salt_kg']

    models = {}

    for target in targets:
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        print(f"📦 {target} MAE:", round(mae, 2))

        models[target] = model

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(models, f)

    return models






def calculate_material_needs(orders_list):
    totals = {m:0.0 for m in ['Hides_kg','Chrome_Chemicals_L','Dyes_kg','Salt_kg']}
    for order in orders_list:
        ptype = order.get('product_type','Custom')
        qty = float(order.get('quantity',0))
        profile = MATERIAL_PROFILE.get(ptype, MATERIAL_PROFILE['Custom'])
        for mat, per_unit in profile.items():
            totals[mat] += qty * per_unit
    return totals

def predict_needs(current_inventory, recent_orders, forecast_days=30):
    import datetime
    month = datetime.datetime.now().month
    season_mul = SEASONAL_FACTOR.get(month, 1.0)
    raw_needed = calculate_material_needs(recent_orders)
    days_sample = max(len(recent_orders) / max(1, len(set(o.get('product_type') for o in recent_orders))), 1)
    daily_rates = {m: v / max(days_sample,1) for m,v in raw_needed.items()}
    projected = {m: daily_rates[m] * forecast_days * season_mul for m in daily_rates}
    INV_MAP = {'Hides_kg':'Hides','Chrome_Chemicals_L':'Chrome Chemicals','Dyes_kg':'Dyes','Salt_kg':'Salt'}
    result = {}
    for mat, needed in projected.items():
        inv_key = INV_MAP.get(mat, mat)
        current = float(current_inventory.get(inv_key, 0))
        daily_use = daily_rates[mat] * season_mul
        days_of_stock = round(current / daily_use, 1) if daily_use>0 else 999
        shortfall = max(0.0, needed - current)
        result[mat] = {
            'material': mat,
            'unit': 'L' if '_L' in mat else 'kg',
            'total_needed': round(needed,1),
            'current_stock': round(current,1),
            'shortfall': round(shortfall,1),
            'order_quantity': round(shortfall*1.25,1),
            'days_of_stock': min(days_of_stock,999),
        }
    return result

def get_7_30_90_predictions(current_inventory, recent_orders):
    return {
        'days_7': predict_needs(current_inventory, recent_orders, 7),
        'days_30': predict_needs(current_inventory, recent_orders, 30),
        'days_90': predict_needs(current_inventory, recent_orders, 90),
    }
if __name__ == "__main__":
    train_model()
    print("Raw material model trained and saved.")