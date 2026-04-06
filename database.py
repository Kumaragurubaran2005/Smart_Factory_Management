import sqlite3
import pandas as pd
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

DB_NAME = "factory.db"

@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_db_connection():
    """Return a plain connection (for pandas)."""
    return sqlite3.connect(DB_NAME)

def init_db():
    """Create all tables if they don't exist."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # ========== EXISTING TABLES ==========
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Workers (
                worker_id TEXT PRIMARY KEY,
                name TEXT,
                overall_skill INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Mechanics (
                mechanic_id TEXT PRIMARY KEY,
                name TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Trucks (
                truck_id TEXT PRIMARY KEY,
                driver_name TEXT,
                capacity REAL,
                status TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT,
                timestamp TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS WorkerHistory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT,
                machine TEXT,
                start_time TEXT,
                end_time TEXT,
                duration REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Overtime_Requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT,
                date TEXT,
                status TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT,
                quantity INTEGER,
                deadline TEXT,
                status TEXT,
                created_at TEXT
            )
        ''')

        # ========== NEW TABLES FOR ORDER & SUPPLY CHAIN ==========
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS CustomerOrders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_ref TEXT UNIQUE,
                customer_name TEXT NOT NULL,
                customer_email TEXT,
                product_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit TEXT DEFAULT 'pieces',
                priority TEXT DEFAULT 'Normal',
                deadline TEXT,
                status TEXT DEFAULT 'pending',
                notes TEXT,
                estimated_cost REAL,
                estimated_completion TEXT,
                actual_completion TEXT,
                raw_material_needed REAL,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS OrderAllocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                worker_id TEXT,
                worker_name TEXT,
                machine_id TEXT,
                machine_type TEXT,
                allocated_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (order_id) REFERENCES CustomerOrders(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Shipments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                warehouse_zone TEXT DEFAULT 'Zone-A',
                quantity INTEGER,
                status TEXT DEFAULT 'pending',
                shipped_at TEXT,
                delivered_at TEXT,
                destination TEXT DEFAULT 'Main Warehouse',
                FOREIGN KEY (order_id) REFERENCES CustomerOrders(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Warehouse (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_type TEXT UNIQUE,
                quantity INTEGER DEFAULT 0,
                min_stock INTEGER DEFAULT 50,
                max_capacity INTEGER DEFAULT 5000,
                zone TEXT DEFAULT 'Zone-A',
                last_updated TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS RawMaterialOrders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit TEXT DEFAULT 'kg',
                supplier TEXT,
                urgency TEXT DEFAULT 'Normal',
                status TEXT DEFAULT 'ordered',
                estimated_delivery TEXT,
                cost REAL,
                notes TEXT,
                created_at TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS RawMaterialInventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_type TEXT UNIQUE,
                quantity REAL DEFAULT 0,
                unit TEXT DEFAULT 'kg',
                min_stock REAL DEFAULT 100,
                last_updated TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS MarketForecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_date TEXT,
                product_type TEXT,
                predicted_demand INTEGER,
                lower_bound INTEGER,
                upper_bound INTEGER,
                scenario TEXT DEFAULT 'base',
                created_at TEXT
            )
        ''')

        # ========== SEED DATA ==========
        now = datetime.now().isoformat()
        # Warehouse
        for pt in ['Belt', 'Bag', 'Shoe Leather', 'Jacket Leather', 'Wallet', 'Custom']:
            cursor.execute('''
                INSERT OR IGNORE INTO Warehouse (product_type, quantity, min_stock, max_capacity, last_updated)
                VALUES (?, 0, 50, 5000, ?)
            ''', (pt, now))
        # Raw material inventory
        for mat, qty, unit, min_stk in [
            ('Hides', 2000, 'kg', 500),
            ('Chrome Chemicals', 1500, 'liters', 300),
            ('Dyes', 800, 'kg', 200),
            ('Salt', 3000, 'kg', 500),
            ('Conditioning Agents', 600, 'liters', 150)
        ]:
            cursor.execute('''
                INSERT OR IGNORE INTO RawMaterialInventory (material_type, quantity, unit, min_stock, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (mat, qty, unit, min_stk, now))

        conn.commit()
        logger.info("Database initialized with all tables")

# ========== HELPER FUNCTIONS ==========

def get_available_workers():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT worker_id FROM Attendance WHERE date(timestamp) = date('now', 'localtime')")
        workers = [row[0] for row in cursor.fetchall()]
        if not workers:
            cursor.execute("SELECT worker_id FROM Workers LIMIT 4")
            workers = [row[0] for row in cursor.fetchall()]
        return workers

def get_worker_skills(worker_ids=None):
    with get_db() as conn:
        if worker_ids:
            placeholders = ','.join(['?'] * len(worker_ids))
            df = pd.read_sql_query(f"SELECT * FROM Workers WHERE worker_id IN ({placeholders})", conn, params=worker_ids)
        else:
            df = pd.read_sql_query("SELECT * FROM Workers", conn)
        return df

def get_machines():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "dataset", "machines.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame({
        "machine_id": ["M1", "M2", "M3"],
        "type": ["tanning", "drying", "finishing"],
        "baseProductivity": [100, 90, 85],
        "initialWear": [20, 30, 25]
    })

def get_machine_health():
    df = get_machines()
    if df.empty:
        return {}
    health_dict = {}
    for _, row in df.iterrows():
        wear = float(row.get('initialWear', 0))
        health = max(0.1, min(1.0, (100.0 - wear) / 100.0))
        health_dict[row['machine_id']] = health
    return health_dict

def get_trucks():
    with get_db() as conn:
        return pd.read_sql_query("SELECT * FROM Trucks WHERE status='available'", conn)

def get_orders():
    with get_db() as conn:
        return pd.read_sql_query("SELECT * FROM Orders ORDER BY created_at DESC", conn)

def add_order(customer_name, quantity, deadline):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Orders (customer_name, quantity, deadline, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (customer_name, quantity, deadline, 'pending', datetime.now().isoformat()))
        conn.commit()
        return cursor.lastrowid

def get_worker_fatigue(worker_id: str) -> float:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT date(timestamp)) FROM Attendance
            WHERE worker_id = ? AND date(timestamp) >= date('now', '-7 days')
        """, (worker_id,))
        result = cursor.fetchone()
        days_worked = result[0] if result and result[0] is not None else 0
        return min(1.0, days_worked / 7.0)

# ========== NEW ORDER & SUPPLY CHAIN FUNCTIONS ==========

def add_customer_order(order_data):
    order_ref = f"ORD-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    now = datetime.now().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO CustomerOrders
        (order_ref, customer_name, customer_email, product_type, quantity, unit, priority, deadline,
         notes, estimated_cost, estimated_completion, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        order_ref, order_data['customer_name'], order_data.get('customer_email'),
        order_data['product_type'], order_data['quantity'], order_data.get('unit', 'pieces'),
        order_data.get('priority', 'Normal'), order_data.get('deadline'),
        order_data.get('notes'), order_data.get('estimated_cost'),
        order_data.get('estimated_completion'), 'pending', now, now
    ))
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return order_id, order_ref

def get_all_customer_orders():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM CustomerOrders ORDER BY created_at DESC", conn)
    conn.close()
    return df

def get_order_by_id(order_id):
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM CustomerOrders WHERE id = ?", conn, params=(order_id,))
    conn.close()
    return df.iloc[0] if not df.empty else None

def update_order_status(order_id, status, actual_completion=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE CustomerOrders SET status = ?, updated_at = ? WHERE id = ?",
                   (status, datetime.now().isoformat(), order_id))
    if actual_completion:
        cursor.execute("UPDATE CustomerOrders SET actual_completion = ? WHERE id = ?",
                       (actual_completion, order_id))
    conn.commit()
    conn.close()

def add_order_allocation(order_id, worker_id, worker_name, machine_id, machine_type):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO OrderAllocations (order_id, worker_id, worker_name, machine_id, machine_type, allocated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (order_id, worker_id, worker_name, machine_id, machine_type, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def add_shipment(order_id, quantity, destination='Main Warehouse'):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Shipments (order_id, quantity, status, shipped_at, destination)
        VALUES (?, ?, ?, ?, ?)
    ''', (order_id, quantity, 'shipped', datetime.now().isoformat(), destination))
    conn.commit()
    conn.close()

def update_warehouse_stock(product_type, quantity_change):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE Warehouse SET quantity = quantity + ?, last_updated = ? WHERE product_type = ?",
                   (quantity_change, datetime.now().isoformat(), product_type))
    conn.commit()
    conn.close()

def get_warehouse_stock():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT product_type, quantity, min_stock, max_capacity FROM Warehouse", conn)
    conn.close()
    return df

def add_raw_material_order(material_type, quantity, supplier, urgency, cost=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO RawMaterialOrders (material_type, quantity, supplier, urgency, status, created_at, cost)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (material_type, quantity, supplier, urgency, 'ordered', datetime.now().isoformat(), cost))
    conn.commit()
    conn.close()

def get_raw_material_inventory():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT material_type, quantity, min_stock FROM RawMaterialInventory", conn)
    conn.close()
    return df

def update_raw_material_stock(material_type, quantity_change):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE RawMaterialInventory SET quantity = quantity + ?, last_updated = ? WHERE material_type = ?",
                   (quantity_change, datetime.now().isoformat(), material_type))
    conn.commit()
    conn.close()

def get_raw_material_orders():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM RawMaterialOrders ORDER BY created_at DESC", conn)
    conn.close()
    return df

def save_market_forecast(forecast_date, product_type, predicted_demand, lower_bound, upper_bound, scenario='base'):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO MarketForecasts (forecast_date, product_type, predicted_demand, lower_bound, upper_bound, scenario, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (forecast_date, product_type, predicted_demand, lower_bound, upper_bound, scenario, datetime.now().isoformat()))
    conn.commit()
    conn.close()