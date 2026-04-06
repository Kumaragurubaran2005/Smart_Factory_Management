"""
Orders / Warehouse / Raw-Material Blueprint
All new routes for the leather-industry order management system.
Import and register with:  app.register_blueprint(orders_bp)
"""
from flask import Blueprint, jsonify, request, render_template, session, redirect, url_for
import sqlite3, pandas as pd, numpy as np, logging, datetime, os

logger = logging.getLogger(__name__)
orders_bp = Blueprint('orders', __name__)

DB_NAME     = 'factory.db'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PRODUCT_TYPES  = ['Belt', 'Bag', 'Shoe Leather', 'Jacket Leather', 'Wallet', 'Custom']
PRIORITY_ORDER = {'Urgent': 1, 'High': 2, 'Normal': 3, 'Low': 4}
PRICE_PER_UNIT = {'Belt': 25, 'Bag': 85, 'Shoe Leather': 40,
                  'Jacket Leather': 120, 'Wallet': 30, 'Custom': 60}
PRIORITY_SURCHARGE = {'Urgent': 1.25, 'High': 1.10, 'Normal': 1.00, 'Low': 0.95}
BASE_DAYS = {'Belt': 2, 'Bag': 3, 'Shoe Leather': 2,
             'Jacket Leather': 5, 'Wallet': 2, 'Custom': 4}
PRIORITY_SPEED = {'Urgent': 0.5, 'High': 0.75, 'Normal': 1.0, 'Low': 1.5}
MATERIAL_KG = {'Belt': 0.35, 'Bag': 0.92, 'Shoe Leather': 0.58,
               'Jacket Leather': 1.73, 'Wallet': 0.29, 'Custom': 1.15}

# ── Helpers ─────────────────────────────────────────────────────────────────
def _db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def _admin_required():
    return session.get('role') == 'admin'

def _gen_ref():
    conn = _db()
    n = conn.execute('SELECT COUNT(*) FROM CustomerOrders').fetchone()[0]
    conn.close()
    return f"ORD-{datetime.datetime.now().year}-{str(n + 1).zfill(4)}"

def _estimate_cost(product, qty, priority):
    return round(PRICE_PER_UNIT.get(product, 50) * qty *
                 PRIORITY_SURCHARGE.get(priority, 1.0), 2)

def _estimate_days(product, qty, priority):
    base = BASE_DAYS.get(product, 3)
    qty_f = max(1, qty / 100)
    return max(1, min(int(base * qty_f * PRIORITY_SPEED.get(priority, 1.0)), 90))

def _raw_needed(product, qty):
    return round(MATERIAL_KG.get(product, 1.0) * qty * 1.15, 2)

def _now():
    return datetime.datetime.now().isoformat()

# ── Page Routes ─────────────────────────────────────────────────────────────
@orders_bp.route('/orders_portal')
def orders_portal():
    if not _admin_required():
        return redirect(url_for('login_page'))
    return render_template('orders_portal.html')

@orders_bp.route('/order_detail/<int:order_id>')
def order_detail(order_id):
    if not _admin_required():
        return redirect(url_for('login_page'))
    return render_template('order_detail.html', order_id=order_id)

@orders_bp.route('/warehouse_dashboard')
def warehouse_dashboard():
    if not _admin_required():
        return redirect(url_for('login_page'))
    return render_template('warehouse_dashboard.html')

@orders_bp.route('/raw_material_portal')
def raw_material_portal():
    if not _admin_required():
        return redirect(url_for('login_page'))
    return render_template('raw_material_portal.html')

# ── Order APIs ───────────────────────────────────────────────────────────────
@orders_bp.route('/api/orders/place', methods=['POST'])
def place_order():
    try:
        d = request.get_json()
        name    = d.get('customer_name', '').strip()
        email   = d.get('customer_email', '')
        product = d.get('product_type', 'Custom')
        qty     = int(d.get('quantity', 1))
        unit    = d.get('unit', 'pieces')
        prio    = d.get('priority', 'Normal')
        dl      = d.get('deadline', '')
        notes   = d.get('notes', '')

        if not name or qty < 1:
            return jsonify({'error': 'customer_name and quantity are required'}), 400

        ref      = _gen_ref()
        cost     = _estimate_cost(product, qty, prio)
        days     = _estimate_days(product, qty, prio)
        est_comp = (datetime.datetime.now() + datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        raw      = _raw_needed(product, qty)
        now      = _now()

        conn = _db(); c = conn.cursor()
        c.execute('''
            INSERT INTO CustomerOrders
            (order_ref, customer_name, customer_email, product_type, quantity, unit,
             priority, deadline, status, notes, estimated_cost, estimated_completion,
             raw_material_needed, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,'pending',?,?,?,?,?,?)
        ''', (ref, name, email, product, qty, unit, prio, dl,
              notes, cost, est_comp, raw, now, now))
        conn.commit(); oid = c.lastrowid; conn.close()

        return jsonify({'success': True, 'order_id': oid, 'order_ref': ref,
                        'estimated_cost': cost, 'estimated_completion': est_comp,
                        'raw_material_needed_kg': raw,
                        'message': f'Order {ref} placed successfully'}), 201
    except Exception as e:
        logger.error(f"place_order: {e}")
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/orders/list')
def list_orders():
    try:
        conn = _db()

        orders_df = pd.read_sql(
            "SELECT * FROM CustomerOrders ORDER BY created_at DESC",
            conn
        )

        if orders_df.empty:
            conn.close()
            return jsonify([])

        shipped_df = pd.read_sql("""
            SELECT order_id, COALESCE(SUM(quantity), 0) AS shipped_quantity
            FROM Shipments
            GROUP BY order_id
        """, conn)

        orders_df = orders_df.merge(
            shipped_df,
            left_on='id',
            right_on='order_id',
            how='left'
        )

        orders_df['shipped_quantity'] = orders_df['shipped_quantity'].fillna(0).astype(int)

        conn.close()

        import numpy as np

        import math

        records = orders_df.to_dict('records')

        # CLEAN EVERY VALUE MANUALLY
        for row in records:
            for k, v in row.items():
                if isinstance(v, float) and math.isnan(v):
                    row[k] = None

        return jsonify(records)

    except Exception as e:
        logger.error(f"list_orders: {e}")
        return jsonify([]), 500

@orders_bp.route('/api/orders/<int:oid>')
def get_order(oid):
    try:
        conn = _db()
        order = conn.execute('SELECT * FROM CustomerOrders WHERE id=?', (oid,)).fetchone()
        if not order:
            conn.close(); return jsonify({'error': 'Not found'}), 404
        allocs = conn.execute('SELECT * FROM OrderAllocations WHERE order_id=?', (oid,)).fetchall()
        ships  = conn.execute('SELECT * FROM Shipments WHERE order_id=?', (oid,)).fetchall()
        conn.close()
        return jsonify({'order': dict(order), 'allocations': [dict(a) for a in allocs],
                        'shipments': [dict(s) for s in ships]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/orders/<int:oid>/process', methods=['POST'])
def process_order(oid):
    """Allocate workers + machines to an order → in_production."""
    try:
        conn = _db()
        order = conn.execute('SELECT * FROM CustomerOrders WHERE id=?', (oid,)).fetchone()
        if not order:
            conn.close(); return jsonify({'error': 'Not found'}), 404
        order = dict(order)
        if order['status'] not in ('pending', 'allocated'):
            conn.close()
            return jsonify({'error': f"Cannot process: status is '{order['status']}'"}), 400

        now = _now()
        # Load workers
        try:
            staff = pd.read_csv(os.path.join(PROJECT_ROOT, 'dataset', 'employees.csv')).head(3).to_dict('records')
        except Exception:
            staff = [{'worker_id': 'W001', 'name': 'Worker Alpha', 'overall_skill': 85},
                     {'worker_id': 'W002', 'name': 'Worker Beta',  'overall_skill': 78}]
        # Load machines
        try:
            machines = pd.read_csv(os.path.join(PROJECT_ROOT, 'dataset', 'machines.csv')).head(3).to_dict('records')
        except Exception:
            machines = [{'machine_id': 'M-101', 'type': 'Tanning Unit'},
                        {'machine_id': 'M-102', 'type': 'Drying Chamber'},
                        {'machine_id': 'M-103', 'type': 'Finishing Line'}]

        c = conn.cursor()
        c.execute('DELETE FROM OrderAllocations WHERE order_id=?', (oid,))
        n = min(len(staff), len(machines), 3)
        for i in range(n):
            w, m = staff[i], machines[i]
            c.execute('''
                INSERT INTO OrderAllocations
                (order_id, worker_id, worker_name, machine_id, machine_type, allocated_at)
                VALUES (?,?,?,?,?,?)
            ''', (oid, w['worker_id'], w['name'],
                  str(m.get('machine_id', f'M-{i+1}')),
                  str(m.get('type', 'Machine')), now))

        days     = _estimate_days(order['product_type'], order['quantity'], order['priority'])
        est_comp = (datetime.datetime.now() + datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        c.execute('''
            UPDATE CustomerOrders
            SET status='in_production', estimated_completion=?, updated_at=? WHERE id=?
        ''', (est_comp, now, oid))
        conn.commit(); conn.close()

        return jsonify({'success': True,
                        'message': f"Order {order['order_ref']} in production",
                        'workers_allocated': n, 'estimated_completion': est_comp})
    except Exception as e:
        logger.error(f"process_order: {e}")
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/orders/<int:oid>/ready', methods=['POST'])
def mark_ready(oid):
    try:
        now = _now()
        conn = _db()
        conn.execute('UPDATE CustomerOrders SET status=?, actual_completion=?, updated_at=? WHERE id=?',
                     ('ready', now, now, oid))
        conn.commit(); conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/orders/<int:oid>/ship', methods=['POST'])
def ship_order(oid):
    try:
        conn = _db()
        order = conn.execute('SELECT * FROM CustomerOrders WHERE id=?', (oid,)).fetchone()
        if not order:
            conn.close(); return jsonify({'error': 'Not found'}), 404
        order = dict(order)
        now = _now()
        c = conn.cursor()
        zone = f"Zone-{chr(65 + (oid % 4))}"
        c.execute('''
            INSERT INTO Shipments (order_id, warehouse_zone, quantity, status, shipped_at, destination)
            VALUES (?,?,?,'in_transit',?,'Main Warehouse')
        ''', (oid, zone, order['quantity'], now))
        c.execute('''
            UPDATE Warehouse SET quantity=quantity+?, last_updated=? WHERE product_type=?
        ''', (order['quantity'], now, order['product_type']))
        c.execute('UPDATE CustomerOrders SET status=?, updated_at=? WHERE id=?', ('shipped', now, oid))
        conn.commit(); conn.close()
        return jsonify({'success': True, 'message': f"Shipped {order['order_ref']} to {zone}"})
    except Exception as e:
        logger.error(f"ship_order: {e}")
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/orders/<int:oid>/cancel', methods=['POST'])
def cancel_order(oid):
    try:
        now = _now()
        conn = _db()
        conn.execute('UPDATE CustomerOrders SET status=?, updated_at=? WHERE id=?', ('cancelled', now, oid))
        conn.commit(); conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/orders/stats')
def order_stats():
    try:
        conn = _db()
        by_status   = dict(conn.execute('SELECT status, COUNT(*) FROM CustomerOrders GROUP BY status').fetchall())
        by_priority = dict(conn.execute('SELECT priority, COUNT(*) FROM CustomerOrders GROUP BY priority').fetchall())
        by_product  = dict(conn.execute('SELECT product_type, COUNT(*) FROM CustomerOrders GROUP BY product_type').fetchall())
        revenue     = conn.execute("SELECT COALESCE(SUM(estimated_cost),0) FROM CustomerOrders WHERE status!='cancelled'").fetchone()[0]
        conn.close()
        return jsonify({'by_status': by_status, 'by_priority': by_priority,
                        'by_product': by_product, 'total_revenue': round(revenue, 2),
                        'total_orders': sum(by_status.values())})
    except Exception as e:
        return jsonify({}), 500



# ── Warehouse APIs ────────────────────────────────────────────────────────────
@orders_bp.route('/api/warehouse')
def get_warehouse():
    try:
        conn = _db()
        stock = [dict(r) for r in conn.execute('SELECT * FROM Warehouse ORDER BY product_type').fetchall()]
        ships = [dict(r) for r in conn.execute('''
            SELECT s.*, co.order_ref, co.customer_name, co.product_type
            FROM Shipments s JOIN CustomerOrders co ON s.order_id = co.id
            ORDER BY s.shipped_at DESC LIMIT 30
        ''').fetchall()]
        total = sum(s['quantity'] for s in stock)
        conn.close()
        return jsonify({'stock': stock, 'recent_shipments': ships, 'total_items': total})
    except Exception as e:
        return jsonify({'stock': [], 'recent_shipments': [], 'total_items': 0}), 500


@orders_bp.route('/api/warehouse/update', methods=['POST'])
def update_warehouse():
    try:
        d   = request.get_json()
        pt  = d.get('product_type')
        qty = int(d.get('quantity', 0))
        op  = d.get('operation', 'set')
        now = _now()
        conn = _db()
        if op == 'add':
            conn.execute('UPDATE Warehouse SET quantity=quantity+?, last_updated=? WHERE product_type=?', (qty, now, pt))
        elif op == 'subtract':
            conn.execute('UPDATE Warehouse SET quantity=MAX(0,quantity-?), last_updated=? WHERE product_type=?', (qty, now, pt))
        else:
            conn.execute('UPDATE Warehouse SET quantity=?, last_updated=? WHERE product_type=?', (qty, now, pt))
        conn.commit(); conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Raw Material APIs ─────────────────────────────────────────────────────────
@orders_bp.route('/api/raw_material/inventory')
def raw_inventory():
    try:
        conn = _db()
        inv = [dict(r) for r in conn.execute('SELECT * FROM RawMaterialInventory').fetchall()]
        conn.close()
        return jsonify(inv)
    except Exception as e:
        return jsonify([]), 500


@orders_bp.route('/api/raw_material/needs')
def raw_needs():
    try:
        from raw_material_model import get_7_30_90_predictions
        conn = _db()
        inv_rows   = conn.execute('SELECT * FROM RawMaterialInventory').fetchall()
        curr_inv   = {r['material_type']: r['quantity'] for r in inv_rows}
        orders_raw = conn.execute('''
            SELECT product_type, quantity FROM CustomerOrders
            WHERE status NOT IN ('cancelled') ORDER BY created_at DESC LIMIT 50
        ''').fetchall()
        conn.close()
        orders_list = [{'product_type': o['product_type'], 'quantity': o['quantity']} for o in orders_raw]
        if not orders_list:
            orders_list = [{'product_type': 'Shoe Leather', 'quantity': 100}]
        return jsonify(get_7_30_90_predictions(curr_inv, orders_list))
    except Exception as e:
        logger.error(f"raw_needs: {e}")
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/raw_material/order', methods=['POST'])
def place_raw_order():
    try:
        d       = request.get_json()
        mat     = d.get('material_type', '').strip()
        qty     = float(d.get('quantity', 0))
        unit    = d.get('unit', 'kg')
        supplier= d.get('supplier', 'Default Supplier')
        urgency = d.get('urgency', 'Normal')
        notes   = d.get('notes', '')
        if not mat or qty <= 0:
            return jsonify({'error': 'material_type and quantity required'}), 400

        cost_map = {'Hides': 8.5, 'Chrome Chemicals': 3.2, 'Dyes': 12.0,
                    'Salt': 0.5, 'Conditioning Agents': 4.8}
        cost = round(qty * cost_map.get(mat, 5.0), 2)
        days_map = {'Urgent': 2, 'High': 5, 'Normal': 10, 'Low': 21}
        est_del = (datetime.datetime.now() + datetime.timedelta(days=days_map.get(urgency, 10))).strftime('%Y-%m-%d')
        now = _now()
        conn = _db(); c = conn.cursor()
        c.execute('''
            INSERT INTO RawMaterialOrders
            (material_type, quantity, unit, supplier, urgency, status, estimated_delivery, cost, notes, created_at)
            VALUES (?,?,?,?,?,'ordered',?,?,?,?)
        ''', (mat, qty, unit, supplier, urgency, est_del, cost, notes, now))
        conn.commit(); roid = c.lastrowid; conn.close()
        return jsonify({'success': True, 'order_id': roid, 'estimated_delivery': est_del,
                        'estimated_cost': cost,
                        'message': f"Procurement: {qty} {unit} of {mat} ordered"}), 201
    except Exception as e:
        logger.error(f"place_raw_order: {e}")
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/api/raw_material/orders')
def list_raw_orders():
    try:
        conn = _db()
        rows = [dict(r) for r in conn.execute(
            'SELECT * FROM RawMaterialOrders ORDER BY created_at DESC').fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify([]), 500


@orders_bp.route('/api/raw_material/receive/<int:roid>', methods=['POST'])
def receive_raw(roid):
    try:
        conn = _db(); c = conn.cursor()
        order = conn.execute('SELECT * FROM RawMaterialOrders WHERE id=?', (roid,)).fetchone()
        if not order:
            conn.close(); return jsonify({'error': 'Not found'}), 404
        order = dict(order)
        now = _now()
        c.execute('UPDATE RawMaterialInventory SET quantity=quantity+?, last_updated=? WHERE material_type=?',
                  (order['quantity'], now, order['material_type']))
        c.execute('UPDATE RawMaterialOrders SET status=? WHERE id=?', ('delivered', roid))
        conn.commit(); conn.close()
        return jsonify({'success': True,
                        'message': f"Received {order['quantity']} {order['unit']} of {order['material_type']}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Feasibility Check (called by place_order form) ────────────────────────────
@orders_bp.route('/api/orders/feasibility', methods=['POST'])
def check_feasibility():
    """Quick AI feasibility check before placing an order."""
    try:
        d       = request.get_json()
        product = d.get('product_type', 'Custom')
        qty     = int(d.get('quantity', 0))
        prio    = d.get('priority', 'Normal')
        deadline= d.get('deadline', '')

        days     = _estimate_days(product, qty, prio)
        cost     = _estimate_cost(product, qty, prio)
        raw      = _raw_needed(product, qty)
        est_comp = (datetime.datetime.now() + datetime.timedelta(days=days)).strftime('%Y-%m-%d')

        # Check raw material availability
        conn = _db()
        inv = {r['material_type']: r['quantity']
               for r in conn.execute('SELECT * FROM RawMaterialInventory').fetchall()}
        conn.close()
        raw_ok = inv.get('Hides', 9999) >= raw * 0.8

        # Check deadline feasibility
        feasible = True
        deadline_warning = ''
        if deadline:
            try:
                dl_date = datetime.datetime.strptime(deadline, '%Y-%m-%d')
                comp_date = datetime.datetime.now() + datetime.timedelta(days=days)
                if comp_date > dl_date:
                    feasible = False
                    deadline_warning = f"⚠️ Estimated completion {est_comp} may miss deadline."
            except Exception:
                pass

        return jsonify({
            'feasible': feasible,
            'estimated_days': days,
            'estimated_completion': est_comp,
            'estimated_cost': cost,
            'raw_material_needed_kg': raw,
            'raw_material_available': raw_ok,
            'deadline_warning': deadline_warning,
            'workers_needed': min(3, max(1, qty // 50)),
            'machines_needed': min(3, max(1, qty // 80)),
        })
    except Exception as e:
        logger.error(f"feasibility: {e}")
        return jsonify({'error': str(e)}), 500


# ── Market Forecast API ───────────────────────────────────────────────────────
@orders_bp.route('/api/market/forecast')
def market_forecast():
    try:
        from market_forecast_model import forecast_30_days
        return jsonify(forecast_30_days())
    except Exception as e:
        logger.error(f"market_forecast: {e}")
        return jsonify({'error': str(e)}), 500