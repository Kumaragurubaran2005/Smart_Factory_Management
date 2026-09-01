# Smart Factory Management & AI Orchestration System

An end-to-end, AI-powered **Smart Factory Management System** tailored for leather manufacturing and industrial supply chain operations. The system integrates **Deep Reinforcement Learning (DQN)** for production floor worker-machine scheduling and logistics dispatching, **Predictive Machine Learning** for market demand and raw material forecasting, and a **Real-Time Flask/WebSocket Web Application** for factory operations monitoring, order processing, and warehouse management.

---

## 📌 Key Features

### 🤖 1. Reinforcement Learning Operations (DQN)
- **Smart Production Allocation (`production_rl.py`)**: Uses a Deep Q-Network (DQN) agent to dynamically allocate workers to machines (tanning, drying, finishing) based on skill levels, machine degradation state, and order priority.
- **Logistics & Dispatch Optimization (`logistics_rl.py`)**: RL truck agent optimizing shipment batching, truck capacity utilization, and dispatch schedules.
- **Baseline Benchmarking (`baselines.py` & `train_system.py`)**: Comparative training framework evaluating DQN policy performance against Greedy and Random baseline strategies, complete with automated plot generation (`learning_curves.png`).

### 📊 2. Predictive Analytics & Forecasting
- **Market Demand Forecasting (`market_forecast_model.py`)**: Predicts 30-day demand trends and seasonal demand fluctuations using historical order datasets.
- **Raw Material Estimation (`raw_material_model.py`)**: Provides 7-day, 30-day, and 90-day forecast estimates for raw hide/leather material requirements to prevent production bottlenecks.
- **Machine Deterioration & Production Physics (`production_model.py`)**: Models machine wear-and-tear, maintenance degradation, and worker skill productivity multipliers.

### 🏭 3. Real-Time Autonomous Factory Loop
- **Multi-Threaded Autonomous Loop (`app.py`)**: Runs continuous factory production loops synchronizing customer demand, warehouse stock levels, raw material availability, and real-time dispatches.
- **WebSocket Streaming (`flask-sock`)**: Real-time bidirectional telemetry streaming to live web dashboards for instant operational status updates.

### 📦 4. Comprehensive Management Portals
- **Executive Operations Dashboard (`templates/index.html`)**: Real-time KPI summary showing active production output, raw material levels, warehouse stock, and machine status.
- **Order & Pricing Portal (`orders_api.py`, `templates/orders_portal.html`)**: Automated order management system featuring automated pricing calculation for product variants (Belts, Bags, Shoes, Jackets, Wallets), priority surcharges, material requirements, and estimated fulfillment windows.
- **Warehouse & Shipments Dashboard (`templates/warehouse_dashboard.html`)**: Live inventory tracking, stock allocation, and auto-shipment fulfillment logging.
- **Mechanic & Maintenance Portal (`templates/mechanic.html`)**: Machine breakdown reporting, maintenance request queues, and health diagnostics.
- **Worker & Attendance Management (`templates/worker.html`)**: Shift attendance tracking, worker assignment status, and skill ratings.

---

## 🏗 System Architecture

```
                       ┌──────────────────────────────────────┐
                       │      Flask Web Application App       │
                       │     (REST API & WebSocket Server)    │
                       └──────────────────┬───────────────────┘
                                          │
       ┌──────────────────────────────────┼──────────────────────────────────┐
       │                                  │                                  │
┌──────▼─────────────────┐     ┌──────────▼─────────────┐      ┌─────────────▼────────────┐
│ Multi-Agent RL Engine  │     │ Predictive Analytics   │      │ Database & Persistence   │
├────────────────────────┤     ├────────────────────────┤      ├──────────────────────────┤
│ • Production DQN Agent │     │ • Market Demand (30-day│      │ • SQLite (factory.db)    │
│ • Logistics Truck Agent│     │ • Raw Material (7-90d) │      │ • Workers, Maintenance   │
│ • Baseline Algorithms  │     │ • Machine Health Model │      │ • Orders & Inventory     │
└────────────────────────┘     └────────────────────────┘      └──────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── app.py                      # Main Flask app, WebSocket server & autonomous loop
├── orders_api.py               # Blueprint for Order Processing, Inventory & Pricing API
├── database.py                 # SQLite database schema initialization & query helpers
├── production_rl.py            # Deep Q-Network (DQN) implementation for Production
├── logistics_rl.py             # RL environment & Truck Agent for Logistics dispatching
├── production_model.py         # Production environment physics & machine health simulator
├── market_forecast_model.py    # Time-series demand forecasting module
├── raw_material_model.py       # Raw material usage calculation & forecasting
├── baselines.py                # Benchmark algorithms (Greedy & Random strategies)
├── train_system.py             # Unified RL training script for multi-agent optimization
├── factory.db                  # SQLite database instance
├── dataset/                    # Historical CSV datasets
│   ├── attendance.csv          # Worker attendance records
│   ├── employees.csv           # Staff registry & skill ratings
│   ├── leather_orders.csv      # Order history
│   ├── leather_raw_materials.csv# Raw material inventory logs
│   ├── machines.csv            # Machine profiles & base productivity
│   └── raw_material_usage.csv  # Historical material usage logs
├── logs/                       # Trained RL models & training metrics
│   ├── prod_best.keras         # Saved Production DQN weights
│   ├── logistics_best.keras    # Saved Logistics DQN weights
│   ├── train_log.csv           # RL training history log
│   └── learning_curves.png     # Performance evaluation plots
├── templates/                  # Frontend HTML Views (Jinja2)
│   ├── index.html              # Main Executive Dashboard
│   ├── orders_portal.html      # Order Management View
│   ├── place_order.html        # Customer Order Creation Page
│   ├── raw_material_portal.html# Material Stock View
│   ├── warehouse_dashboard.html# Inventory & Shipping Dashboard
│   └── mechanic.html           # Maintenance & Diagnostics Portal
└── static/                     # Static UI styling and JS client scripts
```

---

## 🛠 Tech Stack

- **Backend Framework**: Python 3.9+, Flask, Flask-Sock (WebSockets)
- **Machine Learning & RL**: TensorFlow / Keras, NumPy, Pandas, Matplotlib, OpenCV
- **Database**: SQLite3
- **Frontend**: HTML5, CSS3, JavaScript, WebSockets

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Install Dependencies
Install the required packages using `pip`:
```bash
pip install flask flask-sock tensorflow pandas numpy matplotlib opencv-python
```

### 3. Database Initialization
The database initializes automatically when running `app.py`. Alternatively, you can explicitly initialize the database schema in Python:
```python
from database import init_db
init_db()
```

### 4. Run the Web Application
Start the Flask development server:
```bash
python app.py
```
Open your browser and navigate to:
- **Main Dashboard**: `http://localhost:5000/`
- **Orders Portal**: `http://localhost:5000/orders_portal`
- **Place Order**: `http://localhost:5000/place_order`
- **Warehouse Dashboard**: `http://localhost:5000/warehouse_dashboard`
- **Mechanic Portal**: `http://localhost:5000/mechanic`

---

## 🏋️ Training Reinforcement Learning Agents

To train the Production DQN and Logistics RL agents from scratch or fine-tune existing weights:

```bash
python train_system.py
```

### Training Highlights:
- Evaluates **DQN Policy** vs **Greedy Baseline** and **Random Action Baseline**.
- Automatically updates model checkpoints in `logs/prod_best.keras` and `logs/logistics_best.keras`.
- Generates performance metrics visualization at `logs/learning_curves.png`.

---

## 📝 License & Acknowledgments

Developed as a Machine Learning & Smart Industrial Automation project for end-to-end factory orchestration and AI supply chain management.
