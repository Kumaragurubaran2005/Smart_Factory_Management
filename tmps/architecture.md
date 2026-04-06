# Smart AI-Based Workforce Allocation and Predictive Maintenance System

## Architecture Diagram

The system connects real-time worker face recognition with a dynamic workforce allocation engine.

```mermaid
graph TD
    A[Camera Feed] --> B[Face Recognition Module]
    B -->|Worker ID & Confidence| C{Confidence Filter >= 0.85}
    C -->|Pass| D[(SQLite Database: factory.db)]
    C -->|Fail| Z[Reject]
    
    D -->|Daily Attendance Logs| E[Workforce Availability Matrix]
    
    subgraph Optimization Engine
        F((PuLP LP Model))
        E --> F
        G[Worker Skill Matrix] --> F
        H[Machine Requirements] --> F
    end
    
    F -->|Optimized Allocation| I[Streamlit Live Dashboard]
    D -->|Stats & Logs| I
```

## Setup & Execution

### 1. Requirements
```bash
pip install pandas pulp streamlit
```

### 2. Initialization & Pipeline
You can simulate the camera pipeline continuously by running:
```bash
python main_system.py
```

### 3. Live Dashboard
Once the database `factory.db` is initialized, run the dashboard to view real-time smart allocations:
```bash
streamlit run dashboard.py
```
