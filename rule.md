# 📏 MuleShield AI — Project Rules & Guidelines (`SIH26184`)

> **Mandatory reading** for every team member before writing a single line of code.
> These rules exist to keep the project consistent, demo-ready, and SIH-compliant.

---

## 🏆 Rule 0 — SIH Context Always Comes First

- This is a **Smart India Hackathon 2026 submission** for Problem Statement **SIH26184**
- Ministry: **MHA / I4C** — every design and copy decision must feel official and credible
- The product must be **demo-able end-to-end in 3 minutes** at the SIH venue
- If a feature conflicts with demo stability → **cut the feature, keep the demo stable**
- If a dependency requires internet and the venue may not have it → **find an offline alternative**

---

## 📁 Rule 1 — Project Structure

All code for this project lives in `d:\SIH 2026\SIHPROJECT2\`. Do not create files outside this folder.

### Mandatory Folder Layout

```
SIHPROJECT2/
├── memory.md           ← Project memory (read before resuming)
├── phases.md           ← Development milestone tracker
├── product.md          ← Product definition & screens
├── requirement.md      ← Functional & non-functional requirements
├── rule.md             ← This file
├── README.md           ← Setup & run instructions (auto-generated last)
│
├── data/               ← All synthetic datasets (CSV)
│   ├── victim_complaints.csv
│   ├── transactions.csv
│   ├── atm_directory.csv
│   ├── graph_edges.csv       ← NEW: edge list for PyG ingestion
│   └── node_features.csv     ← NEW: per-account feature matrix for GNN
│
├── scripts/            ← One-off data generation & utility scripts
│   └── generate_data.py
│
├── engine/             ← Core Python AI/ML & graph modules
│   ├── graph_engine.py       ← NetworkX BFS/DFS + anomaly detection
│   ├── gnn_model.py          ← NEW: GraphSAGE model definition (PyTorch Geometric)
│   ├── train_gnn.py          ← NEW: Offline GNN training script
│   ├── embed.py              ← NEW: Generates & caches node embeddings
│   ├── feature_builder.py    ← NEW: Combines GNN embeddings + tabular features
│   ├── xgb_model.py          ← XGBoost ATM predictor
│   └── train_xgb.py          ← XGBoost training script
│
├── models/             ← Saved trained model artifacts
│   ├── graphsage_mule.pt     ← NEW: GraphSAGE weights (PyTorch)
│   └── xgb_cashout.pkl       ← XGBoost model
│
├── embeddings/         ← NEW: Cached GNN node embeddings
│   └── node_embeddings.pkl   ← Pre-computed 64-dim vectors per account
│
├── backend/            ← FastAPI application
│   ├── main.py
│   ├── routers/
│   │   ├── complaint.py
│   │   ├── graph.py
│   │   ├── embeddings.py     ← NEW: GNN risk score endpoint
│   │   ├── predict.py
│   │   └── freeze.py
│   ├── models/
│   └── websocket.py
│
└── frontend/           ← React + Vite + Tailwind dashboard
    ├── src/
    │   ├── pages/
    │   ├── components/
    │   └── App.jsx
    └── package.json
```

**Rule:** Never put engine code inside `backend/`. Never put backend code inside `frontend/`.

---

## 🐍 Rule 2 — Python Standards

### 2.1 Version & Environment
- Use **Python 3.11+**
- Always use a **virtual environment**: `python -m venv venv` → activate before running anything
- Never install packages globally; always into the venv

### 2.2 Required Libraries (do not substitute without team discussion)
```
# Data
faker==24.x              # Synthetic data generation
pandas==2.x              # Data handling
numpy==1.x               # Numerics
geopy==2.x               # Distance calculations

# Graph & GNN (Phase 2a)
networkx==3.x            # Graph traversal & feature extraction
torch==2.x               # PyTorch (CPU build sufficient for demo)
torch-geometric==2.x     # PyTorch Geometric for GraphSAGE
scikit-learn==1.x        # Preprocessing, metrics, SMOTE
imbalanced-learn==0.x    # SMOTE-ENN for class imbalance

# ML (Phase 2b)
xgboost==2.x             # ATM predictor (real-time inference)
joblib                   # Model serialization (.pkl)

# Backend
fastapi==0.x             # Backend API
uvicorn[standard]        # ASGI server
pydantic==2.x            # Schema validation
websockets               # WebSocket support
```

> ⚠️ **PyTorch Geometric install note:** Install AFTER PyTorch. Use:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install torch-geometric
> ```

### 2.3 Code Style
- Follow **PEP 8** — 4-space indentation, 88-char line limit (Black formatter)
- Every function must have a **docstring** (one-liner minimum)
- No magic numbers — define constants at the top of each file:
  ```python
  MAX_HOP_DEPTH = 5
  VELOCITY_WINDOW_SECONDS = 300  # 5 minutes
  MIN_SPLIT_DESTINATIONS = 3
  ```
- Use **type hints** on all function signatures:
  ```python
  def build_graph(complaint_id: str, transactions_df: pd.DataFrame) -> nx.DiGraph:
  ```

### 2.4 Data Rules
- **No real PII ever** — all names, account numbers, Aadhaar references must be synthetic
- All account numbers must be masked in format `XXXX-XXXX-XXXX`
- All lat/long must be within India: lat `8.0–37.0`, lon `68.0–97.5`
- CSV files must always include a header row
- Timestamps must always be ISO 8601 format: `2026-08-15T14:32:00`

---

## ⚡ Rule 3 — FastAPI Backend Standards

### 3.1 API Design
- All endpoints must be prefixed with `/api/v1/`
- Always use **Pydantic v2 models** for request and response schemas
- HTTP status codes:
  - `200` — Success
  - `201` — Resource created (new complaint)
  - `422` — Validation error (Pydantic auto-handles)
  - `404` — Complaint/Graph not found
  - `500` — Internal server error (catch and log)
- Every endpoint must have a **FastAPI docstring** (shows in `/docs`)

### 3.2 CORS
- Always enable CORS for `http://localhost:5173` (Vite dev server):
  ```python
  app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], ...)
  ```

### 3.3 WebSocket
- WebSocket endpoint: `ws://localhost:8000/ws/feed`
- Broadcast format (JSON string):
  ```json
  {
    "event": "new_complaint",
    "data": { "complaint_id": "...", "victim_name": "...", "amount": 120000 }
  }
  ```
- Always handle client disconnection gracefully (try/except on send)

### 3.4 Run Command
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## ⚛️ Rule 4 — React Frontend Standards

### 4.1 Framework & Tooling
- **React 18** with **Vite** (not Create React App)
- **Tailwind CSS v3** for all styling — no inline styles except for dynamic values (e.g., map coordinates)
- **Lucide React** for all icons (not FontAwesome, not Material Icons)
- **React Router v6** for navigation between the 4 screens

### 4.2 Component Rules
- Every page lives in `src/pages/`
- Every reusable piece lives in `src/components/`
- No component file should exceed 200 lines — split if it does
- Component naming: **PascalCase** (`ComplaintCard.jsx`, `AtmMarker.jsx`)
- Props must be typed with PropTypes or JSDoc comments

### 4.3 Map Rules (Leaflet / React-Leaflet)
- Always use **OpenStreetMap** tiles (no Google Maps API key needed):
  ```
  https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png
  ```
- ATM markers must use a **custom blinking red icon** (CSS animation, not GIF)
- Map center for demo: Delhi NCR → `[28.6139, 77.2090]`, zoom `12`
- Never hardcode map tiles that require an API key

### 4.4 Graph Visualization
- Use **React Flow** (preferred) or **Cytoscape.js** — decide once and commit
- Node color scheme (must be consistent everywhere):
  - 🔵 Blue — Victim account
  - 🟠 Orange — Layer-1 mule
  - 🔴 Red — Layer-2+ mule
  - ⚡ Blinking Red — Terminal (cashout candidate)
- Edge labels: `₹{amount} | +{seconds}s`

### 4.5 API Communication
- Use **Axios** (not fetch) for all REST API calls
- Base URL must come from environment variable:
  ```
  VITE_API_BASE_URL=http://localhost:8000
  ```
- Never hardcode `localhost:8000` in component files

### 4.6 Run Command
```bash
cd frontend && npm run dev
```

---

## 🎨 Rule 5 — UI/UX Design Standards

### 5.1 Color Palette
```css
--color-bg-primary:    #0d1117;   /* Deep dark background */
--color-bg-surface:    #161b22;   /* Card / panel surface */
--color-bg-border:     #30363d;   /* Borders */
--color-accent-red:    #ff3b3b;   /* Critical alert */
--color-accent-orange: #ff8c42;   /* Warning */
--color-accent-green:  #3fb950;   /* Safe / success */
--color-accent-blue:   #58a6ff;   /* Info / victim node */
--color-text-primary:  #e6edf3;   /* Main text */
--color-text-muted:    #8b949e;   /* Secondary text */
```

### 5.2 Typography
- Import and use **Inter** from Google Fonts
- Font sizes:
  - Heading 1: `2rem` / `font-bold`
  - Heading 2: `1.5rem` / `font-semibold`
  - Body: `0.875rem` (14px)
  - Mono (account numbers): `font-mono`

### 5.3 Design Principles
- **Dark theme only** — this is a law enforcement tool, not a consumer app
- Every number that matters (₹ amount, countdown, confidence %) must be **large and prominent**
- Use **blinking animations** only for truly critical elements (terminal ATM markers, critical complaint cards)
- No placeholder images — if an image is needed, generate it or use an icon
- The UI must look **professional enough to demo to a government ministry**

---

## 🧪 Rule 6 — Testing & Quality

### 6.1 Before Any PR / Commit
- [ ] Run `python -m pytest` — all tests pass
- [ ] Run `uvicorn backend.main:app` — starts without errors
- [ ] Run `npm run dev` in `frontend/` — starts without errors
- [ ] Manually test the 3-minute demo flow end-to-end

### 6.2 Minimum Test Coverage
- `engine/graph_engine.py`: Unit tests for BFS traversal, terminal node detection, fund-splitting
- `engine/gnn_model.py`: Unit test for forward pass output shape `(N, 64)`
- `engine/embed.py`: Test that embeddings are generated for all nodes, values in `[-5, 5]` range
- `engine/xgb_model.py`: Unit test for prediction output shape and confidence score range (0–1)
- `engine/feature_builder.py`: Test that `build_feature_vector` returns exactly
  **80** dimensions (64 GNN embedding + 16 spatial/temporal tabular), and that
  `build_ranking_set` returns **24** columns per candidate row
  (12 context + 12 candidate)
- `backend/routers/*.py`: Integration test for each endpoint (use FastAPI `TestClient`)

### 6.3 Demo Stability Rule
> **If it can fail during the demo, add a fallback.**
- If API call fails → show cached/mock data, not an error screen
- If WebSocket disconnects → auto-reconnect silently
- If model inference fails → show an explicit degraded state in the UI.
  **Never substitute pre-computed results for a live inference.** A number on
  screen that might be a fixture is undefendable in front of ministry
  evaluators; an honest "prediction unavailable" is not.

---

## 🚫 Rule 7 — What We Do NOT Do

| ❌ Don't Do | ✅ Do Instead |
|---|---|
| Use real victim/bank data | Use Faker-generated synthetic data only |
| Use Google Maps (API key risk) | Use Leaflet.js + OpenStreetMap |
| Use TailwindCSS v4 | Use TailwindCSS v3 (stable) |
| Use `fetch()` in React | Use Axios with interceptors |
| Hardcode `localhost` in source | Use `VITE_API_BASE_URL` env var |
| Push `.pkl` model files to GitHub without Git LFS | Use Git LFS or store in `models/` with a `.gitignore` note |
| Use real bank API/NPCI feeds | Simulate with CSV + FastAPI mock |
| Build features that can't be demo'd | Build demo-first, polish later |
| Use English-only city/name data | Use Indian-context data (Faker `hi_IN` locale or custom lists) |

---

## 👥 Rule 8 — Team Workflow

### 8.1 Branch Naming
```
feature/phase-1-data-generator
feature/phase-2-graph-engine
feature/phase-3-fastapi-backend
feature/phase-4-react-dashboard
fix/atm-marker-blinking
```

### 8.2 Commit Message Format
```
[Phase N] Short description of what was done

Examples:
[Phase 1] Add victim complaint CSV + node_features generator with 500 records
[Phase 2a] Implement GraphSAGE 2-layer model + offline training script
[Phase 2a] Generate 64-dim node embeddings for all synthetic accounts
[Phase 2b] Build feature_builder.py combining GNN embeddings + 8 tabular features
[Phase 2b] Train XGBoost on 80-dim hybrid features, Top-3=0.57
[Phase 3] Add /api/v1/embeddings endpoint returning GNN risk scores
[Phase 4] Implement React Flow forensic graph with GNN risk score overlay
```

### 8.3 `memory.md` Updates
- **After every work session**, update the `Session Log` table in `memory.md`
- **After every major decision** (tech choice, design change), update the relevant section in `memory.md`
- Treat `memory.md` as the team's shared brain — it must always reflect the current state of the project

### 8.4 `phases.md` Updates
- Update phase status (`🔴 → 🟡 → 🟢`) as work progresses
- Check off acceptance criteria items as they are completed

---

*Last updated: 2026-08-28*
