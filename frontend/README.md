# MuleShield AI — Frontend (SIH26184)

> Minimalist AEGIS Cyber Command theme — dark, monospace, neon green/cyan, thin panels

## Stack
React 19 + Vite + Tailwind v3 + Lucide + Leaflet + React Flow · Axios · React Router 6

## Run
```bash
cd frontend
npm install
cp .env.example .env  # VITE_API_BASE_URL=http://localhost:8000
npm run dev            # http://localhost:5173
```

Backend must be running:
```bash
uvicorn backend.main:app --reload --port 8000  # /docs, /health, /ws/feed
```

## Screens
- `/` Triage Feed — BDI gauge + live WS complaint stream (🔴<15m 🟡15-30m 🟢>30m) + ingest modal
- `/map?c=TKT-...` Tactical GIS — Leaflet OSM, Top-3 ATM pins with `mm:ss` countdown, PCR route
- `/graph?c=TKT-...` Forensic DAG — React Flow victim→mule→terminal, node inspector, risk overlay
- `/intercept?c=TKT-...` Interception — Freeze (FRZ-...) + Dispatch WhatsApp mock

All API via `VITE_API_BASE_URL`, WS auto-reconnect, fallback mocks if backend offline — demo-stable.
