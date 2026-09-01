# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: FastAPI Application Entry Point
SIH26184 | MHA / I4C

Start server:
    uvicorn backend.main:app --reload --port 8000

API docs (Swagger UI):
    http://localhost:8000/docs

Endpoints:
    POST   /api/v1/complaint/ingest
    GET    /api/v1/complaint/list
    GET    /api/v1/complaint/{id}
    GET    /api/v1/graph/{complaint_id}
    GET    /api/v1/embeddings/{complaint_id}
    GET    /api/v1/predict/cashout/{complaint_id}
    POST   /api/v1/bank/micro-freeze
    WS     /ws/feed
    GET    /health
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import backend.state as state
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.routers import (auth, complaint, graph, embeddings, predict,
                             freeze, audit, intel, hotspot)
from backend import db
from backend.websocket import manager

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("muleshield.main")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — Startup / Shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all data and ML models on startup."""
    logger.info("=" * 60)
    logger.info("  MuleShield AI Backend | SIH26184 | MHA / I4C")
    logger.info("=" * 60)
    state.load_all()
    # Credentials are the one thing in this system that cannot live in memory:
    # an account that vanishes on restart is not an account.
    db.init()
    logger.info("[STARTUP] Server ready. Swagger UI → http://localhost:8000/docs")
    yield
    logger.info("[SHUTDOWN] MuleShield AI shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MuleShield AI — Backend API",
    description=(
        "Real-time cybercrime interdiction API for MHA / I4C. "
        "Forecasts cash-withdrawal locations from 1930 complaints using a "
        "GraphSAGE GNN, a conditional-logit ATM choice model and XGBoost. "
        "SIH26184 | Smart India Hackathon 2026."
    ),
    version="3.0.0",
    contact={
        "name": "Team MuleShield AI",
        "url": "https://github.com/hotshot0104/SIH2026",
    },
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
#
# `allow_origins=["*"]` with `allow_credentials=True` is not a permissive
# configuration -- it is a broken one. Browsers reject the combination outright,
# so it never did what it looked like it did, and it invited any origin to make
# credentialed calls against an API that can freeze a bank account.
#
# The default list is the four addresses `npm run dev` and `npm run preview`
# actually serve from. A production deploy needs NO entry: the container serves
# the console from the same process that answers the API, and the built bundle
# calls relative URLs (see frontend/src/services/api.js), so nothing is
# cross-origin. A split frontend/backend deployment sets the variable.
_DEFAULT_ORIGINS = (
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:4173,http://127.0.0.1:4173"
)
_ORIGINS = [o.strip() for o in
            os.environ.get("MULESHIELD_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
            if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
logger.info("[STARTUP] CORS origins: %s", ", ".join(_ORIGINS))

# ── Routers ───────────────────────────────────────────────────────────────────
# Every one of these must be registered ABOVE the SPA catch-all further down,
# or the console's index.html answers /api/v1/... with a 200 and an HTML body.
app.include_router(auth.router)
app.include_router(complaint.router)
app.include_router(graph.router)
app.include_router(embeddings.router)
app.include_router(predict.router)
app.include_router(freeze.router)
app.include_router(audit.router)
app.include_router(intel.router)
app.include_router(hotspot.router)


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — /ws/feed
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    """
    Real-time event stream for the dashboard.

    Events broadcasted:
      NEW_COMPLAINT    — when a new complaint is ingested
      PREDICTION_READY — when XGBoost prediction completes
      FREEZE_EXECUTED  — when micro-freeze is confirmed
    """
    await manager.connect(websocket)
    try:
        # Send current connection count as welcome message
        await manager.send_personal(
            {
                "event_type": "CONNECTED",
                "complaint_id": "system",
                "payload": {
                    "message": "MuleShield AI live feed connected.",
                    "active_complaints": len(state.complaints),
                    "active_connections": manager.connection_count,
                },
            },
            websocket,
        )
        # Keep connection alive — real events are pushed from routers
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # network resets, abrupt client death, etc.
        logger.warning(f"[WS] Connection closed unexpectedly: {e}")
    finally:
        # Always deregister — a leaked socket makes every later broadcast slower.
        manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint. Used by load balancers and CI."""
    return {
        "status": "ok",
        "service": "MuleShield AI Backend",
        "version": "3.0.0",
        "active_complaints": len(state.complaints),
        "atm_directory_size": len(state.atm_directory),
        "embeddings_loaded": len(state.embeddings),
        "ws_connections": manager.connection_count,
    }


@app.get("/api", tags=["System"])
async def api_root():
    """API index. The bare "/" serves the console when a build is present."""
    return {
        "message": "MuleShield AI Backend is running.",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/feed",
    }


# ── Console ──────────────────────────────────────────────────────────────────
#
# One process serves the API and the built frontend, so a deployment is one
# container on one port. Registered LAST so it cannot shadow /api/v1, /ws,
# /health or /docs -- FastAPI matches routes in declaration order.
#
# Absent in development: `npm run dev` serves the console on its own port and
# this block is skipped, which is why it is guarded rather than assumed.
_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if (_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_console(full_path: str):
        """
        Hand every unmatched path to the SPA.

        React Router owns /map, /graph, /intercept and /model. Without this a
        refresh on any of them 404s from the server, because those paths exist
        only in the browser.
        """
        candidate = _DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")

    logger.info(f"[STARTUP] Serving console from {_DIST}")
else:
    logger.info("[STARTUP] No frontend build found - API only.")
