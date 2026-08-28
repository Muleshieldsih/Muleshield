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
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import backend.state as state
from backend.routers import complaint, graph, embeddings, predict, freeze
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
        "Powered by GraphSAGE GNN + XGBoost with Bayesian Spatial Reranking. "
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allows React frontend on any port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(complaint.router)
app.include_router(graph.router)
app.include_router(embeddings.router)
app.include_router(predict.router)
app.include_router(freeze.router)


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


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "MuleShield AI Backend is running.",
        "docs": "/docs",
        "health": "/health",
        "websocket": "ws://localhost:8000/ws/feed",
    }
