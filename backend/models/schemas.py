# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: Pydantic Schemas
SIH26184 | MHA / I4C

All request/response models for the FastAPI backend.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# COMPLAINT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ComplaintIngestRequest(BaseModel):
    """Payload for POST /api/v1/complaint/ingest"""
    victim_name: str
    victim_bank: str
    victim_account: str
    fraud_type: str
    stolen_amount: float = Field(..., gt=0)
    city: str
    state: str
    complaint_timestamp: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "victim_name": "Rajesh Kumar",
                "victim_bank": "SBI",
                "victim_account": "ACC-12345678",
                "fraud_type": "UPI Fraud",
                "stolen_amount": 75000.0,
                "city": "Mumbai",
                "state": "Maharashtra",
            }
        }
    }


class ComplaintResponse(BaseModel):
    """Full complaint record returned by ingestion and listing endpoints."""
    ticket_id: str
    victim_name: str
    victim_bank: str
    victim_account: str
    fraud_type: str
    stolen_amount: float
    city: str
    state: str
    complaint_timestamp: str
    status: str = "ACTIVE"
    is_live: bool = False   # True for complaints ingested during this session


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    """A single node in React Flow format."""
    id: str
    label: str
    node_type: str          # "victim" | "mule" | "terminal"
    hop_depth: int
    bank: str
    amount: float
    lat: float
    lon: float
    risk_score: float = 0.0


class GraphEdge(BaseModel):
    """A directed transaction edge."""
    id: str
    source: str
    target: str
    amount: float
    timestamp: str
    label: str              # e.g. "₹25,000"


class GraphAnomalies(BaseModel):
    """Anomalies detected on a complaint's sub-graph by the NetworkX engine."""
    velocity_flagged: list[str] = []      # accounts breaching the velocity rule
    fund_split_flagged: list[str] = []    # accounts dispersing 1-to-N evenly
    terminal_leaves: list[str] = []       # leaf nodes = cashout candidates
    velocity_count: int = 0
    fund_split_count: int = 0
    terminal_count: int = 0
    velocity_rule: str = ""               # human-readable threshold, shown in UI
    fund_split_rule: str = ""


class GraphResponse(BaseModel):
    """Node-link graph data in React Flow format for GET /api/v1/graph/{id}"""
    complaint_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    node_count: int
    edge_count: int
    build_time_ms: float
    anomalies: GraphAnomalies = GraphAnomalies()


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class MuleNodeRisk(BaseModel):
    """Risk metadata for a single mule account node."""
    account_id: str
    bank: str
    hop_depth: int
    risk_score: float
    embedding_norm: float           # L2 norm of the 64-dim GNN vector
    embedding_preview: list[float]  # First 8 dims for preview


class EmbeddingResponse(BaseModel):
    """Top-N highest risk mule nodes for GET /api/v1/embeddings/{id}"""
    complaint_id: str
    top_mules: list[MuleNodeRisk]
    total_nodes: int


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ATMPrediction(BaseModel):
    """One ranked candidate cash-out location."""
    rank: int
    atm_id: str
    confidence: float
    lat: float
    lon: float
    bank: str
    address: str
    historical_fraud_count: int


class SearchZone(BaseModel):
    """
    The area to deploy to — the problem statement's primary output.

    SIH26184 asks for likely cash withdrawal *locations*. A unit is dispatched to
    an area, not to a single machine, so the zone is the deliverable and the
    ranked candidate list is the tactical drill-down inside it.
    """
    lat: float
    lon: float
    radius_km: float
    atm_count: int              # machines a team would have to cover
    candidates_covered: int
    probability_mass: float     # share of the model's distribution inside the zone


class PredictionResponse(BaseModel):
    """Full cashout prediction for GET /api/v1/predict/cashout/{id}"""
    complaint_id: str
    search_zone: Optional[SearchZone] = None
    ranked_candidates: list[ATMPrediction]
    time_to_cashout_minutes: float
    # 5th/95th-percentile band. The delay carries irreducible noise, so a point
    # estimate alone overstates what is knowable.
    time_to_cashout_low: Optional[float] = None
    time_to_cashout_high: Optional[float] = None
    interception_confidence: float
    inference_time_ms: float
    stolen_amount: float
    terminal_account: str
    terminal_lat: float
    terminal_lon: float


# ─────────────────────────────────────────────────────────────────────────────
# FREEZE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class FreezeRequest(BaseModel):
    """Payload for POST /api/v1/bank/micro-freeze"""
    account_id: str
    complaint_id: str
    bank: str = "UNKNOWN"       # resolved server-side from the account record
    officer_id: str = "OFFICER-001"

    model_config = {
        "json_schema_extra": {
            "example": {
                "account_id": "ACC-7A9B1C2D",
                "complaint_id": "TKT-0001",
                "bank": "HDFC",
                "officer_id": "OFFICER-001",
            }
        }
    }


class FreezeResponse(BaseModel):
    """Confirmation returned after a micro-freeze operation."""
    status: str                 # "FROZEN"
    account: str
    bank: str
    complaint_id: str
    timestamp: str
    officer_id: str
    freeze_reference: str       # Unique freeze ticket ID


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class WSEvent(BaseModel):
    """Real-time event pushed over WebSocket /ws/feed"""
    event_type: str             # "NEW_COMPLAINT" | "PREDICTION_READY" | "FREEZE_EXECUTED"
    complaint_id: str
    payload: dict
    timestamp: str
