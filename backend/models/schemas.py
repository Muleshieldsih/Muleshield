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
    # Workflow fields. Absent on seed records, so both carry a default rather
    # than 404-ing a case that has simply never been touched.
    assignee: Optional[str] = None
    updated_at: Optional[str] = None
    note_count: int = 0


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
    """
    One ranked candidate cash-out location.

    The directory carries city, district, opening hours and a risk score for
    every ATM, all loaded into memory at startup and none of it previously sent
    to the client. Opening hours in particular decide whether a location is
    worth dispatching to at all -- a machine inside a branch that shut at 21:00
    is not where a 02:00 withdrawal happens.
    """
    rank: int
    atm_id: str
    confidence: float
    lat: float
    lon: float
    bank: str
    address: str
    historical_fraud_count: int
    city: str = ""
    district: str = ""
    state: str = ""
    opening_time: str = ""
    closing_time: str = ""
    cashout_risk_score: float = 0.0


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


# ─────────────────────────────────────────────────────────────────────────────
# CASE WORKFLOW SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class CaseUpdateRequest(BaseModel):
    """PATCH /api/v1/complaint/{id} — move a case through the workflow."""
    status: Optional[str] = Field(default=None, description="One of the six workflow statuses")
    assignee: Optional[str] = Field(default=None, description="Analyst the case is assigned to")
    actor: str = Field(default="ANALYST", description="Who is making the change, for the audit trail")


class NoteRequest(BaseModel):
    """POST /api/v1/complaint/{id}/note"""
    text: str = Field(..., min_length=1, max_length=2000)
    author: str = Field(default="ANALYST")


class NoteResponse(BaseModel):
    id: str
    timestamp: str
    author: str
    text: str


class AuditEntry(BaseModel):
    """One line of the audit trail."""
    id: str
    timestamp: str
    actor: str
    action: str
    object: str
    result: str
    case_id: str


class TransactionRow(BaseModel):
    """
    One transfer in a case's money trail.

    The graph endpoint already returns edges, but it de-duplicates them by
    source-destination pair, so two transfers between the same accounts collapse
    into one and the second disappears. A ledger an investigator works from
    cannot drop rows, so this returns every transaction as recorded.
    """
    txn_id: str
    timestamp: str
    src_account: str
    dst_account: str
    amount: float
    bank_name: str = "Unknown"
    ifsc_code: str = ""
    city: str = ""
    state: str = ""
    channel: str = ""
    hop_depth: int = 0
    is_terminal: bool = False
    cashout_atm_id: Optional[str] = None
    minutes_from_first: float = 0.0
