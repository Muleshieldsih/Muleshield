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


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Credentials posted as JSON.

    JSON rather than an OAuth2 form because the form flow needs python-multipart,
    which is not among this project's dependencies. Adding it would mean touching
    the Dockerfile's pip layer for no functional gain.
    """
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)

    model_config = {
        "json_schema_extra": {
            "example": {"username": "officer", "password": "your-password"}
        }
    }


class UserOut(BaseModel):
    """An officer, as shown to the client. Carries no hash and no session."""
    id: int
    username: str
    display_name: str
    is_admin: bool = False
    locked: bool = False
    created_at: str = ""
    last_login: Optional[str] = None


class LoginResponse(BaseModel):
    """Bearer token plus the officer it belongs to.

    The field names mirror the OAuth2 response shape so the payload reads as
    conventional, even though the request was JSON. `expires_at` lets the console
    pre-empt an expiry instead of discovering it through a failed request.
    """
    access_token: str
    token_type: str = "bearer"
    expires_at: str
    user: UserOut


class ForgotPasswordRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)


class CreateUserRequest(BaseModel):
    """Credentials an administrator issues to a new officer.

    There is no self-registration. An officer has an account because somebody
    with authority created one.
    """
    username: str = Field(..., min_length=1, max_length=64)
    display_name: str = Field(..., min_length=1, max_length=120)
    password: str = Field(..., min_length=8, max_length=256)
    is_admin: bool = False


class ResetRequestOut(BaseModel):
    """One queued reset, as an administrator sees it."""
    id: int
    username: str
    display_name: str
    status: str
    requested_at: str
    decided_by: Optional[str] = None
    decided_at: Optional[str] = None
    expires_at: Optional[str] = None


class ResetApprovalResponse(BaseModel):
    """Returned ONCE, to the approving administrator.

    The token is not stored in the clear and cannot be retrieved again. It is
    handed to the officer by whatever channel the administrator already trusts.
    """
    detail: str
    reset_token: str
    expires_in_minutes: int


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=256,
                              description="Minimum eight characters.")


# ---------------------------------------------------------------------------
# Cross-case intelligence
# ---------------------------------------------------------------------------

class ATMIntelRow(BaseModel):
    """One ATM, summarised across every case in the corpus.

    The per-case screens answer "where will this withdrawal happen". This answers
    the question I4C actually cares about across a district: which machines keep
    coming back. The model already consumes that history as a feature
    (`atm_prior_count`); until now nothing surfaced it to a person.
    """
    atm_id: str
    cashouts: int
    distinct_complaints: int
    total_amount: float
    bank_name: str = ""
    city: str = ""
    district: str = ""
    state: str = ""
    lat: float = 0.0
    lon: float = 0.0
    cashout_risk_score: float = 0.0
    last_seen: str = ""


# ---------------------------------------------------------------------------
# Forward hotspot surface
# ---------------------------------------------------------------------------

class HotspotWindow(BaseModel):
    """One forecast window for one cell.

    The three money fields are the point of this model and are kept apart
    deliberately. conditional_rupees is forecast mass contributed by complaints
    open right now; prior_rupees is the capped historical term. A reader -- an
    officer, or a judge -- can see at a glance which one is carrying the cell,
    and that is the difference between a forecast and a density map.
    """
    window_start_min: int
    window_end_min: int
    score: float = 0.0
    conditional_rupees: float = 0.0
    prior_rupees: float = 0.0
    prior_share: float = 0.0
    case_count: int = 0


class HotspotCell(BaseModel):
    """One cash-out cell: a cluster of machines a single team could cover.

    Not a district. 1,000 ATMs across 78 districts averages 12.8 machines per
    district, and a district is an administrative boundary rather than a
    deployable one. district and state ride along as roll-up keys so the
    dashboard drill-down is a group-by rather than a second model.
    """
    cell_id: str
    lat: float = 0.0
    lon: float = 0.0
    district: str = ""
    state: str = ""
    city: str = ""
    atm_count: int = 0
    score: float = 0.0
    conditional_rupees: float = 0.0
    prior_rupees: float = 0.0
    prior_share: float = 0.0
    case_count: int = 0
    # Which complaints, not just how many. The console lists them with links back
    # into triage, so an officer reading a hot cell can open the cases driving it
    # rather than taking the number on trust.
    complaint_ids: list[str] = []
    windows: list[HotspotWindow] = []


class HotspotSurfaceResponse(BaseModel):
    """The national forward surface.

    degraded is load-bearing rather than cosmetic. With no complaints open the
    surface is entirely historical -- and a historical density map presented as
    a forecast is precisely what this component exists not to be. When it is
    true the console says "no live cases; this is history only" instead of
    drawing circles that imply prediction.
    """
    as_of: str
    degraded: bool = False
    prior_weight: float = 0.15
    cell_radius_km: float = 12.0
    n_cells: int = 0
    n_open_complaints: int = 0
    windows_min: list[list[int]] = []
    total_conditional_rupees: float = 0.0
    total_prior_rupees: float = 0.0
    prior_share_national: float = 0.0
    filters: dict = {}
    cells: list[HotspotCell] = []


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

class AlertOut(BaseModel):
    """One raised alert.

    prior_share travels with the alert rather than being recomputed later. An
    officer reading this tomorrow has to be able to see how much of it was live
    forecast and how much was historical pattern -- otherwise a density map and
    a forecast look identical once they are both just rows in an inbox.
    """
    id: str
    created_at: str = ""
    rule_id: str = ""
    severity: str = ""                 # "CRITICAL" | "HIGH" | "WATCH"
    cell_id: str = ""
    district: str = ""
    state: str = ""
    window_start_min: int = 0
    window_end_min: int = 0
    score: float = 0.0
    rupees_at_risk: float = 0.0
    case_count: int = 0
    complaint_ids: list[str] = []
    prior_share: float = 0.0
    headline: str = ""
    status: str = "open"               # "open" | "acknowledged" | "dismissed"
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    disposition: Optional[str] = None
    dedupe_bucket: str = ""


class DeliveryOut(BaseModel):
    """One attempt to reach one recipient.

    This is the evidence that a force WAS warned, which is a fact an inquiry
    would want established. attempts and last_error are exposed rather than
    hidden: a delivery that silently never arrived is the worst outcome this
    system can produce, and it should be visible on the screen.
    """
    id: int
    alert_id: str
    channel: str = ""
    recipient: str = ""
    state: str = ""                    # "queued" | "sent" | "failed" | "dead"
    attempts: int = 0
    last_error: Optional[str] = None
    queued_at: str = ""
    sent_at: Optional[str] = None
    next_retry_at: Optional[str] = None
    provider_ref: Optional[str] = None


class AlertDetail(AlertOut):
    deliveries: list[DeliveryOut] = []


class AlertAckRequest(BaseModel):
    """Acknowledgement payload.

    disposition is required, with no default. Making it optional would let the
    common path skip it, and the whole reason it exists is to capture the
    outcome -- including "False positive", which is the one an operator is least
    motivated to record and the one this system most needs.
    """
    disposition: str = Field(..., min_length=1, max_length=40,
                             description="Dispatched | Monitoring | False positive | Duplicate")


class RecipientOut(BaseModel):
    id: int
    name: str = ""
    role: str = ""                     # "LEA" | "I4C" | "BANK"
    channel: str = ""
    address: str = ""
    scope_state: str = ""
    scope_district: str = ""
    active: int = 1


class AlertSummary(BaseModel):
    deliveries: dict = {}
    dispositions: dict = {}
    actioned: int = 0
    false_positive_rate: float = 0.0


class RuleRunSummary(BaseModel):
    """The result of one rule pass, including when nothing fired.

    degraded is carried out so a caller can distinguish "the rules ran and
    nothing qualified" from "there were no live cases to run against". Those
    look the same in a raised-count of zero and mean completely different things.
    """
    as_of: str = ""
    degraded: bool = False
    cells_considered: int = 0
    open_complaints: int = 0
    raised: int = 0
    dry_run: bool = False
    alerts: list[AlertOut] = []
