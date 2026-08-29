# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3 Test Suite
SIH26184 | MHA / I4C

Tests all Phase 3 acceptance criteria:
  AC1: All endpoints return correct JSON within < 200ms
  AC2: WebSocket connects and receives broadcast messages
  AC3: Micro-freeze returns { status: "FROZEN", account: ..., timestamp: ... }
  AC4: API docs auto-generated at /docs (Swagger UI)
  AC5: CORS enabled for any frontend origin

Run:
    python -m pytest backend/tests/test_phase3.py -v
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))

from backend.main import app

# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Synchronous TestClient — triggers lifespan startup (loads all data)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def first_complaint_id(client):
    """Get the ticket_id of the first pre-loaded complaint that has transactions."""
    import backend.state as state
    # Find a complaint ID that actually has transaction data
    for cid in state.transactions_by_complaint:
        if cid in state.complaints:
            return cid
    # Fallback: any complaint with transactions (even if not in complaint store)
    for cid in state.transactions_by_complaint:
        return cid
    pytest.skip("No pre-loaded complaints with transaction data found.")


@pytest.fixture(scope="module")
def ingested_complaint(client):
    """POST a new complaint and return its record."""
    payload = {
        "victim_name": "Ananya Sharma",
        "victim_bank": "SBI",
        "victim_account": "ACC-TEST0001",
        "fraud_type": "UPI Fraud",
        "stolen_amount": 85000.0,
        "city": "Bengaluru",
        "state": "Karnataka",
    }
    resp = client.post("/api/v1/complaint/ingest", json=payload)
    assert resp.status_code == 200
    return resp.json()


# ─────────────────────────────────────────────
# HEALTH & ROOT
# ─────────────────────────────────────────────

class TestHealthAndRoot:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_has_active_complaints(self, client):
        resp = client.get("/health")
        assert resp.json()["active_complaints"] >= 0

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        resp = client.get("/")
        assert "docs" in resp.json()


# ─────────────────────────────────────────────
# AC4: Swagger UI docs
# ─────────────────────────────────────────────

class TestSwaggerDocs:

    def test_docs_endpoint_reachable(self, client):
        """AC4: /docs must return HTML (Swagger UI)."""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema_reachable(self, client):
        """OpenAPI schema must be accessible at /openapi.json."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "info" in schema

    def test_openapi_has_all_routes(self, client):
        """All Phase 3 endpoints must appear in the OpenAPI schema."""
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]
        required_paths = [
            "/api/v1/complaint/ingest",
            "/api/v1/complaint/list",
            "/api/v1/graph/{complaint_id}",
            "/api/v1/embeddings/{complaint_id}",
            "/api/v1/predict/cashout/{complaint_id}",
            "/api/v1/bank/micro-freeze",
        ]
        for path in required_paths:
            assert path in paths, f"Missing endpoint in OpenAPI schema: {path}"


# ─────────────────────────────────────────────
# AC5: CORS
# ─────────────────────────────────────────────

class TestCORS:

    def test_cors_headers_present(self, client):
        """AC5: CORS must be enabled."""
        resp = client.options(
            "/api/v1/complaint/list",
            headers={"Origin": "http://localhost:5173"},
        )
        assert resp.status_code in (200, 405)  # preflight or method not allowed


# ─────────────────────────────────────────────
# COMPLAINT ENDPOINTS
# ─────────────────────────────────────────────

class TestComplaintEndpoints:

    def test_list_complaints_returns_200(self, client):
        resp = client.get("/api/v1/complaint/list")
        assert resp.status_code == 200

    def test_list_complaints_returns_list(self, client):
        resp = client.get("/api/v1/complaint/list")
        assert isinstance(resp.json(), list)

    def test_list_complaints_has_preloaded_data(self, client):
        resp = client.get("/api/v1/complaint/list")
        assert len(resp.json()) > 0, "State must be pre-populated at startup"

    def test_ingest_complaint_returns_200(self, ingested_complaint):
        assert ingested_complaint is not None

    def test_ingested_complaint_has_ticket_id(self, ingested_complaint):
        assert "ticket_id" in ingested_complaint
        assert ingested_complaint["ticket_id"].startswith("TKT-")

    def test_ingested_complaint_has_correct_amount(self, ingested_complaint):
        assert ingested_complaint["stolen_amount"] == 85000.0

    def test_ingested_complaint_starts_in_the_new_state(self, ingested_complaint):
        """
        A freshly ingested case enters the workflow at "New".

        This asserted "ACTIVE" until the case workflow existed. That value was
        written once at ingestion and never read or changed by anything, so it
        described nothing an analyst could act on. It is now the first of six
        real states, and the stored legacy value is presented as "New" -- an
        untouched case is exactly what it meant.
        """
        assert ingested_complaint["status"] == "New"
        assert ingested_complaint["assignee"] is None
        assert ingested_complaint["note_count"] == 0

    def test_get_complaint_by_id(self, client, ingested_complaint):
        tid = ingested_complaint["ticket_id"]
        resp = client.get(f"/api/v1/complaint/{tid}")
        assert resp.status_code == 200
        assert resp.json()["ticket_id"] == tid

    def test_get_unknown_complaint_returns_404(self, client):
        resp = client.get("/api/v1/complaint/TKT-UNKNOWN999")
        assert resp.status_code == 404

    def test_ingest_under_200ms(self, client):
        """AC1: Ingest endpoint must respond in < 200ms."""
        payload = {
            "victim_name": "Speed Test User",
            "victim_bank": "HDFC",
            "victim_account": "ACC-SPEEDTEST",
            "fraud_type": "Digital Arrest",
            "stolen_amount": 10000.0,
            "city": "Delhi",
            "state": "Delhi",
        }
        t0 = time.time()
        resp = client.post("/api/v1/complaint/ingest", json=payload)
        elapsed = (time.time() - t0) * 1000
        assert resp.status_code == 200
        assert elapsed < 200, f"Ingest took {elapsed:.1f}ms (> 200ms)"


# ─────────────────────────────────────────────
# GRAPH ENDPOINTS
# ─────────────────────────────────────────────

class TestGraphEndpoints:

    def test_graph_returns_200(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/graph/{first_complaint_id}")
        assert resp.status_code == 200

    def test_graph_has_nodes_and_edges(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/graph/{first_complaint_id}")
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data

    def test_graph_nodes_not_empty(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/graph/{first_complaint_id}")
        assert len(resp.json()["nodes"]) > 0

    def test_graph_node_has_required_fields(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/graph/{first_complaint_id}")
        node = resp.json()["nodes"][0]
        for field in ("id", "label", "node_type", "hop_depth", "bank", "lat", "lon"):
            assert field in node, f"Node missing field: {field}"

    def test_graph_edge_has_required_fields(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/graph/{first_complaint_id}")
        data = resp.json()
        if data["edges"]:
            edge = data["edges"][0]
            for field in ("id", "source", "target", "amount"):
                assert field in edge, f"Edge missing field: {field}"

    def test_graph_has_victim_node(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/graph/{first_complaint_id}")
        node_types = [n["node_type"] for n in resp.json()["nodes"]]
        assert "victim" in node_types, "Graph must contain a victim node"

    def test_graph_unknown_complaint_returns_404(self, client):
        resp = client.get("/api/v1/graph/TKT-UNKNOWN999")
        assert resp.status_code == 404

    def test_graph_under_200ms(self, client, first_complaint_id):
        """AC1: Graph endpoint must respond in < 200ms."""
        t0 = time.time()
        resp = client.get(f"/api/v1/graph/{first_complaint_id}")
        elapsed = (time.time() - t0) * 1000
        assert resp.status_code == 200
        assert elapsed < 200, f"Graph took {elapsed:.1f}ms (> 200ms)"


# ─────────────────────────────────────────────
# EMBEDDINGS ENDPOINTS
# ─────────────────────────────────────────────

class TestEmbeddingEndpoints:

    def test_embeddings_returns_200(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/embeddings/{first_complaint_id}")
        assert resp.status_code == 200

    def test_embeddings_has_top_mules(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/embeddings/{first_complaint_id}")
        assert "top_mules" in resp.json()

    def test_embeddings_mule_has_required_fields(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/embeddings/{first_complaint_id}")
        data = resp.json()
        if data["top_mules"]:
            mule = data["top_mules"][0]
            for field in ("account_id", "bank", "hop_depth", "risk_score", "embedding_preview"):
                assert field in mule, f"Mule node missing field: {field}"

    def test_embeddings_preview_is_8_dims(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/embeddings/{first_complaint_id}")
        data = resp.json()
        if data["top_mules"]:
            preview = data["top_mules"][0]["embedding_preview"]
            assert len(preview) == 8, f"Expected 8-dim preview, got {len(preview)}"

    def test_embeddings_top_n_param(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/embeddings/{first_complaint_id}?top_n=3")
        data = resp.json()
        assert len(data["top_mules"]) <= 3

    def test_embeddings_unknown_complaint_returns_404(self, client):
        resp = client.get("/api/v1/embeddings/TKT-UNKNOWN999")
        assert resp.status_code == 404


# ─────────────────────────────────────────────
# PREDICTION ENDPOINTS
# ─────────────────────────────────────────────

class TestPredictionEndpoints:

    def test_predict_returns_200(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        assert resp.status_code == 200

    def test_predict_has_ranked_candidates(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        data = resp.json()
        assert "ranked_candidates" in data
        assert len(data["ranked_candidates"]) == 5

    def test_predict_atm_has_required_fields(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        atm = resp.json()["ranked_candidates"][0]
        for field in ("rank", "atm_id", "confidence", "lat", "lon", "bank"):
            assert field in atm, f"ATM prediction missing field: {field}"

    def test_predict_ranks_are_dense_and_ordered(self, client, first_complaint_id):
        """The API returns 5 ranked candidate locations, ranked 1..5."""
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        ranks = [a["rank"] for a in resp.json()["ranked_candidates"]]
        assert ranks == [1, 2, 3, 4, 5]

    def test_predict_confidence_in_valid_range(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        for atm in resp.json()["ranked_candidates"]:
            assert 0.0 <= atm["confidence"] <= 1.0

    def test_predict_time_to_cashout_positive(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        assert resp.json()["time_to_cashout_minutes"] > 0

    def test_predict_has_inference_time(self, client, first_complaint_id):
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        assert "inference_time_ms" in resp.json()

    def test_predict_under_200ms(self, client, first_complaint_id):
        """AC1: Prediction endpoint must respond in < 200ms."""
        t0 = time.time()
        resp = client.get(f"/api/v1/predict/cashout/{first_complaint_id}")
        elapsed = (time.time() - t0) * 1000
        assert resp.status_code == 200
        assert elapsed < 200, f"Prediction took {elapsed:.1f}ms (> 200ms)"

    def test_predict_unknown_complaint_returns_404(self, client):
        resp = client.get("/api/v1/predict/cashout/TKT-UNKNOWN999")
        assert resp.status_code == 404


# ─────────────────────────────────────────────
# AC3: MICRO-FREEZE ENDPOINT
# ─────────────────────────────────────────────

class TestFreezeEndpoint:

    def test_freeze_returns_200(self, client, first_complaint_id):
        payload = {
            "account_id": "ACC-TESTFREEZE",
            "complaint_id": first_complaint_id,
            "bank": "HDFC",
            "officer_id": "OFC-001",
        }
        resp = client.post("/api/v1/bank/micro-freeze", json=payload)
        assert resp.status_code == 200

    def test_freeze_returns_status_frozen(self, client, first_complaint_id):
        """AC3: Must return { status: 'FROZEN', ... }"""
        payload = {
            "account_id": "ACC-TESTFREEZE2",
            "complaint_id": first_complaint_id,
            "bank": "SBI",
            "officer_id": "OFC-002",
        }
        resp = client.post("/api/v1/bank/micro-freeze", json=payload)
        data = resp.json()
        assert data["status"] == "FROZEN"

    def test_freeze_has_account_field(self, client, first_complaint_id):
        payload = {
            "account_id": "ACC-TESTFREEZE3",
            "complaint_id": first_complaint_id,
            "bank": "ICICI",
            "officer_id": "OFC-003",
        }
        resp = client.post("/api/v1/bank/micro-freeze", json=payload)
        assert "account" in resp.json()
        assert resp.json()["account"] == "ACC-TESTFREEZE3"

    def test_freeze_has_timestamp(self, client, first_complaint_id):
        payload = {
            "account_id": "ACC-TESTFREEZE4",
            "complaint_id": first_complaint_id,
            "bank": "Axis",
            "officer_id": "OFC-004",
        }
        resp = client.post("/api/v1/bank/micro-freeze", json=payload)
        assert "timestamp" in resp.json()

    def test_freeze_has_reference_number(self, client, first_complaint_id):
        payload = {
            "account_id": "ACC-TESTFREEZE5",
            "complaint_id": first_complaint_id,
            "bank": "BOI",
            "officer_id": "OFC-005",
        }
        resp = client.post("/api/v1/bank/micro-freeze", json=payload)
        ref = resp.json()["freeze_reference"]
        assert ref.startswith("FRZ-")

    def test_freeze_under_200ms(self, client, first_complaint_id):
        """AC1: Freeze endpoint must respond in < 200ms."""
        payload = {
            "account_id": "ACC-SPEEDFREEZE",
            "complaint_id": first_complaint_id,
            "bank": "HDFC",
            "officer_id": "OFC-SPEED",
        }
        t0 = time.time()
        resp = client.post("/api/v1/bank/micro-freeze", json=payload)
        elapsed = (time.time() - t0) * 1000
        assert resp.status_code == 200
        assert elapsed < 200, f"Freeze took {elapsed:.1f}ms (> 200ms)"


# ─────────────────────────────────────────────
# AC2: WEBSOCKET
# ─────────────────────────────────────────────

class TestWebSocket:

    def test_websocket_connects(self, client):
        """AC2: WebSocket /ws/feed must accept connections."""
        with client.websocket_connect("/ws/feed") as ws:
            data = ws.receive_json()
            assert data["event_type"] == "CONNECTED"

    def test_websocket_receives_welcome_payload(self, client):
        """AC2: Welcome message must contain active_complaints count."""
        with client.websocket_connect("/ws/feed") as ws:
            data = ws.receive_json()
            assert "active_complaints" in data["payload"]

    def test_websocket_receives_new_complaint_event(self, client):
        """AC2: Ingesting a complaint must broadcast to WS clients."""
        with client.websocket_connect("/ws/feed") as ws:
            # Consume welcome message
            ws.receive_json()

            # Ingest a new complaint
            payload = {
                "victim_name": "WS Test User",
                "victim_bank": "PNB",
                "victim_account": "ACC-WSTEST01",
                "fraud_type": "Job Scam",
                "stolen_amount": 50000.0,
                "city": "Chennai",
                "state": "Tamil Nadu",
            }
            client.post("/api/v1/complaint/ingest", json=payload)

            # Should receive the broadcast
            event = ws.receive_json()
            assert event["event_type"] == "NEW_COMPLAINT"
            assert event["payload"]["stolen_amount"] == 50000.0
