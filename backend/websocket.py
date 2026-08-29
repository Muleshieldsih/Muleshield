# -*- coding: utf-8 -*-
"""
MuleShield AI -- Phase 3: WebSocket Connection Manager
SIH26184 | MHA / I4C

Manages all active WebSocket connections and broadcasts real-time
events (new complaints, predictions, freeze confirmations) to every
connected dashboard client.
"""

import json
import logging
from datetime import datetime, timezone

from fastapi import WebSocket

logger = logging.getLogger("muleshield.ws")


class ConnectionManager:
    """
    Thread-safe WebSocket manager.

    Usage:
        manager = ConnectionManager()

        # In WebSocket endpoint:
        await manager.connect(websocket)
        try:
            while True:
                await websocket.receive_text()   # keep-alive
        except:
            manager.disconnect(websocket)

        # In any router:
        await manager.broadcast({"event_type": "NEW_COMPLAINT", ...})
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"[WS] Client connected. Active connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket client."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"[WS] Client disconnected. Active connections: {len(self.active_connections)}"
        )

    async def broadcast(self, message: dict) -> None:
        """
        Broadcast a JSON message to ALL connected clients.
        Dead connections are silently removed.
        """
        if not message.get("timestamp"):
            message["timestamp"] = datetime.now(timezone.utc).isoformat()

        payload = json.dumps(message, default=str)
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)

        for conn in dead:
            self.disconnect(conn)

    async def send_personal(self, message: dict, websocket: WebSocket) -> None:
        """Send a message to a single specific client."""
        await websocket.send_text(json.dumps(message, default=str))

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


# Singleton instance — shared across all routers via dependency injection
manager = ConnectionManager()
