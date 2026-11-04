"""FastAPI WebSocket Connection Manager for RailPulse AI Telemetry Streaming.

Manages active client sessions, broadcasts live ST-GNN inference telemetry,
and handles graceful disconnects and heartbeats.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket client connections for real-time telemetry broadcast."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept an incoming WebSocket connection and register client."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(
            "WebSocket client connected. Total active clients: %d",
            len(self.active_connections),
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket client."""
        async with self._lock:
            # Match by object identity or equality
            self.active_connections = [c for c in self.active_connections if c is not websocket and c != websocket]
        logger.info(
            "WebSocket client disconnected. Total active clients: %d",
            len(self.active_connections),
        )

    async def send_personal_json(self, data: Dict[str, Any], websocket: WebSocket) -> None:
        """Send a JSON payload directly to a specific connected client."""
        await websocket.send_json(data)

    async def broadcast_json(self, data: Dict[str, Any]) -> None:
        """Broadcast a JSON payload concurrently to all registered WebSocket clients.
        Automatically prunes dead or stalled connections.
        """
        async with self._lock:
            stale: List[WebSocket] = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(data)
                except Exception as exc:
                    logger.warning("Failed to send telemetry to client (%s); queuing removal", exc)
                    stale.append(connection)

            for dead in stale:
                if dead in self.active_connections:
                    self.active_connections.remove(dead)

    @property
    def client_count(self) -> int:
        """Return the number of currently active WebSocket clients."""
        return len(self.active_connections)

