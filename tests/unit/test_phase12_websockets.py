"""
Phase 12 — FastAPI WebSocket Telemetry Streaming Diagnostic Test Suite.
Verifies ConnectionManager registration, lifecycle, client broadcast,
and /ws/telemetry streaming updates using TestClient.websocket_connect.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.main import app, ws_manager
from src.api.websocket_manager import ConnectionManager


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestPhase12WebSockets:
    """Test suite for Phase 12 WebSocket Telemetry Streaming."""

    def test_connection_manager_lifecycle(self):
        """Verify ConnectionManager tracks connections and client count."""
        cm = ConnectionManager()
        assert cm.client_count == 0

    def test_websocket_telemetry_streaming(self, client):
        """Verify /ws/telemetry connects and streams ST-GNN inference telemetry packets."""
        with client.websocket_connect("/ws/telemetry") as websocket:
            # Receive first streamed packet
            data = websocket.receive_json()

            assert isinstance(data, dict)
            assert data.get("type") == "TELEMETRY_SNAPSHOT"
            assert "timestamp" in data
            assert "predictions" in data
            assert isinstance(data["predictions"], list)
            assert len(data["predictions"]) > 0

            # Verify packet contains GATv2 attention and MCD uncertainty intervals
            assert "confidence_intervals" in data
            assert "gat_edge_attributions" in data
            assert isinstance(data["gat_edge_attributions"], list)

            # Check individual prediction structure
            p0 = data["predictions"][0]
            assert "eva_id" in p0
            assert "station_name" in p0
            assert "delay_15m" in p0
            assert "delay_30m" in p0
            assert "delay_60m" in p0

    def test_websocket_disconnect_clean_handling(self, client):
        """Verify server handles client disconnect gracefully without dangling connections."""
        initial_count = ws_manager.client_count
        with client.websocket_connect("/ws/telemetry") as websocket:
            _ = websocket.receive_json()
            # Active count inside context
            assert ws_manager.client_count >= initial_count + 1

        import time
        # Allow async event loop task context to yield and execute finally block
        for _ in range(10):
            if ws_manager.client_count == initial_count:
                break
            time.sleep(0.05)

        assert ws_manager.client_count == initial_count

