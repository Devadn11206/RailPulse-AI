"""
Phase 13 — Dispatcher Decision Support System (DSS) & Mitigation Engine Tests.
Verifies hold-delay trade-off logic across delay levels, connection preservation calculations,
and API/WebSocket integration for mitigation directives.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.mitigation_engine import DispatcherDecisionSupportEngine
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestPhase13Mitigation:
    """Test suite for Phase 13 Dispatcher Decision Support (DSS)."""

    def test_low_delay_dispatch_on_time(self):
        """When delay is minimal (< 2 min), DSS should recommend DISPATCH_ON_TIME."""
        dss = DispatcherDecisionSupportEngine()
        # 1.0 min delay with 8.0 min margin -> effective buffer = 7.0 min (>= 5 min safe)
        dirs = dss.evaluate_transfer_mitigation(
            station_id=8000105,
            predicted_delays={"delay_15m": 1.0},
            connection_data=[{
                "eva_id": 8000105,
                "inbound_service": "RE 50",
                "connecting_service": "ICE 27",
                "scheduled_transfer_margin_min": 8.0,
            }]
        )
        assert len(dirs) == 1
        d0 = dirs[0]
        assert d0["action"] == "DISPATCH_ON_TIME"
        assert d0["hold_duration_min"] == 0.0
        assert d0["secondary_delay_impact_min"] == 0.0
        assert d0["connection_preservation_pct"] >= 95.0

    def test_moderate_delay_hold_tradeoff(self):
        """When delay creates a vulnerable buffer, DSS recommends HOLD within bounds [2, 8] min."""
        dss = DispatcherDecisionSupportEngine()
        # 6.0 min delay with 7.0 min margin -> effective buffer = +1.0 min (< 5 min)
        dirs = dss.evaluate_transfer_mitigation(
            station_id=8000240,
            predicted_delays=[6.0],
            connection_data=[{
                "eva_id": 8000240,
                "inbound_service": "RE 3",
                "connecting_service": "ICE 26",
                "scheduled_transfer_margin_min": 7.0,
            }]
        )
        assert len(dirs) == 1
        d0 = dirs[0]
        assert d0["action"] == "HOLD"
        assert 2.0 <= d0["hold_duration_min"] <= 8.0
        assert d0["secondary_delay_impact_min"] > 0.0
        assert d0["connection_preservation_pct"] > 50.0

    def test_severe_delay_reroute_recommendation(self):
        """When delay exceeds viable hold bounds (> 15 min), DSS recommends REROUTE."""
        dss = DispatcherDecisionSupportEngine()
        # 20 min delay with 6 min margin -> effective buffer = -14 min (< -8 min threshold)
        dirs = dss.evaluate_transfer_mitigation(
            station_id=8000105,
            predicted_delays=[20.0],
            connection_data=[{
                "eva_id": 8000105,
                "inbound_service": "RE 50",
                "connecting_service": "S-Bahn S8",
                "scheduled_transfer_margin_min": 6.0,
            }]
        )
        assert len(dirs) == 1
        d0 = dirs[0]
        assert d0["action"] == "REROUTE"
        assert d0["hold_duration_min"] == 0.0
        assert "reroute" in d0["rationale"].lower()

    def test_api_predict_station_mitigation_directives(self, client):
        """Verify POST /api/v1/predict/station returns mitigation_directives."""
        resp = client.post("/api/v1/predict/station", json={"eva_id": 8000105})
        assert resp.status_code == 200
        data = resp.json()
        assert "mitigation_directives" in data
        assert data["mitigation_directives"] is not None
        assert isinstance(data["mitigation_directives"], list)

        if len(data["mitigation_directives"]) > 0:
            m = data["mitigation_directives"][0]
            assert "action" in m
            assert "target_train" in m
            assert "hold_duration_min" in m
            assert "connection_preservation_pct" in m

    def test_api_passenger_connections_mitigation_directives(self, client):
        """Verify GET /api/v1/passenger/connections attaches mitigation_directives."""
        resp = client.get("/api/v1/passenger/connections")
        assert resp.status_code == 200
        connections = resp.json()
        assert len(connections) > 0

        first = connections[0]
        assert "mitigation_directives" in first
        assert first["mitigation_directives"] is not None

    def test_websocket_telemetry_includes_mitigation_directives(self, client):
        """Verify WebSocket stream broadcasts mitigation_directives in telemetry packets."""
        with client.websocket_connect("/ws/telemetry") as websocket:
            data = websocket.receive_json()
            assert "mitigation_directives" in data
            assert isinstance(data["mitigation_directives"], list)

