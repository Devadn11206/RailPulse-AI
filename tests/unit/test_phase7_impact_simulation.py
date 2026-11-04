"""Phase 7 — Passenger Impact & Disruption Simulation Diagnostic Test Suite.

Tests PRD Module 05 (Passenger Transfer Impact) and PRD Module 04
(Disruption Propagation Simulator), along with their FastAPI endpoints.

Run with:
    pytest tests/unit/test_phase7_impact_simulation.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient

from src.analytics.passenger_impact import PassengerImpactEngine
from src.analytics.propagation_simulator import DisruptionSimulator
from src.api.inference_engine import RailPulseInferenceEngine
from src.api.main import app
from src.config import BASE_DIR, PROCESSED_DIR, TARGET_STATION_EVAS

# ===========================================================================
# 1. Passenger Impact Engine Unit Tests
# ===========================================================================


def test_passenger_impact_status_and_probabilities():
    """Verify SAFE, VULNERABLE, and MISSED connection logic and sigmoid calibration."""
    engine = PassengerImpactEngine(minimum_transfer_window_min=5.0)

    # Scenario A: Generous buffer -> SAFE (incoming delay 2m, margin 10m -> buffer 8m)
    res_safe = engine.evaluate_connection_risk(incoming_delay_min=2.0, scheduled_transfer_margin_min=10.0)
    assert res_safe["effective_transfer_buffer_min"] == 8.0
    assert res_safe["connection_status"] == "SAFE"
    assert res_safe["missed_connection_probability"] < 0.10

    # Scenario B: Tight buffer -> VULNERABLE (incoming delay 6m, margin 8m -> buffer 2m)
    res_vuln = engine.evaluate_connection_risk(incoming_delay_min=6.0, scheduled_transfer_margin_min=8.0)
    assert res_vuln["effective_transfer_buffer_min"] == 2.0
    assert res_vuln["connection_status"] == "VULNERABLE"
    assert 0.40 <= res_vuln["missed_connection_probability"] <= 0.85

    # Scenario C: Negative buffer -> MISSED (incoming delay 15m, margin 7m -> buffer -8m)
    res_missed = engine.evaluate_connection_risk(incoming_delay_min=15.0, scheduled_transfer_margin_min=7.0)
    assert res_missed["effective_transfer_buffer_min"] == -8.0
    assert res_missed["connection_status"] == "MISSED"
    assert res_missed["missed_connection_probability"] > 0.95


def test_corridor_connections_analysis():
    """Verify corridor-wide connection analysis ranks transfers by risk probability."""
    engine = PassengerImpactEngine(minimum_transfer_window_min=5.0)
    # Simulate high delay at Frankfurt Hbf and moderate at Mainz Hbf
    mock_preds = [
        {"eva_id": 8000105, "delays": [12.0, 14.0, 15.0]},
        {"eva_id": 8000240, "delays": [4.0, 5.0, 5.0]},
        {"eva_id": 8000244, "delays": [1.0, 1.0, 1.0]},
    ]
    evals = engine.analyze_corridor_connections(station_predictions=mock_preds)
    assert len(evals) > 0
    # Confirm descending risk probability ordering
    for i in range(len(evals) - 1):
        assert evals[i]["missed_connection_probability"] >= evals[i + 1]["missed_connection_probability"]

    # High delay at Frankfurt (margin 6m, delay 12m) should be marked MISSED
    frankfurt_s8 = next((c for c in evals if c["eva_id"] == 8000105 and "S8" in c["connecting_service"]), None)
    assert frankfurt_s8 is not None
    assert frankfurt_s8["connection_status"] == "MISSED"


# ===========================================================================
# 2. Disruption Propagation Simulator Unit Tests
# ===========================================================================


@pytest.fixture(scope="module")
def shared_inference_engine():
    model_path = BASE_DIR / "models" / "st_gnn_best.pth"
    npz_path = PROCESSED_DIR / "frankfurt_corridor_features.npz"
    return RailPulseInferenceEngine(model_path=model_path, npz_path=npz_path)


def test_disruption_simulator_propagation(shared_inference_engine):
    """Confirm injected delay shock updates origin and propagates downstream."""
    simulator = DisruptionSimulator(inference_engine=shared_inference_engine)

    # Injected +25.0 min delay shock at Frankfurt Hbf (8000105)
    shock_eva = 8000105
    injected_min = 25.0
    sim_res = simulator.simulate_station_shock(shock_eva_id=shock_eva, injected_delay_min=injected_min)

    assert "origin" in sim_res
    origin = sim_res["origin"]
    assert origin["eva_id"] == shock_eva
    assert origin["injected_shock_min"] == injected_min
    assert origin["delta_forecast"][0] > 0.0  # +15m forecast increased

    # Verify downstream neighbors are populated
    assert "downstream_1hop_neighbors" in sim_res
    downstream_1hop = sim_res["downstream_1hop_neighbors"]
    assert len(downstream_1hop) > 0

    # Downstream stations should be in corridor
    one_hop_evas = [d["eva_id"] for d in downstream_1hop]
    assert all(eva in TARGET_STATION_EVAS for eva in one_hop_evas)

    # Total network impact summary
    assert "network_impact" in sim_res
    assert "total_propagated_delay_15m" in sim_res["network_impact"]


def test_disruption_simulator_invalid_eva(shared_inference_engine):
    """Simulator raises ValueError on unknown EVA."""
    simulator = DisruptionSimulator(inference_engine=shared_inference_engine)
    with pytest.raises(ValueError, match="Unknown station EVA"):
        simulator.simulate_station_shock(shock_eva_id=9999999, injected_delay_min=20.0)


# ===========================================================================
# 3. FastAPI Endpoint Integration Tests
# ===========================================================================


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as c:
        yield c


def test_api_simulate_disruption_endpoint(api_client):
    """POST /api/v1/simulate/disruption returns valid propagation payload."""
    payload = {
        "eva_id": 8000105,
        "injected_delay_min": 20.0,
    }
    resp = api_client.post("/api/v1/simulate/disruption", json=payload)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
    body = resp.json()
    assert "origin" in body
    assert body["origin"]["eva_id"] == 8000105
    assert "downstream_1hop_neighbors" in body
    assert "network_impact" in body


def test_api_simulate_disruption_invalid_eva(api_client):
    """POST /api/v1/simulate/disruption with invalid EVA returns 404."""
    payload = {
        "eva_id": 1234567,
        "injected_delay_min": 15.0,
    }
    resp = api_client.post("/api/v1/simulate/disruption", json=payload)
    assert resp.status_code == 404


def test_api_passenger_connections_endpoint(api_client):
    """GET /api/v1/passenger/connections returns connection risks for corridor."""
    resp = api_client.get("/api/v1/passenger/connections")
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) > 0
    for conn in body:
        assert "eva_id" in conn
        assert "station_name" in conn
        assert "connection_status" in conn
        assert conn["connection_status"] in ("SAFE", "VULNERABLE", "MISSED")
        assert 0.0 <= conn["missed_connection_probability"] <= 1.0

