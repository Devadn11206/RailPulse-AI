"""Phase 5 — FastAPI Microservice Diagnostic Test Suite.

Tests the RailPulse AI API endpoints and PostGIS spatial handler.
Covers: health, topology, single-station prediction, corridor prediction,
invalid EVA handling, and GeoJSON format validation.

Run with:
    pytest tests/unit/test_phase5_api.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Imports — delayed so collection errors are readable
# ---------------------------------------------------------------------------
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import TARGET_STATION_EVAS
from src.database.postgis_handler import PostGISHandler

# ---------------------------------------------------------------------------
# Shared TestClient fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient — loads the model once for all tests."""
    with TestClient(app) as c:
        yield c


# ===========================================================================
# T1: Health endpoint
# ===========================================================================


def test_health_returns_200(client):
    """GET /health must return HTTP 200 with model_loaded=True."""
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("status") == "ok", f"status not 'ok': {body}"
    assert body.get("model_loaded") is True, f"model_loaded not True: {body}"
    assert "gpu" in body
    assert "vram_used_mb" in body


# ===========================================================================
# T2: Topology endpoint — basic structure
# ===========================================================================


def test_topology_returns_10_stations(client):
    """GET /api/v1/topology must return exactly 10 corridor stations."""
    resp = client.get("/api/v1/topology")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "stations" in body, "Response missing 'stations' key"
    assert "edges" in body, "Response missing 'edges' key"
    assert len(body["stations"]) == 10, (
        f"Expected 10 stations, got {len(body['stations'])}"
    )
    assert len(body["edges"]) > 0, "Edge list must not be empty"


# ===========================================================================
# T3: Topology endpoint — field validation
# ===========================================================================


def test_topology_station_fields(client):
    """Each station in topology must have valid lat/lon and centrality fields."""
    resp = client.get("/api/v1/topology")
    assert resp.status_code == 200
    for station in resp.json()["stations"]:
        assert "eva_id" in station
        assert "name" in station
        assert isinstance(station["lat"], float), "lat must be float"
        assert isinstance(station["lon"], float), "lon must be float"
        assert -90 <= station["lat"] <= 90, f"lat out of range: {station['lat']}"
        assert -180 <= station["lon"] <= 180, f"lon out of range: {station['lon']}"
        assert station["degree_centrality"] >= 0.0
        assert station["betweenness"] >= 0.0


# ===========================================================================
# T4: Single-station prediction — valid EVA
# ===========================================================================


def test_predict_station_valid_eva(client):
    """POST /api/v1/predict/station with a known EVA must return 3 finite delays."""
    payload = {"eva_id": TARGET_STATION_EVAS[0]}  # Frankfurt Hbf
    resp = client.post("/api/v1/predict/station", json=payload)
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
    body = resp.json()
    assert "delays" in body, f"'delays' missing in response: {body}"
    delays = body["delays"]
    assert len(delays) == 3, f"Expected 3 delay horizons, got {len(delays)}"
    for d in delays:
        assert isinstance(d, (int, float)), f"Non-numeric delay: {d}"
        assert d == d, "Delay is NaN"  # NaN != NaN in Python
    assert body["eva_id"] == TARGET_STATION_EVAS[0]
    assert isinstance(body["station_name"], str) and len(body["station_name"]) > 0


# ===========================================================================
# T5: Single-station prediction — invalid EVA
# ===========================================================================


def test_predict_station_invalid_eva(client):
    """POST /api/v1/predict/station with an unknown EVA must return 404."""
    payload = {"eva_id": 9999999}
    resp = client.post("/api/v1/predict/station", json=payload)
    assert resp.status_code == 404, (
        f"Expected 404 for unknown EVA, got {resp.status_code}: {resp.text}"
    )


# ===========================================================================
# T6: Corridor prediction — all 10 stations
# ===========================================================================


def test_predict_corridor_returns_all_stations(client):
    """GET /api/v1/predict/corridor must return predictions for all 10 stations."""
    resp = client.get("/api/v1/predict/corridor")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list), "Response must be a list"
    assert len(body) == 10, f"Expected 10 entries, got {len(body)}"
    for entry in body:
        assert "delays" in entry and len(entry["delays"]) == 3
        assert "eva_id" in entry
        assert "station_name" in entry
        # All delays must be finite numbers
        for d in entry["delays"]:
            assert isinstance(d, (int, float))
            assert d == d, "Delay is NaN"


# ===========================================================================
# T7: PostGIS handler — station GeoJSON
# ===========================================================================


def test_postgis_station_geojson():
    """PostGISHandler.get_station_geojson() must return a valid FeatureCollection."""
    handler = PostGISHandler()
    geojson = handler.get_station_geojson()
    assert geojson["type"] == "FeatureCollection", "type must be FeatureCollection"
    features = geojson["features"]
    assert len(features) == 10, f"Expected 10 station features, got {len(features)}"
    for feat in features:
        assert feat["type"] == "Feature"
        geom = feat["geometry"]
        assert geom["type"] == "Point"
        coords = geom["coordinates"]
        assert len(coords) == 2, "Point must have exactly [lon, lat]"
        lon, lat = coords
        assert -180 <= lon <= 180, f"lon out of WGS84 range: {lon}"
        assert -90 <= lat <= 90, f"lat out of WGS84 range: {lat}"
        assert "eva_id" in feat["properties"]
        assert "name" in feat["properties"]


# ===========================================================================
# T8: PostGIS handler — track GeoJSON
# ===========================================================================


def test_postgis_track_geojson():
    """PostGISHandler.get_track_geojson() must return LineString features."""
    handler = PostGISHandler()
    geojson = handler.get_track_geojson()
    assert geojson["type"] == "FeatureCollection"
    features = geojson["features"]
    assert len(features) > 0, "Track GeoJSON must have at least one feature"
    for feat in features:
        assert feat["type"] == "Feature"
        geom = feat["geometry"]
        assert geom["type"] == "LineString", f"Expected LineString, got {geom['type']}"
        coords = geom["coordinates"]
        assert len(coords) == 2, "LineString must have exactly 2 coordinate pairs"
        for point in coords:
            lon, lat = point
            assert -180 <= lon <= 180
            assert -90 <= lat <= 90
        props = feat["properties"]
        assert "source_eva" in props
        assert "target_eva" in props
        assert "weight_minutes" in props
        assert props["weight_minutes"] > 0

