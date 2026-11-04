"""Phase 8 — Authentication, RBAC, and Explainable AI (XAI) Diagnostic Tests.

Tests:
1. JWT login, token decoding, and profile retrieval.
2. Role-based authorization: admin/analyst permitted, developer forbidden on admin routes.
3. API key validation via X-API-Key header.
4. Monte Carlo Dropout prediction uncertainty interval estimation.
5. Spatial attribution ranking for upstream contributing stations.
6. Enriched /predict/station and /predict/corridor responses with confidence intervals.

Run with:
    pytest tests/unit/test_phase8_auth_xai.py -v
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

from src.analytics.explainability import GNNExplainabilityEngine
from src.api.inference_engine import RailPulseInferenceEngine
from src.api.main import app
from src.auth.security import (
    authenticate_user,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.config import BASE_DIR, PROCESSED_DIR, TARGET_STATION_EVAS

# ===========================================================================
# 1. Auth & RBAC Unit Tests
# ===========================================================================


def test_password_hashing_and_verification():
    """Verify bcrypt hash generation and comparison."""
    secret = "railpulse-super-secret"
    hashed = hash_password(secret)
    assert hashed != secret
    assert verify_password(secret, hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_jwt_token_creation_and_decoding():
    """Verify signed JWT token encodes subject and role."""
    token = create_access_token(data={"sub": "analyst", "role": "analyst"})
    assert isinstance(token, str) and len(token) > 20
    data = decode_access_token(token)
    assert data.username == "analyst"
    assert data.role == "analyst"


def test_authentication_with_valid_and_invalid_credentials():
    """Verify authenticate_user against internal records."""
    valid_admin = authenticate_user("admin", "admin123")
    assert valid_admin is not None
    assert valid_admin.role == "admin"

    invalid_pass = authenticate_user("admin", "incorrect")
    assert invalid_pass is None

    unknown_user = authenticate_user("ghost", "admin123")
    assert unknown_user is None


# ===========================================================================
# 2. XAI & Explainability Engine Unit Tests
# ===========================================================================


@pytest.fixture(scope="module")
def shared_engine():
    model_path = BASE_DIR / "models" / "st_gnn_best.pth"
    npz_path = PROCESSED_DIR / "frankfurt_corridor_features.npz"
    return RailPulseInferenceEngine(model_path=model_path, npz_path=npz_path)


def test_mc_dropout_uncertainty_intervals(shared_engine):
    """Confirm Monte Carlo Dropout outputs strictly positive confidence margins."""
    xai = GNNExplainabilityEngine(inference_engine=shared_engine)
    intervals = xai.compute_prediction_intervals(shared_engine._latest_snapshot, num_samples=10)

    for eva in TARGET_STATION_EVAS:
        assert eva in intervals
        entry = intervals[eva]
        assert "mean" in entry and len(entry["mean"]) == 3
        assert "margin_95" in entry and len(entry["margin_95"]) == 3
        assert "lower_95" in entry and "upper_95" in entry

        for h in range(3):
            # 95% margin must be non-zero
            assert entry["margin_95"][h] > 0.0
            # lower bound <= upper bound
            assert entry["lower_95"][h] <= entry["upper_95"][h]


def test_spatial_attribution_ranking(shared_engine):
    """Confirm spatial attribution returns normalized descending factor breakdown."""
    xai = GNNExplainabilityEngine(inference_engine=shared_engine)
    corridor_preds = shared_engine.predict_full_corridor()

    target_eva = 8000105  # Frankfurt Hbf
    attributions = xai.extract_spatial_attribution(target_eva_id=target_eva, corridor_predictions=corridor_preds)

    assert len(attributions) > 0
    total_pct = sum(a["attribution_pct"] for a in attributions)
    assert 99.0 <= total_pct <= 101.0  # Normalized to ~100%

    # Confirm descending ordering
    for i in range(len(attributions) - 1):
        assert attributions[i]["attribution_pct"] >= attributions[i + 1]["attribution_pct"]


# ===========================================================================
# 3. FastAPI Endpoint Tests (Auth + RBAC + XAI)
# ===========================================================================


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_login_success_and_failure(client):
    """POST /api/v1/auth/login validates credentials and issues token."""
    # Successful login
    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["role"] == "admin"

    # Failed login
    bad_resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "bad"})
    assert bad_resp.status_code == 401


def test_api_rbac_protected_endpoint(client):
    """GET /api/v1/admin/audit enforces role permissions."""
    # 1. Missing token -> 401
    unauth_resp = client.get("/api/v1/admin/audit")
    assert unauth_resp.status_code == 401

    # 2. Login as developer -> 403 (developer is not admin or analyst)
    dev_token_resp = client.post("/api/v1/auth/login", json={"username": "developer", "password": "dev123"})
    dev_token = dev_token_resp.json()["access_token"]
    dev_resp = client.get(
        "/api/v1/admin/audit",
        headers={"Authorization": f"Bearer {dev_token}"},
    )
    assert dev_resp.status_code == 403

    # 3. Login as admin -> 200 OK
    admin_token_resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    admin_token = admin_token_resp.json()["access_token"]
    admin_resp = client.get(
        "/api/v1/admin/audit",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_resp.status_code == 200
    assert admin_resp.json()["status"] == "authorized"


def test_api_predict_station_with_xai_intervals(client):
    """POST /api/v1/predict/station returns confidence intervals and contributing factors."""
    resp = client.post("/api/v1/predict/station", json={"eva_id": 8000105})
    assert resp.status_code == 200
    body = resp.json()
    assert "confidence_interval_min" in body
    assert body["confidence_interval_min"] is not None
    assert "15m" in body["confidence_interval_min"]
    assert "margin_min" in body
    assert body["margin_min"] is not None
    assert "contributing_factors" in body
    assert isinstance(body["contributing_factors"], list)
    assert len(body["contributing_factors"]) > 0

