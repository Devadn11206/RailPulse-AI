"""RailPulse AI — FastAPI application entry-point.

Endpoints
---------
GET  /health                      — Model & GPU status
GET  /api/v1/topology             — Station network graph topology
POST /api/v1/predict/station      — Single-station delay forecast
GET  /api/v1/predict/corridor     — All-corridor delay forecast
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.explainability import GNNExplainabilityEngine
from src.analytics.passenger_impact import PassengerImpactEngine
from src.analytics.propagation_simulator import DisruptionSimulator
from src.analytics.mitigation_engine import DispatcherDecisionSupportEngine
from src.api.inference_engine import STATION_METADATA, RailPulseInferenceEngine
from src.api.schemas import (
    CorridorHealthResponse,
    DisruptionSimulationRequest,
    PassengerConnectionRisk,
    PredictionRequest,
    PredictionResponse,
    StationInfo,
    StationTopologyResponse,
)
from src.auth.models import LoginRequest, Token, UserInDB, UserResponse
from src.auth.security import (
    authenticate_user,
    create_access_token,
    get_current_user,
    require_role,
    verify_api_key,
)
from src.api.websocket_manager import ConnectionManager

ws_manager = ConnectionManager()
from src.config import BASE_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = BASE_DIR / "models" / "st_gnn_best.pth"
NPZ_PATH = PROCESSED_DIR / "frankfurt_corridor_features.npz"

# ---------------------------------------------------------------------------
# Shared state — populated at startup
# ---------------------------------------------------------------------------
_engine: RailPulseInferenceEngine | None = None
_passenger_engine: PassengerImpactEngine | None = None
_simulator: DisruptionSimulator | None = None
_explainability_engine: GNNExplainabilityEngine | None = None
_dss_engine: DispatcherDecisionSupportEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    global _engine, _passenger_engine, _simulator, _explainability_engine
    logger.info("Starting RailPulse AI API — loading inference engine …")
    _engine = RailPulseInferenceEngine(model_path=MODEL_PATH, npz_path=NPZ_PATH)
    _passenger_engine = PassengerImpactEngine()
    _simulator = DisruptionSimulator(inference_engine=_engine)
    _explainability_engine = GNNExplainabilityEngine(inference_engine=_engine)
    _dss_engine = DispatcherDecisionSupportEngine()
    logger.info("Inference, Passenger, Simulator, Explainability, and DSS engines ready.")
    yield
    # Shutdown — nothing to clean up for torch CPU/GPU inference
    logger.info("RailPulse AI API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RailPulse AI API",
    version="1.0.0",
    description=(
        "Real-time railway delay prediction API powered by a "
        "Spatio-Temporal Graph Neural Network (ST-GNN)."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------
def _get_engine() -> RailPulseInferenceEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialised.")
    return _engine


def _get_passenger_engine() -> PassengerImpactEngine:
    global _passenger_engine
    if _passenger_engine is None:
        _passenger_engine = PassengerImpactEngine()
    return _passenger_engine


def _get_simulator() -> DisruptionSimulator:
    global _simulator
    if _simulator is None:
        engine = _get_engine()
        _simulator = DisruptionSimulator(inference_engine=engine)
    return _simulator


def _get_explainability_engine() -> GNNExplainabilityEngine:
    global _explainability_engine
    if _explainability_engine is None:
        engine = _get_engine()
        _explainability_engine = GNNExplainabilityEngine(inference_engine=engine)
    return _explainability_engine


def _get_dss_engine() -> DispatcherDecisionSupportEngine:
    global _dss_engine
    if _dss_engine is None:
        _dss_engine = DispatcherDecisionSupportEngine()
    return _dss_engine


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health():
    """Return model load status, GPU availability, and VRAM usage."""
    import torch

    engine = _get_engine()
    gpu = engine.device.type == "cuda"
    vram_mb = 0.0
    if gpu:
        try:
            vram_mb = torch.cuda.memory_allocated(engine.device) / 1024 ** 2
        except Exception:
            pass
    return {
        "status": "ok",
        "model_loaded": True,
        "gpu": gpu,
        "vram_used_mb": round(vram_mb, 2),
        "model_version": engine.model_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/api/v1/topology",
    response_model=StationTopologyResponse,
    summary="Station network topology",
)
async def get_topology(scale: str = "core"):
    """Return the static corridor station graph (nodes + directed edges).
    Supports scale='core' (10 stations) or scale='regional' (50+ stations).
    """
    engine = _get_engine()
    topo = engine.get_topology(scale=scale)
    stations = [StationInfo(**s) for s in topo["stations"]]
    return StationTopologyResponse(stations=stations, edges=topo["edges"])


@app.post(
    "/api/v1/auth/login",
    response_model=Token,
    summary="Login to obtain JWT access token",
)
async def login(credentials: LoginRequest):
    """Authenticate with username/password and receive a signed Bearer token."""
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )
    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
        expires_in_minutes=60,
    )


@app.get(
    "/api/v1/auth/me",
    response_model=UserResponse,
    summary="Get current user profile and role",
)
async def get_me(current_user: UserInDB = Depends(get_current_user)):
    """Return the profile and role of the currently authenticated user."""
    return UserResponse(
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        disabled=current_user.disabled,
    )


@app.get(
    "/api/v1/admin/audit",
    summary="Role-protected audit endpoint (Requires 'admin' or 'analyst' role)",
)
async def admin_audit(
    current_user: UserInDB = Depends(require_role(["admin", "analyst"])),
):
    """Audit endpoint accessible only by authorized roles."""
    return {
        "status": "authorized",
        "user": current_user.username,
        "role": current_user.role,
        "access_granted_at": datetime.now(timezone.utc).isoformat(),
        "audit_code": "SEC-RBAC-VERIFIED",
    }


@app.post(
    "/api/v1/predict/station",
    response_model=PredictionResponse,
    summary="Single-station delay forecast",
)
async def predict_station(request: PredictionRequest):
    """Return +15m, +30m, +60m delay predictions for the requested station with XAI intervals."""
    engine = _get_engine()
    xai = _get_explainability_engine()
    try:
        preds = engine.predict_station_delay(
            eva_id=request.eva_id,
            lookback=request.lookback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    meta = STATION_METADATA.get(
        request.eva_id,
        {"name": f"Station {request.eva_id}", "lat": 0.0, "lon": 0.0},
    )

    # Compute Monte Carlo Dropout prediction intervals
    delays_list = [preds["delay_15m"], preds["delay_30m"], preds["delay_60m"]]
    mcd_intervals = xai.compute_prediction_intervals(engine._latest_snapshot, num_samples=10)
    station_mcd = mcd_intervals.get(request.eva_id)

    # Compute spatial attribution factors and GAT dynamic attention
    corridor_preds = engine.predict_full_corridor()
    factors = xai.extract_spatial_attribution(request.eva_id, corridor_preds)
    gat_attributions = xai.extract_gat_attention()
    target_edge_attributions = [
        a for a in gat_attributions if a["target_eva"] == request.eva_id or a["source_eva"] == request.eva_id
    ][:5]

    # Evaluate Dispatcher Decision Support (DSS) mitigation directives
    dss = _get_dss_engine()
    directives = dss.evaluate_transfer_mitigation(
        station_id=request.eva_id,
        predicted_delays=delays_list,
    )

    ci_dict = None
    margin_dict = None
    if station_mcd:
        ci_dict = {
            "15m": [station_mcd["lower_95"][0], station_mcd["upper_95"][0]],
            "30m": [station_mcd["lower_95"][1], station_mcd["upper_95"][1]],
            "60m": [station_mcd["lower_95"][2], station_mcd["upper_95"][2]],
        }
        margin_dict = {
            "15m": station_mcd["margin_95"][0],
            "30m": station_mcd["margin_95"][1],
            "60m": station_mcd["margin_95"][2],
        }

    return PredictionResponse(
        eva_id=request.eva_id,
        station_name=meta["name"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        delays=delays_list,
        confidence=0.92,
        confidence_interval_min=ci_dict,
        margin_min=margin_dict,
        contributing_factors=factors[:3],
        edge_attributions=target_edge_attributions if target_edge_attributions else None,
        mitigation_directives=directives if directives else None,
    )


@app.get(
    "/api/v1/predict/corridor",
    response_model=List[PredictionResponse],
    summary="Corridor-wide delay forecast",
)
async def predict_corridor():
    """Return delay predictions for all 10 corridor hub stations with MC uncertainty margins."""
    engine = _get_engine()
    xai = _get_explainability_engine()
    corridor_preds = engine.predict_full_corridor()
    mcd_intervals = xai.compute_prediction_intervals(engine._latest_snapshot, num_samples=10)
    gat_attributions = xai.extract_gat_attention()
    top_corridor_attributions = gat_attributions[:10] if gat_attributions else None

    responses = []
    for p in corridor_preds:
        eva = p["eva_id"]
        station_mcd = mcd_intervals.get(eva)
        ci_dict = None
        margin_dict = None
        if station_mcd:
            ci_dict = {
                "15m": [station_mcd["lower_95"][0], station_mcd["upper_95"][0]],
                "30m": [station_mcd["lower_95"][1], station_mcd["upper_95"][1]],
                "60m": [station_mcd["lower_95"][2], station_mcd["upper_95"][2]],
            }
            margin_dict = {
                "15m": station_mcd["margin_95"][0],
                "30m": station_mcd["margin_95"][1],
                "60m": station_mcd["margin_95"][2],
            }

        responses.append(
            PredictionResponse(
                eva_id=p["eva_id"],
                station_name=p["station_name"],
                timestamp=p["timestamp"],
                delays=[p["delay_15m"], p["delay_30m"], p["delay_60m"]],
                confidence=0.94,
                confidence_interval_min=ci_dict,
                margin_min=margin_dict,
                contributing_factors=None,
                edge_attributions=top_corridor_attributions,
            )
        )
    return responses


@app.post(
    "/api/v1/simulate/disruption",
    summary="Simulate injected delay shock propagation across corridor",
)
async def simulate_disruption(request: DisruptionSimulationRequest):
    """Simulates multi-hop disruption propagation through the ST-GNN topology."""
    simulator = _get_simulator()
    try:
        results = simulator.simulate_station_shock(
            shock_eva_id=request.eva_id,
            injected_delay_min=request.injected_delay_min,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return results


@app.get(
    "/api/v1/passenger/connections",
    response_model=List[PassengerConnectionRisk],
    summary="Analyze passenger transfer connection vulnerabilities across corridor",
)
async def get_passenger_connections():
    """Evaluates transfer margins and missed connection probabilities under current delays."""
    engine = _get_engine()
    corridor_preds = engine.predict_full_corridor()
    passenger_engine = _get_passenger_engine()
    dss = _get_dss_engine()

    evaluations = passenger_engine.analyze_corridor_connections(station_predictions=corridor_preds)

    # Attach DSS mitigation directives for each connection risk
    for item in evaluations:
        eva = item.get("eva_id")
        d_in = item.get("incoming_delay_min", 0.0)
        directives = dss.evaluate_transfer_mitigation(
            station_id=eva,
            predicted_delays=[d_in],
            connection_data=[item],
        )
        item["mitigation_directives"] = directives

    return evaluations


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Sub-second live ST-GNN inference and GATv2 telemetry streaming endpoint.
    Streams delay forecasts, attention weights (alpha_ij), and MCD uncertainty bounds every 2s.
    """
    await ws_manager.connect(websocket)
    engine = _get_engine()
    xai = _get_explainability_engine()
    dss = _get_dss_engine()

    try:
        # Stream telemetry snapshots periodically to connected client
        while True:
            corridor_preds = engine.predict_full_corridor()
            mcd_intervals = xai.compute_prediction_intervals(engine._latest_snapshot, num_samples=10)
            gat_attributions = xai.extract_gat_attention()

            # High-priority mitigation directives for corridor
            corridor_directives = []
            for p in corridor_preds:
                dirs = dss.evaluate_transfer_mitigation(p["eva_id"], [p["delay_15m"]])
                corridor_directives.extend(dirs)

            telemetry_packet = {
                "type": "TELEMETRY_SNAPSHOT",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stations_count": len(corridor_preds),
                "predictions": corridor_preds,
                "confidence_intervals": mcd_intervals,
                "gat_edge_attributions": gat_attributions[:15] if gat_attributions else [],
                "mitigation_directives": corridor_directives[:8],
            }

            await websocket.send_json(telemetry_packet)
            await asyncio.sleep(2.0)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        logging.getLogger(__name__).warning("WebSocket client connection terminated: %s", exc)
    finally:
        try:
            await ws_manager.disconnect(websocket)
        except Exception:
            pass



