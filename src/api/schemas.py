'''Pydantic schemas for RailPulse FastAPI service.

These models define the request and response payloads for the API endpoints.
''' 

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for a single‑station delay prediction.

    Attributes
    ----------
    eva_id: int
        Unique EVA identifier of the station.
    lookback: Optional[List[float]]
        Optional custom look‑back feature vector (flattened, length = 12 * num_features).
        If omitted the engine will use the latest snapshot from the stored dataset.
    """

    eva_id: int = Field(..., description="Station EVA identifier")
    lookback: Optional[List[float]] = Field(
        None,
        description="Optional custom 12‑step look‑back feature vector (flattened).",
    )


class EdgeAttribution(BaseModel):
    """Dynamic GATv2 edge attention attribution across connected track segments."""

    source_eva: int
    target_eva: int
    source_name: Optional[str] = None
    target_name: Optional[str] = None
    attention_weight: float = Field(..., ge=0.0, le=1.0, description="Normalized dynamic GATv2 attention alpha_ij")


class MitigationDirective(BaseModel):
    """Dispatcher Decision Support recommendation for mitigating passenger transfer risks."""

    action: str = Field(..., description="'HOLD', 'DISPATCH_ON_TIME', or 'REROUTE'")
    target_train: str = Field(..., description="Connecting train service identifier")
    inbound_train: Optional[str] = Field(None, description="Inbound delayed feeder train")
    hold_duration_min: float = Field(..., ge=0.0, le=15.0, description="Recommended hold duration in minutes")
    connection_preservation_pct: float = Field(..., ge=0.0, le=100.0, description="Estimated connection save probability")
    secondary_delay_impact_min: float = Field(..., ge=0.0, description="Predicted downstream block propagation")
    rationale: str = Field(..., description="Operational explanation for dispatcher")


class PredictionResponse(BaseModel):
    """Response for a single‑station prediction."""

    eva_id: int
    station_name: str
    timestamp: str = Field(..., description="ISO‑8601 timestamp of the prediction request")
    delays: List[float] = Field(..., description="[+15m, +30m, +60m] delay predictions in minutes")
    confidence: Optional[float] = Field(
        None,
        description="Optional confidence / uncertainty score (0‑1).",
    )
    confidence_interval_min: Optional[Dict[str, List[float]]] = Field(
        None,
        description="95% confidence intervals per horizon [lower_bound, upper_bound]",
    )
    margin_min: Optional[Dict[str, float]] = Field(
        None,
        description="Confidence margin (+/- 1.96 * sigma) per horizon in minutes",
    )
    contributing_factors: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="XAI spatial attribution breakdown of upstream contributing stations",
    )
    edge_attributions: Optional[List[EdgeAttribution]] = Field(
        None,
        description="Dynamic GATv2 multi-head attention weights along connected tracks",
    )
    mitigation_directives: Optional[List[MitigationDirective]] = Field(
        None,
        description="Dispatcher Decision Support holding & mitigation recommendations",
    )


class StationInfo(BaseModel):
    """Metadata for a single station used in the topology response."""

    eva_id: int
    name: str
    lat: float
    lon: float
    degree_centrality: float
    betweenness: float


class StationTopologyResponse(BaseModel):
    """Topology information for the corridor network.

    The ``stations`` field contains a list of ``StationInfo`` objects. The ``edges``
    list describes directed connections – each entry is a dict with ``source``
    (EVA), ``target`` (EVA) and optional ``weight``.
    """

    stations: List[StationInfo]
    edges: List[dict]


class CorridorHealthResponse(BaseModel):
    """High‑level health / performance metrics of the deployed model."""

    model_config = {"protected_namespaces": ()}

    overall_mae: float
    overall_rmse: float
    overall_r2: float
    model_version: str = Field(..., description="Version / checkpoint identifier")
    gpu: bool = Field(..., description="Whether inference is running on a GPU")


class DisruptionSimulationRequest(BaseModel):
    """Request payload for simulating an injected delay shock at a station."""

    eva_id: int = Field(..., description="Target station EVA ID to inject delay shock into")
    injected_delay_min: float = Field(..., ge=0.0, le=180.0, description="Delay magnitude in minutes")


class PassengerConnectionRisk(BaseModel):
    """Vulnerability analysis for an individual scheduled passenger transfer."""

    eva_id: int
    station_name: str
    inbound_service: str
    connecting_service: str
    platform_change: str
    incoming_delay_min: float
    scheduled_transfer_margin_min: float
    effective_transfer_buffer_min: float
    connection_status: str = Field(..., description="'SAFE', 'VULNERABLE', or 'MISSED'")
    missed_connection_probability: float = Field(..., ge=0.0, le=1.0)
    mitigation_directives: Optional[List[MitigationDirective]] = Field(
        None,
        description="Recommended mitigation directives for this transfer",
    )


