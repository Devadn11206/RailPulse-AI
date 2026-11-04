"""
Phase 11 — GATv2 Dynamic Spatial Attention & Graph Explainability Diagnostic Test Suite.
Verifies GATv2 multi-head attention forward pass, dynamic alpha_ij extraction,
and API endpoint enrichment with edge_attributions.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.st_gnn import RailPulseSTGNN
from src.analytics.explainability import GNNExplainabilityEngine
from src.api.inference_engine import RailPulseInferenceEngine
from src.api.main import app, _get_engine, _get_explainability_engine


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestPhase11GATv2:
    """Test suite for Phase 11 GATv2 dynamic spatial attention and XAI."""

    def test_gatv2_model_instantiation_and_forward(self):
        """Verify GATv2 multi-head forward pass output shape and finite values."""
        model = RailPulseSTGNN(
            node_in_dim=11,
            hidden_dim=64,
            num_layers=2,
            out_horizons=3,
            heads=4,
            use_gatv2=True,
        )
        model.eval()

        data = RailPulseSTGNN.create_synthetic_data(num_nodes=10, num_edges=20)
        with torch.no_grad():
            out = model(data, return_attention_weights=False)

        assert out.shape == (10, 3)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gatv2_attention_weights_extraction(self):
        """Verify dynamic attention matrix extraction (alpha_ij) across edges."""
        model = RailPulseSTGNN(
            node_in_dim=11,
            hidden_dim=64,
            num_layers=2,
            out_horizons=3,
            heads=4,
            use_gatv2=True,
        )
        model.eval()

        data = RailPulseSTGNN.create_synthetic_data(num_nodes=10, num_edges=20)
        with torch.no_grad():
            out, (edge_index_attn, alpha) = model(data, return_attention_weights=True)

        assert out.shape == (10, 3)
        assert edge_index_attn.shape[0] == 2
        # Alpha should have [E_attn, heads]
        assert alpha.shape[0] == edge_index_attn.shape[1]
        assert alpha.shape[1] == 4  # 4 heads

        # Verify attention normalization (values in [0, 1])
        assert (alpha >= 0.0).all()
        assert (alpha <= 1.0).all()

    def test_explainability_extract_gat_attention(self, client):
        """Verify GNNExplainabilityEngine maps attention scores to station EVAs."""
        engine = _get_engine()
        xai = _get_explainability_engine()

        edge_attributions = xai.extract_gat_attention()
        assert isinstance(edge_attributions, list)
        assert len(edge_attributions) > 0

        first_edge = edge_attributions[0]
        assert "source_eva" in first_edge
        assert "target_eva" in first_edge
        assert "source_name" in first_edge
        assert "target_name" in first_edge
        assert "attention_weight" in first_edge
        assert 0.0 <= first_edge["attention_weight"] <= 1.0

    def test_api_predict_station_returns_edge_attributions(self, client):
        """Verify POST /api/v1/predict/station contains edge_attributions array."""
        resp = client.post("/api/v1/predict/station", json={"eva_id": 8000105})
        assert resp.status_code == 200
        data = resp.json()

        assert "edge_attributions" in data
        assert data["edge_attributions"] is not None
        assert isinstance(data["edge_attributions"], list)
        if len(data["edge_attributions"]) > 0:
            attr = data["edge_attributions"][0]
            assert "source_eva" in attr
            assert "target_eva" in attr
            assert "attention_weight" in attr

    def test_api_predict_corridor_returns_edge_attributions(self, client):
        """Verify GET /api/v1/predict/corridor contains edge_attributions."""
        resp = client.get("/api/v1/predict/corridor")
        assert resp.status_code == 200
        corridor = resp.json()
        assert len(corridor) > 0

        first = corridor[0]
        assert "edge_attributions" in first
        assert first["edge_attributions"] is not None
        assert isinstance(first["edge_attributions"], list)

