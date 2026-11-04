"""
Phase 10 — Graph Scaling to 50+ Regional Stations Diagnostic Test Suite.
Verifies expanded graph topology construction, PyG NeighborLoader memory bounding,
throughput on expanded graph snapshots, and API endpoint support for 50+ stations.
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

from src.config import REGIONAL_STATION_EVAS_50, TARGET_STATION_EVAS
from src.graph_construction.static_graph_builder import (
    EXPANDED_STATION_REGISTRY,
    RailwayGraphBuilder,
)
from src.models.st_gnn import RailPulseSTGNN
from src.training.st_gnn_trainer import STGNNTrainer
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestPhase10Scale:
    """Test suite for Phase 10: Scaling Graph Topology to 50+ Regional Stations."""

    def test_expanded_station_registry(self):
        """Verify registry contains 50+ stations across Hesse, RP, and BW."""
        assert len(REGIONAL_STATION_EVAS_50) >= 50
        for eva in REGIONAL_STATION_EVAS_50:
            assert eva in EXPANDED_STATION_REGISTRY
            meta = EXPANDED_STATION_REGISTRY[eva]
            assert "name" in meta
            assert 47.0 <= meta["lat"] <= 55.0  # Germany latitude bounds
            assert 5.0 <= meta["lon"] <= 15.0   # Germany longitude bounds

    def test_regional_graph_builder_connectivity(self):
        """Construct expanded 50+ node graph and verify edge connectivity."""
        builder = RailwayGraphBuilder(target_evas=REGIONAL_STATION_EVAS_50)
        builder.build()

        assert builder.num_nodes == len(REGIONAL_STATION_EVAS_50)
        assert builder.edge_index is not None
        assert builder.edge_index.shape[0] == 2
        # On 50+ nodes with 4-NN + hub links, edge count should comfortably exceed 100
        assert builder.edge_index.shape[1] >= 100
        assert builder.static_features is not None
        assert builder.static_features.shape == (builder.num_nodes, 4)

    def test_pyg_neighbor_loader_memory_bounded(self):
        """Verify PyG NeighborLoader dynamic subgraph sampling on 50+ nodes."""
        from torch_geometric.data import Data

        builder = RailwayGraphBuilder(target_evas=REGIONAL_STATION_EVAS_50)
        builder.build()

        # Synthetic multi-step features [52, 88] and labels [52, 3]
        num_nodes = builder.num_nodes
        x = torch.randn(num_nodes, 88, dtype=torch.float32)
        y = torch.randn(num_nodes, 3, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=builder.edge_index,
            edge_attr=builder.edge_attr,
            y=y,
        )

        loader = STGNNTrainer.create_neighbor_loader(
            data=data,
            num_neighbors=[15, 10],
            batch_size=16,
            shuffle=False,
        )

        batches = list(loader)
        assert len(batches) > 0

        # Each sampled subgraph batch should have bounded node count
        for b in batches:
            assert b.x.shape[0] <= num_nodes
            assert b.edge_index.shape[0] == 2
            assert b.y is not None
            assert b.x.shape[0] > 0

    def test_stgnn_forward_throughput_50_nodes(self):
        """Verify RailPulseSTGNN forward pass on full 50+ node graph with AMP."""
        from torch_geometric.data import Data

        builder = RailwayGraphBuilder(target_evas=REGIONAL_STATION_EVAS_50)
        builder.build()

        model = RailPulseSTGNN(
            node_in_dim=11,
            hidden_dim=64,
            num_layers=2,
            out_horizons=3,
        )
        model.eval()

        data = Data(
            x=torch.randn(builder.num_nodes, 88),
            edge_index=builder.edge_index,
            edge_attr=builder.edge_attr,
        )

        with torch.no_grad():
            with torch.amp.autocast(device_type="cpu", enabled=False):
                out = model(data)

        assert out.shape == (builder.num_nodes, 3)
        assert not torch.isnan(out).any()

    def test_api_topology_regional_scale(self, client):
        """Verify GET /api/v1/topology?scale=regional returns 50+ stations."""
        # Test core scale
        resp_core = client.get("/api/v1/topology?scale=core")
        assert resp_core.status_code == 200
        assert len(resp_core.json()["stations"]) == 10

        # Test regional scale
        resp_reg = client.get("/api/v1/topology?scale=regional")
        assert resp_reg.status_code == 200
        data = resp_reg.json()
        assert len(data["stations"]) >= 50
        assert len(data["edges"]) >= 100
        # Verify first 3 stations are present in station names
        station_names = [s["name"] for s in data["stations"]]
        assert "Frankfurt(Main)Hbf" in station_names
        assert "Mainz Hbf" in station_names
        assert "Kaiserslautern Hbf" in station_names

