"""Phase 2 spatio-temporal railway graph diagnostic tests."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_STATION_EVAS, PROCESSED_DIR
from src.graph_construction.static_graph_builder import RailwayGraphBuilder
from src.graph_construction.temporal_graph_dataset import (
    RailwayGraphDataset,
    create_pyg_dataloaders,
)

NPZ_PATH = PROCESSED_DIR / "frankfurt_corridor_features.npz"


# ---------------------------------------------------------------------------
# RailwayGraphBuilder tests
# ---------------------------------------------------------------------------

def test_builder_node_count_matches_target_evas():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    G = builder.build_networkx_graph()
    assert G.number_of_nodes() == len(TARGET_STATION_EVAS) == 10
    assert set(builder.eva_to_idx.keys()) == set(TARGET_STATION_EVAS)
    assert set(builder.eva_to_idx.values()) == set(range(len(TARGET_STATION_EVAS)))


def test_builder_edge_index_shape_and_dtype():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    G = builder.build_networkx_graph()
    edge_index, edge_attr = builder.extract_pyg_edges(G)
    assert edge_index.dtype == torch.int64
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == G.number_of_edges()
    assert edge_attr.dtype == torch.float32
    assert edge_attr.shape == (edge_index.shape[1], 1)


def test_builder_static_node_features_shape_and_dtype():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    G = builder.build_networkx_graph()
    feats = builder.compute_topological_node_features(G)
    assert feats.dtype == torch.float32
    assert feats.shape == (G.number_of_nodes(), 4)


def test_builder_full_build_consistency():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    builder.build()
    assert builder.num_nodes == len(TARGET_STATION_EVAS)
    assert builder.static_features.shape == (len(TARGET_STATION_EVAS), 4)
    assert builder.static_features.dtype == torch.float32
    assert builder.edge_index.dtype == torch.int64
    assert builder.edge_index.shape[0] == 2
    assert builder.edge_attr.dtype == torch.float32
    assert builder.edge_attr.shape[1] == 1


# ---------------------------------------------------------------------------
# RailwayGraphDataset tests
# ---------------------------------------------------------------------------

def test_dataset_instantiation():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    builder.build()
    ds = RailwayGraphDataset(NPZ_PATH, builder)
    assert len(ds) > 0


def test_getitem_returns_valid_data_object():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    builder.build()
    ds = RailwayGraphDataset(NPZ_PATH, builder)
    data = ds[0]
    assert isinstance(data, Data)
    assert data.x.dtype == torch.float32
    assert data.x.shape[0] == ds.num_nodes
    assert data.x.shape[1] == 4 + 12 * 7  # static 4 + temporal 84
    assert data.edge_index.dtype == torch.int64
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.dtype == torch.float32
    assert data.edge_attr.shape == (data.edge_index.shape[1], 1)
    assert data.y.dtype == torch.float32
    assert data.y.shape == (ds.num_nodes, 3)
    assert isinstance(data.snapshot_idx, int)


def test_dataset_len_matches_snapshots():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    builder.build()
    ds = RailwayGraphDataset(NPZ_PATH, builder)
    data = np.load(NPZ_PATH, allow_pickle=False)
    n_rows = data["X"].shape[0]
    assert len(ds) == n_rows


# ---------------------------------------------------------------------------
# create_pyg_dataloaders tests
# ---------------------------------------------------------------------------

def test_dataloaders_batch_structure():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    builder.build()
    ds = RailwayGraphDataset(NPZ_PATH, builder)
    train_loader, val_loader, test_loader = create_pyg_dataloaders(ds, batch_size=8)

    for loader in (train_loader, val_loader, test_loader):
        batch = next(iter(loader))
        assert hasattr(batch, "x")
        assert hasattr(batch, "edge_index")
        assert hasattr(batch, "batch")
        assert batch.x.dtype == torch.float32
        assert batch.x.shape[1] == 4 + 12 * 7
        assert batch.edge_index.dtype == torch.int64
        assert batch.edge_index.shape[0] == 2
        assert batch.y.dtype == torch.float32
        assert batch.y.shape[1] == 3


def test_chronological_split_no_overlap():
    builder = RailwayGraphBuilder(target_evas=TARGET_STATION_EVAS)
    builder.build()
    ds = RailwayGraphDataset(NPZ_PATH, builder)
    n = len(ds)
    train_end = int(n * 0.70)
    val_end = int(n * (0.70 + 0.15))

    train_idx = set(range(0, train_end))
    val_idx = set(range(train_end, val_end))
    test_idx = set(range(val_end, n))

    assert train_idx & val_idx == set()
    assert train_idx & test_idx == set()
    assert val_idx & test_idx == set()
    assert train_idx | val_idx | test_idx == set(range(n))
