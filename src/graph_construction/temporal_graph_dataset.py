"""Dynamic spatio-temporal PyTorch Geometric dataset for RailPulse AI."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.graph_construction.static_graph_builder import RailwayGraphBuilder

logger = logging.getLogger(__name__)


class RailwayGraphDataset(Dataset):
    def __init__(self, npz_path: Path, static_graph_builder: RailwayGraphBuilder):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=False)
        self.X = data["X"].astype(np.float32)  # [N, 12, 7]
        self.Y = data["Y"].astype(np.float32)  # [N, 3]

        # Optional metadata
        if "node_id" in data.files:
            self.node_ids = data["node_id"].astype(np.int64)
        elif "station_id" in data.files:
            self.node_ids = data["station_id"].astype(np.int64)
        else:
            self.node_ids = np.arange(len(self.X), dtype=np.int64) % static_graph_builder.num_nodes

        if "snapshot_id" in data.files:
            self.snapshot_ids = data["snapshot_id"]
        elif "timestamp" in data.files:
            self.snapshot_ids = data["timestamp"]
        else:
            self.snapshot_ids = np.arange(len(self.X))

        self.builder = static_graph_builder
        if self.builder.static_features is None:
            self.builder.build()
        self.static_features = self.builder.static_features  # [Num_Nodes, 4]
        self.edge_index = self.builder.edge_index
        self.edge_attr = self.builder.edge_attr
        self.num_nodes = self.builder.num_nodes

        # Map snapshots to row indices
        self._snapshots = list(np.unique(self.snapshot_ids))
        self._snapshot_to_rows = {}
        for snap in self._snapshots:
            self._snapshot_to_rows[snap] = np.where(self.snapshot_ids == snap)[0]

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, idx: int) -> Data:
        snap = self._snapshots[idx]
        rows = self._snapshot_to_rows[snap]

        # Build node feature matrix
        x = torch.zeros((self.num_nodes, 12 * 7 + 4), dtype=torch.float32)
        # static features for all nodes
        x[:, :4] = self.static_features
        y = torch.zeros((self.num_nodes, 3), dtype=torch.float32)

        for r in rows:
            node = int(self.node_ids[r]) % self.num_nodes
            temporal = self.X[r].reshape(-1)  # [84]
            x[node, 4:] = torch.from_numpy(temporal).to(torch.float32)
            y[node] = torch.from_numpy(self.Y[r]).to(torch.float32)

        data = Data(
            x=x,
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone(),
            y=y,
        )
        data.snapshot_idx = int(idx)
        return data


def create_pyg_dataloaders(
    dataset: RailwayGraphDataset,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader, test_loader
