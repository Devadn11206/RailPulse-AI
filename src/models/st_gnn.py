"""Spatio-Temporal Graph Neural Network for RailPulse AI.

Phases 4 & 11: Hybrid architecture combining PyTorch Geometric GATv2Conv
(multi-head dynamic spatial attention) over station topology with a GRU temporal
encoder for multi-horizon delay forecasting. Designed for edge GPU execution
with bounded hidden dims (<=64), heads=4, GRU layers (<=2), and VRAM < 1.5 GB.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv

logger = logging.getLogger(__name__)


class RailPulseSTGNN(nn.Module):
    """Hybrid spatio-temporal GNN with GATv2 dynamic spatial attention.

    Architecture:
      - Spatial Module: 2-layer GATv2Conv aggregating node features with dynamic
        multi-head attention weights conditioned on real-time delay states.
      - Temporal Module: 2-layer nn.GRU processing the 12-step lookback
        sequence per station node.
      - Prediction Head: MLP (Linear -> ReLU -> Dropout -> Linear)
        producing multi-horizon outputs [Num_Nodes, out_horizons].
    """

    def __init__(
        self,
        node_in_dim: int = 11,
        hidden_dim: int = 64,
        num_layers: int = 2,
        out_horizons: int = 3,
        dropout: float = 0.2,
        heads: int = 4,
        use_gatv2: bool = True,
    ):
        super().__init__()
        self.node_in_dim = int(node_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_horizons = int(out_horizons)
        self.dropout = float(dropout)
        self.heads = int(heads)
        self.use_gatv2 = bool(use_gatv2)
        self.temporal_in_dim = 7  # features per timestep in the lookback window
        self.lookback_steps = 12  # T=12
        self.temporal_flat_dim = self.lookback_steps * self.temporal_in_dim  # 84

        # ------------------------------------------------------------------
        # Spatial Module: Multi-Head GATv2 dynamic attention
        # ------------------------------------------------------------------
        if self.use_gatv2:
            self.gat1 = GATv2Conv(
                in_channels=self.node_in_dim,
                out_channels=self.hidden_dim,
                heads=self.heads,
                concat=False,
                dropout=self.dropout,
                add_self_loops=False,
            )
            self.gat2 = GATv2Conv(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                heads=self.heads,
                concat=False,
                dropout=self.dropout,
                add_self_loops=False,
            )
        else:
            # Fallback legacy GCNConv for strict backwards compatibility
            self.gcn1 = GCNConv(self.node_in_dim, self.hidden_dim)
            self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)

        # ------------------------------------------------------------------
        # Temporal Module: GRU over the 12-step lookback sequence
        # ------------------------------------------------------------------
        gru_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.temporal_in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        # ------------------------------------------------------------------
        # Prediction Head: MLP producing multi-horizon delay outputs
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.out_horizons),
        )

    def _split_static_temporal(self, x: torch.Tensor) -> tuple:
        """Split data.x into static node features and temporal sequence.

        Returns:
            static: [N, node_in_dim] static node features
            temporal_seq: [N, 12, 7] temporal sequence per node
        """
        N = x.shape[0]
        total_feat = x.shape[1]

        if total_feat >= self.node_in_dim + self.temporal_flat_dim:
            static_dim = self.node_in_dim
        elif total_feat > self.temporal_flat_dim:
            static_dim = total_feat - self.temporal_flat_dim
        else:
            static_dim = total_feat

        static = x[:, :static_dim]
        temporal_flat = x[:, static_dim : static_dim + self.temporal_flat_dim]

        if static_dim < self.node_in_dim:
            pad = torch.zeros(
                N, self.node_in_dim - static_dim, device=x.device, dtype=x.dtype
            )
            static = torch.cat([static, pad], dim=1)
        elif static_dim > self.node_in_dim:
            static = static[:, : self.node_in_dim]

        if temporal_flat.shape[1] >= self.temporal_flat_dim:
            temporal_seq = temporal_flat[:, : self.temporal_flat_dim].view(
                N, self.lookback_steps, self.temporal_in_dim
            )
        else:
            if temporal_flat.shape[1] > 0:
                n_steps = temporal_flat.shape[1] // self.temporal_in_dim
                temporal_seq = temporal_flat[:, : n_steps * self.temporal_in_dim].view(
                    N, n_steps, self.temporal_in_dim
                )
                if n_steps < self.lookback_steps:
                    pad_steps = self.lookback_steps - n_steps
                    pad = torch.zeros(
                        N, pad_steps, self.temporal_in_dim,
                        device=x.device, dtype=x.dtype,
                    )
                    temporal_seq = torch.cat([temporal_seq, pad], dim=1)
            else:
                temporal_seq = torch.zeros(
                    N, self.lookback_steps, self.temporal_in_dim,
                    device=x.device, dtype=x.dtype,
                )

        return static, temporal_seq

    def forward(
        self,
        data: Data,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass.

        Args:
            data: PyG Data/Batch object with x, edge_index, edge_attr.
            return_attention_weights: Whether to return dynamic attention weights (alpha_ij).

        Returns:
            If return_attention_weights is False:
                output: [total_nodes, out_horizons] (float32).
            If return_attention_weights is True:
                (output, (edge_index, attention_weights))
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        N = x.shape[0]
        static, temporal_seq = self._split_static_temporal(x)

        # ------------------------------------------------------------------
        # Spatial path: GATv2 dynamic message passing
        # ------------------------------------------------------------------
        attn_tuple = None
        if hasattr(self, "gat1"):
            h1 = F.relu(self.gat1(static, edge_index))
            h1 = F.dropout(h1, p=self.dropout, training=self.training)

            if return_attention_weights:
                h2, (edge_index_attn, alpha) = self.gat2(
                    h1, edge_index, return_attention_weights=True
                )
                attn_tuple = (edge_index_attn, alpha)
            else:
                h2 = self.gat2(h1, edge_index)

            h_spatial = F.relu(h2)
            h_spatial = F.dropout(h_spatial, p=self.dropout, training=self.training)
        else:
            edge_weight = edge_attr.view(-1) if edge_attr is not None else None
            h_spatial = F.relu(self.gcn1(static, edge_index, edge_weight))
            h_spatial = F.dropout(h_spatial, p=self.dropout, training=self.training)
            h_spatial = F.relu(self.gcn2(h_spatial, edge_index, edge_weight))
            h_spatial = F.dropout(h_spatial, p=self.dropout, training=self.training)

        # ------------------------------------------------------------------
        # Temporal path: GRU over lookback sequence
        # ------------------------------------------------------------------
        _, h_n = self.gru(temporal_seq)
        h_temporal = h_n[-1]

        # ------------------------------------------------------------------
        # Combine spatial + temporal and predict
        # ------------------------------------------------------------------
        combined = torch.cat([h_spatial, h_temporal], dim=1)
        output = self.head(combined)

        if return_attention_weights and attn_tuple is not None:
            return output, attn_tuple
        return output

    @staticmethod
    def create_synthetic_data(num_nodes: int = 10, num_edges: int = 20) -> Data:
        """Create a synthetic PyG Data object for testing."""
        x = torch.randn(num_nodes, 88, dtype=torch.float32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
        edge_attr = torch.rand(num_edges, 1, dtype=torch.float32) * 60.0
        y = torch.randn(num_nodes, 3, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
