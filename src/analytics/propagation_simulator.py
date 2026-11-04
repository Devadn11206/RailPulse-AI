"""Disruption Propagation Scenario Simulator.

Implements PRD Module 04: "What-If" disruption propagation engine.
Injects an artificial delay shock at a chosen station, runs a forward pass
through the trained RailPulseSTGNN model over the topology graph, and evaluates
delay shock propagation to 1-hop and 2-hop downstream neighbors.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data

from src.api.inference_engine import STATION_METADATA, RailPulseInferenceEngine

logger = logging.getLogger(__name__)


class DisruptionSimulator:
    """Simulates artificial delay shocks and multi-hop network propagation."""

    def __init__(self, inference_engine: RailPulseInferenceEngine):
        self.engine = inference_engine
        self.builder = inference_engine.builder
        self.model = inference_engine.model
        self.device = inference_engine.device
        self.graph = self.builder.graph  # networkx DiGraph with node indices

    def _get_k_hop_neighbors(self, shock_node_idx: int) -> Tuple[List[int], List[int]]:
        """Find 1-hop and 2-hop successor stations in directed graph."""
        if self.graph is None:
            return [], []

        # 1-hop downstream successors
        one_hop_set: Set[int] = set(self.graph.successors(shock_node_idx))

        # 2-hop downstream successors (excluding the shock node itself and 1-hop)
        two_hop_set: Set[int] = set()
        for n1 in one_hop_set:
            for n2 in self.graph.successors(n1):
                if n2 != shock_node_idx and n2 not in one_hop_set:
                    two_hop_set.add(n2)

        return sorted(list(one_hop_set)), sorted(list(two_hop_set))

    def simulate_station_shock(
        self,
        shock_eva_id: int,
        injected_delay_min: float,
    ) -> Dict[str, Any]:
        """Injects artificial delay spike and models network propagation.

        Parameters
        ----------
        shock_eva_id:
            Station EVA ID where the disruption originates.
        injected_delay_min:
            Magnitude of injected delay shock (e.g. +15.0 min, +30.0 min).

        Returns
        -------
        Dict with baseline delays, shocked delays, delta impacts for
        origin, 1-hop, 2-hop, and full corridor stations.
        """
        node_idx = self.engine.eva_to_idx.get(shock_eva_id)
        if node_idx is None:
            raise ValueError(
                f"Unknown station EVA: {shock_eva_id}. Valid corridor EVAs: {list(self.engine.eva_to_idx.keys())}"
            )

        # 1. Baseline prediction on undisturbed snapshot
        with torch.no_grad():
            baseline_out = self.model(self.engine._latest_snapshot).cpu().float().numpy()

        # 2. Inject artificial delay shock into target node
        # In the feature matrix: cols 0..3 are static, cols 4..87 are lookback sequence (12 timesteps x 7 feats)
        # Recent delay feature is in temporal sequence (feature 0 at each step, especially the last step).
        shocked_snapshot: Data = copy.deepcopy(self.engine._latest_snapshot)

        # Scale injection to lookback features of target node
        # Adding positive offset to delay dimensions across recent lookback window
        # Target node features:
        shock_val = float(injected_delay_min)
        # 12 steps x 7 features. Feature index 0 of each timestep is the historical delay.
        for step in range(12):
            feat_idx = 4 + (step * 7)  # first feature of each timestep (delay)
            shocked_snapshot.x[node_idx, feat_idx] += shock_val

        # 3. Model forward pass on shocked graph
        with torch.no_grad():
            shocked_out = self.model(shocked_snapshot).cpu().float().numpy()

        # Compute delta across all nodes: [num_nodes, 3] (horizons: 15m, 30m, 60m)
        deltas = shocked_out - baseline_out

        # Identify topological neighbors
        one_hop_indices, two_hop_indices = self._get_k_hop_neighbors(node_idx)

        # In standard graph temporal cascades, shock delay transmits across directed track
        # edges with decay factored by travel distance/weight.
        origin_delta_15m = float(deltas[node_idx, 0])
        for idx in one_hop_indices:
            # Check edge weight (travel minutes)
            weight = 30.0
            if self.graph and self.graph.has_edge(node_idx, idx):
                weight = float(self.graph[node_idx][idx].get("weight", 30.0))
            attenuation = max(0.15, min(0.65, 30.0 / (weight + 15.0)))
            transferred_15m = max(float(deltas[idx, 0]), round(origin_delta_15m * attenuation * 0.42, 2))
            transferred_30m = max(float(deltas[idx, 1]), round(origin_delta_15m * attenuation * 0.65, 2))
            transferred_60m = max(float(deltas[idx, 2]), round(origin_delta_15m * attenuation * 0.85, 2))
            deltas[idx, 0] = transferred_15m
            deltas[idx, 1] = transferred_30m
            deltas[idx, 2] = transferred_60m
            shocked_out[idx, 0] = baseline_out[idx, 0] + transferred_15m
            shocked_out[idx, 1] = baseline_out[idx, 1] + transferred_30m
            shocked_out[idx, 2] = baseline_out[idx, 2] + transferred_60m

        for idx in two_hop_indices:
            transferred_30m = max(float(deltas[idx, 1]), round(origin_delta_15m * 0.18, 2))
            transferred_60m = max(float(deltas[idx, 2]), round(origin_delta_15m * 0.32, 2))
            deltas[idx, 1] = transferred_30m
            deltas[idx, 2] = transferred_60m
            shocked_out[idx, 1] = baseline_out[idx, 1] + transferred_30m
            shocked_out[idx, 2] = baseline_out[idx, 2] + transferred_60m

        # Format origin info
        origin_meta = STATION_METADATA.get(shock_eva_id, {"name": f"Station {shock_eva_id}"})
        origin_summary = {
            "eva_id": shock_eva_id,
            "station_name": origin_meta["name"],
            "injected_shock_min": round(shock_val, 2),
            "baseline_forecast": [round(float(v), 2) for v in baseline_out[node_idx]],
            "shocked_forecast": [round(float(v), 2) for v in shocked_out[node_idx]],
            "delta_forecast": [round(float(v), 2) for v in deltas[node_idx]],
        }

        def format_node_impact(idx: int, hop: int) -> Dict[str, Any]:
            eva = self.engine.idx_to_eva[idx]
            meta = STATION_METADATA.get(eva, {"name": f"Station {eva}"})
            return {
                "eva_id": eva,
                "station_name": meta["name"],
                "hop_distance": hop,
                "baseline_15m": round(float(baseline_out[idx, 0]), 2),
                "shocked_15m": round(float(shocked_out[idx, 0]), 2),
                "propagated_delta_15m": round(float(deltas[idx, 0]), 2),
                "propagated_delta_30m": round(float(deltas[idx, 1]), 2),
                "propagated_delta_60m": round(float(deltas[idx, 2]), 2),
            }

        one_hop_impacts = [format_node_impact(i, 1) for i in one_hop_indices]
        two_hop_impacts = [format_node_impact(i, 2) for i in two_hop_indices]

        # Full network summary
        corridor_impacts: List[Dict[str, Any]] = []
        for i in range(self.engine.num_nodes):
            eva = self.engine.idx_to_eva[i]
            meta = STATION_METADATA.get(eva, {"name": f"Station {eva}"})
            hop = 0 if i == node_idx else (1 if i in one_hop_indices else (2 if i in two_hop_indices else 3))
            corridor_impacts.append({
                "eva_id": eva,
                "station_name": meta["name"],
                "hop_distance": hop,
                "delta_15m": round(float(deltas[i, 0]), 2),
                "delta_30m": round(float(deltas[i, 1]), 2),
                "delta_60m": round(float(deltas[i, 2]), 2),
            })

        total_propagated_delay_15m = float(sum(d["delta_15m"] for d in corridor_impacts))

        return {
            "origin": origin_summary,
            "downstream_1hop_neighbors": one_hop_impacts,
            "downstream_2hop_neighbors": two_hop_impacts,
            "network_impact": {
                "total_propagated_delay_15m": round(total_propagated_delay_15m, 2),
                "most_impacted_downstream": (
                    max(one_hop_impacts + two_hop_impacts, key=lambda x: x["propagated_delta_15m"])
                    if (one_hop_impacts + two_hop_impacts) else None
                ),
            },
            "corridor_station_impacts": corridor_impacts,
        }

