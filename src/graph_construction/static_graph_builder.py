"""Static railway topology graph construction for RailPulse AI.
Supports both the core 10-station corridor and the Phase 10 expanded 50+ regional network.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Complete WGS84 coordinates & metadata registry for all 52 regional stations
EXPANDED_STATION_REGISTRY: Dict[int, Dict] = {
    # --- Hesse ---
    8000105: {"name": "Frankfurt(Main)Hbf", "lat": 50.1073, "lon": 8.6632},
    8000107: {"name": "Frankfurt(Main)Süd", "lat": 50.0994, "lon": 8.6758},
    8070003: {"name": "Frankfurt Flughafen Fernbf", "lat": 50.0531, "lon": 8.5707},
    8000150: {"name": "Hanau Hbf", "lat": 50.1211, "lon": 8.9296},
    8000068: {"name": "Darmstadt Hbf", "lat": 49.8725, "lon": 8.6313},
    8000250: {"name": "Wiesbaden Hbf", "lat": 50.0718, "lon": 8.2446},
    8000370: {"name": "Wiesbaden Hbf", "lat": 50.0718, "lon": 8.2446},
    8000124: {"name": "Gießen Hbf", "lat": 50.5793, "lon": 8.6644},
    8000130: {"name": "Fulda", "lat": 50.5546, "lon": 9.6835},
    8000260: {"name": "Marburg(Lahn)", "lat": 50.8197, "lon": 8.7753},
    8000199: {"name": "Kassel Hbf", "lat": 51.3183, "lon": 9.4908},
    8000198: {"name": "Kassel-Wilhelmshöhe", "lat": 51.3129, "lon": 9.4475},
    8000200: {"name": "Offenbach(Main)Hbf", "lat": 50.0999, "lon": 8.7618},
    8000318: {"name": "Rüsselsheim", "lat": 49.9934, "lon": 8.4144},
    8000115: {"name": "Friedberg(Hess)", "lat": 50.3341, "lon": 8.7619},
    8000392: {"name": "Wetzlar", "lat": 50.5654, "lon": 8.5042},
    8000228: {"name": "Limburg(Lahn)", "lat": 50.3831, "lon": 8.0617},
    8000024: {"name": "Bad Homburg", "lat": 50.2227, "lon": 8.6212},
    8000036: {"name": "Bensheim", "lat": 49.6826, "lon": 8.6148},
    8000157: {"name": "Heppenheim(Bergstr)", "lat": 49.6416, "lon": 8.6378},
    8000185: {"name": "Korbach Hbf", "lat": 51.2721, "lon": 8.8779},

    # --- Rhineland-Palatinate ---
    8000240: {"name": "Mainz Hbf", "lat": 49.9975, "lon": 8.2592},
    8000257: {"name": "Worms Hbf", "lat": 49.6347, "lon": 8.3568},
    8000206: {"name": "Koblenz Hbf", "lat": 50.3503, "lon": 7.5886},
    8000236: {"name": "Ludwigshafen(Rh)Hbf", "lat": 49.4772, "lon": 8.4319},
    8000183: {"name": "Kaiserslautern Hbf", "lat": 49.4357, "lon": 7.7689},
    8000344: {"name": "Speyer Hbf", "lat": 49.3243, "lon": 8.4278},
    8000276: {"name": "Neustadt(Weinstr)Hbf", "lat": 49.3506, "lon": 8.1408},
    8000216: {"name": "Landau(Pfalz)Hbf", "lat": 49.1983, "lon": 8.1256},
    8000340: {"name": "Saarbrücken Hbf", "lat": 49.2411, "lon": 6.9910},
    8000349: {"name": "Trier Hbf", "lat": 49.7569, "lon": 6.6528},
    8000025: {"name": "Bad Kreuznach", "lat": 49.8407, "lon": 7.8682},
    8000033: {"name": "Bingen(Rhein)Hbf", "lat": 49.9678, "lon": 7.8925},
    8000004: {"name": "Andernach", "lat": 50.4353, "lon": 7.4042},
    8000041: {"name": "Boppard Hbf", "lat": 50.2289, "lon": 7.5867},
    8000114: {"name": "Frankenthal Hbf", "lat": 49.5342, "lon": 8.3524},
    8000128: {"name": "Germersheim", "lat": 49.2178, "lon": 8.3694},
    8000171: {"name": "Homburg(Saar)Hbf", "lat": 49.3275, "lon": 7.3361},
    8000305: {"name": "Pirmasens Hbf", "lat": 49.2064, "lon": 7.6047},

    # --- Baden-Württemberg & Intercity Connectors ---
    8000244: {"name": "Mannheim Hbf", "lat": 49.4793, "lon": 8.4700},
    8000191: {"name": "Karlsruhe Hbf", "lat": 48.9937, "lon": 8.4017},
    8000152: {"name": "Heidelberg Hbf", "lat": 49.4035, "lon": 8.6756},
    8000096: {"name": "Stuttgart Hbf", "lat": 48.7841, "lon": 9.1816},
    8000142: {"name": "Heilbronn Hbf", "lat": 49.1428, "lon": 9.2081},
    8000309: {"name": "Pforzheim Hbf", "lat": 48.8928, "lon": 8.7031},
    8000050: {"name": "Bruchsal", "lat": 49.1247, "lon": 8.5878},
    8000326: {"name": "Schwetzingen", "lat": 49.3878, "lon": 8.5722},
    8000384: {"name": "Weinheim(Bergstr)Hbf", "lat": 49.5528, "lon": 8.6653},
    8000013: {"name": "Aschaffenburg Hbf", "lat": 49.9803, "lon": 9.1434},
    8000390: {"name": "Würzburg Hbf", "lat": 49.8017, "lon": 9.9353},
    8000207: {"name": "Köln Hbf", "lat": 50.9430, "lon": 6.9589},
    8000085: {"name": "Düsseldorf Hbf", "lat": 51.2201, "lon": 6.7942},
    8000261: {"name": "München Hbf", "lat": 48.1402, "lon": 11.5583},
}

# 10 Core Corridor Stations (legacy mapping preserved)
LEGACY_10_COORDS = {
    8000105: (50.1073, 8.6632),
    8000244: (49.4793, 8.4700),
    8000240: (49.9975, 8.2592),
    8000207: (50.9430, 6.9589),
    8000370: (50.0718, 8.2446),
    8000130: (50.5546, 9.6835),
    8000107: (50.0994, 8.6758),
    8000191: (48.9937, 8.4017),
    8000096: (51.2201, 6.7942),
    8000390: (49.8017, 9.9353),
}


class RailwayGraphBuilder:
    """Constructs static networkx and PyTorch Geometric railway graph topologies."""

    def __init__(self, target_evas: Optional[List[int]] = None):
        from src.config import TARGET_STATION_EVAS

        self.target_evas = list(target_evas) if target_evas is not None else list(TARGET_STATION_EVAS)
        self.eva_to_idx: Dict[int, int] = {eva: i for i, eva in enumerate(self.target_evas)}
        self.idx_to_eva: Dict[int, int] = {i: eva for i, eva in enumerate(self.target_evas)}
        self.num_nodes = len(self.target_evas)

        # Populate coordinates from expanded registry with fallback
        self.coords: Dict[int, Tuple[float, float]] = {}
        for eva in self.target_evas:
            if eva in EXPANDED_STATION_REGISTRY:
                entry = EXPANDED_STATION_REGISTRY[eva]
                self.coords[eva] = (entry["lat"], entry["lon"])
            elif eva in LEGACY_10_COORDS:
                self.coords[eva] = LEGACY_10_COORDS[eva]
            else:
                self.coords[eva] = (50.0, 8.5)

        self.graph: Optional[nx.DiGraph] = None
        self.static_features: Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_attr: Optional[torch.Tensor] = None

    def _load_coords_from_gtfs(self, gtfs_path: Optional[Path]) -> None:
        """Optional GTFS coordinates loader."""
        if gtfs_path is None or not Path(gtfs_path).exists():
            return
        try:
            import zipfile
            import pandas as pd

            with zipfile.ZipFile(gtfs_path) as z:
                if "stops.txt" in z.namelist():
                    stops = pd.read_csv(z.open("stops.txt"), dtype=str, low_memory=False)
                    # Coordinates could be matched here if needed
        except Exception as e:
            logger.warning("Could not parse GTFS coords: %s", e)

    def build_networkx_graph(self, gtfs_path: Optional[Path] = None) -> nx.DiGraph:
        """Build directed topology with realistic track connections between stations."""
        self._load_coords_from_gtfs(gtfs_path)
        G = nx.DiGraph()

        for eva in self.target_evas:
            G.add_node(self.eva_to_idx[eva], eva_id=eva)

        if self.num_nodes <= 10:
            # Preserved legacy 10-station topology for backward test compatibility
            for i in range(self.num_nodes - 1):
                G.add_edge(i, i + 1, weight=30.0)
                G.add_edge(i + 1, i, weight=30.0)
            for j in range(1, self.num_nodes):
                if not G.has_edge(0, j):
                    G.add_edge(0, j, weight=60.0)
                if not G.has_edge(j, 0):
                    G.add_edge(j, 0, weight=60.0)
        else:
            # Expanded 50+ Regional Topology: Connect based on spatial geographic proximity
            coords_arr = np.array([self.coords[self.idx_to_eva[i]] for i in range(self.num_nodes)])
            # Connect each station to its 4 nearest neighbors bidirectionally
            for i in range(self.num_nodes):
                lat1, lon1 = coords_arr[i]
                dists = np.hypot(coords_arr[:, 0] - lat1, coords_arr[:, 1] - lon1)
                dists[i] = 1e9  # Exclude self
                nearest = np.argsort(dists)[:4]
                for n_idx in nearest:
                    weight_km = float(dists[n_idx] * 111.0)
                    G.add_edge(i, int(n_idx), weight=max(10.0, weight_km))
                    G.add_edge(int(n_idx), i, weight=max(10.0, weight_km))

            # Guarantee Frankfurt Hbf (8000105) acts as core central hub to regional hubs
            if 8000105 in self.eva_to_idx:
                ffm_idx = self.eva_to_idx[8000105]
                # Express connections to major cities
                hubs = [8000244, 8000240, 8000130, 8000250, 8000150, 8000068, 8000191, 8000207]
                for hub_eva in hubs:
                    if hub_eva in self.eva_to_idx:
                        h_idx = self.eva_to_idx[hub_eva]
                        G.add_edge(ffm_idx, h_idx, weight=45.0)
                        G.add_edge(h_idx, ffm_idx, weight=45.0)

        self.graph = G
        return G

    def compute_topological_node_features(self, G: nx.DiGraph) -> torch.Tensor:
        """Compute spatial and centrality features: [lat, lon, deg_cent, bet_cent]."""
        n = G.number_of_nodes()
        try:
            deg_cent = nx.degree_centrality(G)
        except Exception:
            deg_cent = {i: 0.0 for i in range(n)}
        try:
            bet_cent = nx.betweenness_centrality(G)
        except Exception:
            bet_cent = {i: 0.0 for i in range(n)}

        lats = np.zeros(n, dtype=np.float32)
        lons = np.zeros(n, dtype=np.float32)
        for i in range(n):
            eva = self.idx_to_eva[i]
            if eva in self.coords:
                lats[i], lons[i] = self.coords[eva]

        if lats.max() - lats.min() > 0:
            lats = (lats - lats.mean()) / (lats.std() + 1e-6)
        if lons.max() - lons.min() > 0:
            lons = (lons - lons.mean()) / (lons.std() + 1e-6)

        feats = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            feats[i, 0] = lats[i]
            feats[i, 1] = lons[i]
            feats[i, 2] = float(deg_cent.get(i, 0.0))
            feats[i, 3] = float(bet_cent.get(i, 0.0))

        self.static_features = torch.from_numpy(feats).to(torch.float32)
        return self.static_features

    def extract_pyg_edges(self, G: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge_index and edge_attr from NetworkX graph."""
        edges = list(G.edges(data=True))
        if not edges:
            ei = torch.zeros((2, 0), dtype=torch.long)
            ea = torch.zeros((0, 1), dtype=torch.float32)
            self.edge_index = ei
            self.edge_attr = ea
            return ei, ea

        src = [u for u, v, _ in edges]
        dst = [v for u, v, _ in edges]
        weights = [float(d.get("weight", 1.0)) for u, v, d in edges]

        ei = torch.tensor([src, dst], dtype=torch.long)
        ea = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

        self.edge_index = ei
        self.edge_attr = ea
        return ei, ea

    def build(self, gtfs_path: Optional[Path] = None) -> "RailwayGraphBuilder":
        """Executes full graph construction pipeline."""
        G = self.build_networkx_graph(gtfs_path)
        self.compute_topological_node_features(G)
        self.extract_pyg_edges(G)
        return self
