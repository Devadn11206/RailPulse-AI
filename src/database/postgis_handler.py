"""PostGIS spatial interface for RailPulse AI.

Provides GeoJSON FeatureCollections for station points and track line
geometries.  The current implementation is a **mock** that derives all
spatial data from the in-memory ``RailwayGraphBuilder`` topology — no
real database connection is required.

Replace the ``_fetch_*`` helper methods with asyncpg / psycopg2 queries
in a future phase when a live PostGIS instance is available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project root importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.api.inference_engine import STATION_METADATA
from src.config import TARGET_STATION_EVAS
from src.graph_construction.static_graph_builder import RailwayGraphBuilder


class PostGISHandler:
    """Mock spatial data handler returning GeoJSON FeatureCollections.

    Parameters
    ----------
    builder:
        Optional pre-built ``RailwayGraphBuilder`` instance.  If not
        provided a new one is built from the default corridor config.
    """

    def __init__(self, builder: Optional[RailwayGraphBuilder] = None) -> None:
        if builder is None:
            self._builder = RailwayGraphBuilder(target_evas=list(TARGET_STATION_EVAS))
            self._builder.build()
        else:
            self._builder = builder
            if self._builder.static_features is None:
                self._builder.build()

    # ------------------------------------------------------------------
    # Station point GeoJSON
    # ------------------------------------------------------------------

    def get_station_geojson(
        self,
        delay_overlay: Optional[Dict[int, float]] = None,
    ) -> Dict:
        """Return a GeoJSON FeatureCollection of station Point features.

        Parameters
        ----------
        delay_overlay:
            Optional mapping of ``{eva_id: delay_minutes}`` to attach as
            a ``current_delay`` property on each feature.

        Returns
        -------
        dict:
            GeoJSON FeatureCollection with one Point feature per station.
        """
        features: List[Dict] = []
        for eva in TARGET_STATION_EVAS:
            meta = STATION_METADATA.get(
                eva,
                {"name": f"Station {eva}", "lat": 0.0, "lon": 0.0},
            )
            props: Dict = {
                "eva_id": eva,
                "name": meta["name"],
            }
            if delay_overlay and eva in delay_overlay:
                props["current_delay"] = float(delay_overlay[eva])

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [meta["lon"], meta["lat"]],  # GeoJSON: [lon, lat]
                    },
                    "properties": props,
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    # ------------------------------------------------------------------
    # Track LineString GeoJSON
    # ------------------------------------------------------------------

    def get_track_geojson(self) -> Dict:
        """Return a GeoJSON FeatureCollection of directed track LineString features.

        Each feature represents one directed edge in the station graph.
        Returns
        -------
        dict:
            GeoJSON FeatureCollection with one LineString feature per edge.
        """
        features: List[Dict] = []

        edge_index = self._builder.edge_index
        edge_attr = self._builder.edge_attr
        idx_to_eva = self._builder.idx_to_eva

        if edge_index is None:
            return {"type": "FeatureCollection", "features": []}

        ei = edge_index.cpu().numpy()
        ea = (
            edge_attr.cpu().numpy().flatten()
            if edge_attr is not None
            else [1.0] * ei.shape[1]
        )

        for k in range(ei.shape[1]):
            src_eva = idx_to_eva[int(ei[0, k])]
            dst_eva = idx_to_eva[int(ei[1, k])]
            src_meta = STATION_METADATA.get(src_eva, {"lat": 0.0, "lon": 0.0})
            dst_meta = STATION_METADATA.get(dst_eva, {"lat": 0.0, "lon": 0.0})

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [src_meta["lon"], src_meta["lat"]],
                            [dst_meta["lon"], dst_meta["lat"]],
                        ],
                    },
                    "properties": {
                        "source_eva": src_eva,
                        "target_eva": dst_eva,
                        "weight_minutes": float(ea[k]),
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    # ------------------------------------------------------------------
    # Delay heatmap GeoJSON (convenience wrapper)
    # ------------------------------------------------------------------

    def get_delay_heatmap_geojson(
        self,
        delay_map: Dict[int, float],
    ) -> Dict:
        """Return station GeoJSON enriched with delay values for heatmap rendering.

        Parameters
        ----------
        delay_map:
            Mapping ``{eva_id: delay_minutes}``.
        """
        return self.get_station_geojson(delay_overlay=delay_map)

