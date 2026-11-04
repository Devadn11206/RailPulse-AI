"""GTFS static feed ingestion for RailPulse AI."""

import os
import zipfile
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

_MOCK_STATIONS = {
    8000105: ("Frankfurt (Main) Hbf", 50.1071, 8.6638),
    8000244: ("Mannheim Hbf", 49.4794, 8.4699),
    8000240: ("Mainz Hbf", 50.0010, 8.2590),
    8000207: ("Köln Hbf", 50.9431, 6.9588),
    8000370: ("Wiesbaden Hbf", 50.0826, 8.2436),
    8000130: ("Fulda", 50.5547, 9.6843),
    8000107: ("Frankfurt (Main) Süd", 50.0994, 8.6864),
    8000191: ("Karlsruhe Hbf", 48.9938, 8.4020),
    8000096: ("Düsseldorf Hbf", 51.2199, 6.7943),
    8000390: ("Würzburg Hbf", 49.8018, 9.9358),
}
_PREFERRED_STOP_NAMES = {
    8000105: ["Frankfurt(Main)Hbf", "Frankfurt (Main) Hbf", "Frankfurt Hbf"],
    8000244: ["Mannheim Hbf"],
    8000240: ["Mainz Hbf"],
    8000207: ["Köln Hbf"],
    8000370: ["Wiesbaden Hbf"],
    8000130: ["Fulda Bahnhof", "Fulda Hbf", "Fulda"],
    8000107: ["Frankfurt(Main)Süd", "Frankfurt (Main) Süd", "Frankfurt Süd"],
    8000191: ["Karlsruhe Hbf"],
    8000096: ["Düsseldorf Hbf"],
    8000390: ["Würzburg Hbf"],
}

_NAME_KEYWORDS = {
    8000105: ["Frankfurt", "Hbf"],
    8000244: ["Mannheim", "Hbf"],
    8000240: ["Mainz", "Hbf"],
    8000207: ["Köln", "Hbf"],
    8000370: ["Wiesbaden", "Hbf"],
    8000130: ["Fulda", "Bahnhof"],
    8000107: ["Frankfurt", "Süd"],
    8000191: ["Karlsruhe", "Hbf"],
    8000096: ["Düsseldorf", "Hbf"],
    8000390: ["Würzburg", "Hbf"],
}


class GTFSProcessor:
    def __init__(self, gtfs_dir=None, target_evas=None):
        from src.config import GTFS_DIR, TARGET_STATION_EVAS
        self.gtfs_dir = Path(gtfs_dir) if gtfs_dir is not None else GTFS_DIR
        self.target_evas = list(target_evas) if target_evas is not None else list(TARGET_STATION_EVAS)
        self.stops = None
        self.routes = None
        self.stop_times = None
        self.trips = None
        self._mock = False
        self._stop_id_to_eva = None

    def _find_zip(self):
        zips = sorted(self.gtfs_dir.glob("*.zip"))
        return zips[0] if zips else None

    def load(self):
        zp = self._find_zip()
        if zp is None:
            logger.warning("No GTFS zip found in %s; using mock fallback.", self.gtfs_dir)
            self._load_mock()
            return self
        try:
            with zipfile.ZipFile(zp) as z:
                self.stops = self._read_csv_from_zip(z, "stops.txt")
                self.routes = self._read_csv_from_zip(z, "routes.txt")
                self.stop_times = self._read_csv_from_zip(z, "stop_times.txt")
                try:
                    self.trips = self._read_csv_from_zip(z, "trips.txt")
                except Exception:
                    self.trips = None
            self._mock = False
            self._build_stop_id_to_eva_map()
        except Exception as e:
            logger.warning("Failed to parse GTFS zip (%s); falling back to mock.", e)
            self._load_mock()
        return self

    @staticmethod
    def _read_csv_from_zip(z, name):
        with z.open(name) as f:
            return pd.read_csv(f, dtype=str, low_memory=False)

    def _load_mock(self):
        self._mock = True
        rows = []
        for eva in self.target_evas:
            name, lat, lon = _MOCK_STATIONS.get(eva, (f"Station {eva}", 50.0, 8.0))
            rows.append({"stop_id": str(eva), "stop_name": name, "stop_lat": lat, "stop_lon": lon})
        self.stops = pd.DataFrame(rows)
        self.routes = pd.DataFrame({"route_id": ["R1"], "route_short_name": ["ICE"], "route_long_name": ["Corridor"], "route_type": ["2"]})
        self.stop_times = pd.DataFrame({"trip_id": ["T1"] * len(self.target_evas), "stop_id": [str(e) for e in self.target_evas], "stop_sequence": list(range(1, len(self.target_evas) + 1)), "arrival_time": ["08:00:00"] * len(self.target_evas), "departure_time": ["08:02:00"] * len(self.target_evas)})
        self.trips = pd.DataFrame({"trip_id": ["T1"], "route_id": ["R1"]})
        self._build_stop_id_to_eva_map()

    @property
    def is_mock(self):
        return self._mock

    def _eva_from_stop_id(self, stop_id):
        if stop_id is None:
            return None
        # Use precomputed map if available
        if self._stop_id_to_eva is not None:
            return self._stop_id_to_eva.get(str(stop_id).strip())
        s = str(stop_id).strip()
        try:
            v = int(s)
            if v in self.target_evas:
                return v
        except ValueError:
            pass
        for eva in self.target_evas:
            if s.startswith(str(eva)):
                return eva
        return None

    def _eva_from_name(self, name):
        if name is None:
            return None
        n = str(name).strip()
        # Preferred exact names
        for eva, names in _PREFERRED_STOP_NAMES.items():
            if n in names:
                return eva
        nl = n.lower()
        # Keyword fallback
        for eva, keywords in _NAME_KEYWORDS.items():
            if all(k.lower() in nl for k in keywords):
                return eva
        return None

    def _build_stop_id_to_eva_map(self):
        self._stop_id_to_eva = {}
        if self.stops is None:
            return
        for _, row in self.stops.iterrows():
            sid = str(row.get("stop_id")).strip()
            eva = None
            try:
                v = int(sid)
                if v in self.target_evas:
                    eva = v
            except ValueError:
                pass
            if eva is None:
                eva = self._eva_from_name(row.get("stop_name"))
            if eva is not None and eva in self.target_evas:
                self._stop_id_to_eva[sid] = eva

    def get_target_stops(self):
        if self.stops is None:
            self.load()
        df = self.stops.copy()
        df["_eva"] = df["stop_id"].astype(str).map(self._eva_from_stop_id)
        df = df[df["_eva"].notna()].copy()
        df["eva"] = df["_eva"].astype(int)
        for c in ("stop_lat", "stop_lon"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop_duplicates(subset=["eva"], keep="first")
        return df[["eva", "stop_id", "stop_name", "stop_lat", "stop_lon"]]

    def get_stop_sequences(self):
        if self.stop_times is None or self.stops is None:
            self.load()
        if self._stop_id_to_eva:
            valid_ids = set(self._stop_id_to_eva.keys())
            st = self.stop_times[self.stop_times["stop_id"].astype(str).isin(valid_ids)].copy()
            st["eva"] = st["stop_id"].astype(str).map(self._stop_id_to_eva)
        else:
            st = self.stop_times.copy()
            st["_eva"] = st["stop_id"].astype(str).map(self._eva_from_stop_id)
            st = st[st["_eva"].notna()].copy()
            st["eva"] = st["_eva"].astype(int)
        st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
        out = {e: [] for e in self.target_evas}
        for r in st.itertuples(index=False):
            eva = getattr(r, "eva")
            if eva in out:
                seq = getattr(r, "stop_sequence")
                out[eva].append({
                    "trip_id": getattr(r, "trip_id", None),
                    "stop_sequence": int(seq) if pd.notna(seq) else None,
                    "arrival_time": getattr(r, "arrival_time", None),
                    "departure_time": getattr(r, "departure_time", None),
                })
        return out

    def validate_target_mapping(self):
        ts = self.get_target_stops()
        mapped = set(ts["eva"].tolist())
        return set(self.target_evas).issubset(mapped)
