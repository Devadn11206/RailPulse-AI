"""Contextual weather feature matching via spatial KD-Tree."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

_LAT_CANDIDATES = ["lat", "latitude", "stop_lat", "weather_lat"]
_LON_CANDIDATES = ["lon", "lng", "longitude", "stop_lon", "weather_lon"]
_EVA_CANDIDATES = ["eva", "station_id", "stop_id", "station_eva", "eva_id"]
_TEMP_CANDIDATES = ["temperature", "temp", "air_temp"]
_PRECIP_CANDIDATES = ["precipitation", "precip", "rain", "rainfall"]
_WIND_CANDIDATES = ["wind_speed", "wind", "windspeed"]


class WeatherContextMatcher:
    def __init__(self, station_coords_df: pd.DataFrame):
        self.station_coords = self._normalize_coords(station_coords_df)
        self.tree = None
        self.weather_stations = None
        self._eva_to_weather = {}

    @staticmethod
    def _normalize_coords(df):
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["eva", "lat", "lon"])
        out = df.copy()
        eva_col = None
        for c in _EVA_CANDIDATES:
            if c in out.columns:
                eva_col = c
                break
        if eva_col is None:
            out["eva"] = range(len(out))
        elif eva_col != "eva":
            out = out.rename(columns={eva_col: "eva"})
        lat_col = None
        for c in _LAT_CANDIDATES:
            if c in out.columns:
                lat_col = c
                break
        if lat_col is None:
            out["lat"] = 0.0
        elif lat_col != "lat":
            out = out.rename(columns={lat_col: "lat"})
        lon_col = None
        for c in _LON_CANDIDATES:
            if c in out.columns:
                lon_col = c
                break
        if lon_col is None:
            out["lon"] = 0.0
        elif lon_col != "lon":
            out = out.rename(columns={lon_col: "lon"})
        out["eva"] = pd.to_numeric(out["eva"], errors="coerce")
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce").fillna(0.0)
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce").fillna(0.0)
        return out[["eva", "lat", "lon"]].reset_index(drop=True)

    @staticmethod
    def _normalize_weather(df):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        out = df.copy()
        if "weather_station_id" not in out.columns:
            for c in ["station_id", "id", "station", "wmo", "dwd_id"]:
                if c in out.columns:
                    out = out.rename(columns={c: "weather_station_id"})
                    break
            if "weather_station_id" not in out.columns:
                out["weather_station_id"] = range(len(out))
        lat_col = None
        for c in _LAT_CANDIDATES:
            if c in out.columns:
                lat_col = c
                break
        if lat_col and lat_col != "lat":
            out = out.rename(columns={lat_col: "lat"})
        if "lat" not in out.columns:
            out["lat"] = 0.0
        lon_col = None
        for c in _LON_CANDIDATES:
            if c in out.columns:
                lon_col = c
                break
        if lon_col and lon_col != "lon":
            out = out.rename(columns={lon_col: "lon"})
        if "lon" not in out.columns:
            out["lon"] = 0.0
        out["lat"] = pd.to_numeric(out["lat"], errors="coerce").fillna(0.0)
        out["lon"] = pd.to_numeric(out["lon"], errors="coerce").fillna(0.0)

        def _rename_feature(candidates, target):
            for c in candidates:
                if c in out.columns and c != target:
                    out.rename(columns={c: target}, inplace=True)
                    return

        _rename_feature(_TEMP_CANDIDATES, "temperature")
        _rename_feature(_PRECIP_CANDIDATES, "precipitation")
        _rename_feature(_WIND_CANDIDATES, "wind_speed")
        for t in ["temperature", "precipitation", "wind_speed"]:
            if t not in out.columns:
                out[t] = 0.0
        return out.reset_index(drop=True)

    def _mock_weather_stations(self):
        rows = []
        for _, r in self.station_coords.iterrows():
            rows.append({
                "weather_station_id": f"WX_{int(r['eva']) if pd.notna(r['eva']) else len(rows)}",
                "lat": r["lat"] + 0.05,
                "lon": r["lon"] + 0.05,
                "temperature": 12.0,
                "precipitation": 0.0,
                "wind_speed": 5.0,
            })
        if not rows:
            rows.append({"weather_station_id": "WX_0", "lat": 50.1, "lon": 8.6, "temperature": 12.0, "precipitation": 0.0, "wind_speed": 5.0})
        return pd.DataFrame(rows)

    def build_spatial_kdtree(self, weather_stations_df=None):
        if weather_stations_df is None or (isinstance(weather_stations_df, pd.DataFrame) and len(weather_stations_df) == 0):
            ws = self._mock_weather_stations()
        else:
            ws = self._normalize_weather(weather_stations_df)
            if len(ws) == 0:
                ws = self._mock_weather_stations()
        self.weather_stations = ws.reset_index(drop=True)
        coords = self.weather_stations[["lat", "lon"]].to_numpy(dtype=float)
        self.tree = KDTree(coords)
        self._eva_to_weather = {}
        for _, r in self.station_coords.iterrows():
            eva = r["eva"]
            if pd.isna(eva):
                continue
            q = np.array([[r["lat"], r["lon"]]])
            dist, idx = self.tree.query(q)
            idx = int(idx) if np.isscalar(idx) else int(idx[0])
            self._eva_to_weather[int(eva)] = idx
        return self

    def parse_dwd_zip(self, zip_path) -> pd.DataFrame:
        import zipfile
        from pathlib import Path

        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path) as z:
            product = None
            for n in z.namelist():
                if n.lower().endswith(".txt") and "produkt_tu_stunde" in n.lower():
                    product = n
                    break
            if product is None:
                raise FileNotFoundError("No produkt_tu_stunde_*.txt found in weather zip.")
            with z.open(product) as f:
                weather = pd.read_csv(f, sep=";", skipinitialspace=True)
        # Normalize columns
        time_col = None
        for c in ["MESS_DATUM", "mess_datum"]:
            if c in weather.columns:
                time_col = c
                break
        temp_col = None
        for c in ["TT_TU", "tt_tu", "temperature"]:
            if c in weather.columns:
                temp_col = c
                break
        if time_col is None or temp_col is None:
            raise ValueError("Weather file missing MESS_DATUM or TT_TU columns.")
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(weather[time_col].astype(str), format="%Y%m%d%H", errors="coerce")
        out["temperature"] = pd.to_numeric(weather[temp_col], errors="coerce")
        # Localize Europe/Berlin -> UTC
        if getattr(out["timestamp"].dt, "tz", None) is None:
            out["timestamp"] = out["timestamp"].dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward")
        out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        # Fill missing temperatures
        out["temperature"] = out["temperature"].ffill().bfill().fillna(0.0)
        return out

    def parse_dwd_zip(self, zip_path) -> "pd.DataFrame":
        import zipfile
        from pathlib import Path

        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path) as z:
            product = None
            for n in z.namelist():
                if n.lower().endswith(".txt") and "produkt_tu_stunde" in n.lower():
                    product = n
                    break
            if product is None:
                raise FileNotFoundError("No produkt_tu_stunde_*.txt found in weather zip.")
            with z.open(product) as f:
                weather = pd.read_csv(f, sep=";", skipinitialspace=True)
        time_col = None
        for c in ["MESS_DATUM", "mess_datum"]:
            if c in weather.columns:
                time_col = c
                break
        temp_col = None
        for c in ["TT_TU", "tt_tu", "temperature"]:
            if c in weather.columns:
                temp_col = c
                break
        if time_col is None or temp_col is None:
            raise ValueError("Weather file missing MESS_DATUM or TT_TU columns.")
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(weather[time_col].astype(str), format="%Y%m%d%H", errors="coerce")
        out["temperature"] = pd.to_numeric(weather[temp_col], errors="coerce")
        if getattr(out["timestamp"].dt, "tz", None) is None:
            out["timestamp"] = out["timestamp"].dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward")
        out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        out["temperature"] = out["temperature"].ffill().bfill().fillna(0.0)
        return out


    def attach_weather_features(self, delay_df, weather_obs_df=None):
        if self.tree is None:
            self.build_spatial_kdtree(None)

        df = delay_df.copy()

        station_col = None
        for c in _EVA_CANDIDATES:
            if c in df.columns:
                station_col = c
                break
        if station_col is None:
            df["station_id"] = 0
            station_col = "station_id"

        time_col = None
        for c in ["observation_timestamp", "event_timestamp", "timestamp", "time", "datetime", "scheduled_arrival"]:
            if c in df.columns:
                time_col = c
                break

        def _nearest_idx(eva):
            try:
                eva_i = int(eva)
            except (ValueError, TypeError):
                return 0
            return self._eva_to_weather.get(eva_i, 0)

        df["_wx_idx"] = df[station_col].map(_nearest_idx).fillna(0).astype(int)
        df["nearest_weather_station"] = df["_wx_idx"].map(
            lambda i: self.weather_stations.loc[i, "weather_station_id"] if i < len(self.weather_stations) else None
        )

        # Ensure weather columns exist
        for t in ["temperature", "precipitation", "wind_speed"]:
            if t not in df.columns:
                df[t] = 0.0

        if weather_obs_df is not None and len(weather_obs_df) > 0:
            wdf = weather_obs_df.copy()
            # Resolve weather time column
            wtime_col = None
            for c in ["timestamp", "observation_timestamp", "time", "datetime", "MESS_DATUM"]:
                if c in wdf.columns:
                    wtime_col = c
                    break
            # Resolve temperature column
            wtemp_col = None
            for c in ["temperature", "TT_TU", "air_temp", "temp"]:
                if c in wdf.columns:
                    wtemp_col = c
                    break
            if wtemp_col is not None and wtemp_col != "temperature":
                wdf["temperature"] = wdf[wtemp_col]

            if time_col is not None and wtime_col is not None:
                try:
                    left = df.copy()
                    left["_ts"] = pd.to_datetime(left[time_col], errors="coerce", utc=True)
                    wdf["_ts"] = pd.to_datetime(wdf[wtime_col], errors="coerce", utc=True)
                    if getattr(wdf["_ts"].dt, "tz", None) is None:
                        wdf["_ts"] = wdf["_ts"].dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
                    merged = pd.merge_asof(
                        left.sort_values("_ts"),
                        wdf[["_ts", "temperature"]].sort_values("_ts"),
                        on="_ts",
                        direction="nearest",
                    )
                    if "temperature_y" in merged.columns:
                        df["temperature"] = merged["temperature_y"].values
                    elif "temperature" in merged.columns:
                        df["temperature"] = merged["temperature"].values
                except Exception:
                    pass
        else:
            for idx, ws_row in self.weather_stations.iterrows():
                mask = df["_wx_idx"] == idx
                df.loc[mask, "temperature"] = ws_row.get("temperature", 12.0)

        # Fill any remaining NaNs
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").fillna(0.0)
        df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0.0)
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce").fillna(0.0)

        df = df.drop(columns=["_wx_idx"], errors="ignore")
        return df.reset_index(drop=True)
