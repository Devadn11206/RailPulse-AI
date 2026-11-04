"""Leak-proof delay data cleaning and normalization."""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import TARGET_STATION_EVAS

logger = logging.getLogger(__name__)

_TIME_ALIASES = {
    "journey_id": ["journey_id", "trip_id", "train_id", "journey", "train_line_ride_id", "id"],
    "station_id": ["station_id", "stop_id", "eva", "station_eva", "eva_id", "stop", "xml_station_name"],
    "scheduled_arrival": ["scheduled_arrival", "scheduled_arrival_time", "planned_arrival", "arrival_scheduled", "sched_arrival", "arrival_planned_time"],
    "actual_arrival": ["actual_arrival", "actual_arrival_time", "arrival_actual", "actual_arr", "arrival_change_time"],
    "observation_timestamp": ["observation_timestamp", "obs_timestamp", "recorded_at", "observation_time", "time"],
    "event_timestamp": ["event_timestamp", "event_time"],
}


class DelayDataCleaner:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df.copy() if raw_df is not None else pd.DataFrame()
        self.quality_report: Dict[str, int] = {
            "rows_input": len(self.raw_df),
            "duplicates_dropped": 0,
            "leakage_dropped": 0,
            "rows_output": 0,
        }

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for canonical, aliases in _TIME_ALIASES.items():
            if canonical not in df.columns:
                for a in aliases:
                    if a in df.columns:
                        df = df.rename(columns={a: canonical})
                        break
        return df

    @staticmethod
    def _to_utc(series: pd.Series) -> pd.Series:
        s = pd.to_datetime(series, errors="coerce")
        if getattr(s.dt, "tz", None) is None:
            try:
                s = s.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="shift_forward")
            except Exception:
                s = s.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward")
        return s.dt.tz_convert("UTC")

    def clean_and_normalize(self) -> pd.DataFrame:
        df = self._standardize_columns(self.raw_df)
        n0 = len(df)

        # 1. Standardize timezone representation to UTC
        for col in ["scheduled_arrival", "actual_arrival", "observation_timestamp", "event_timestamp"]:
            if col in df.columns:
                df[col] = self._to_utc(df[col])

        # 2. Normalize station IDs to numeric EVAs
        if "station_id" in df.columns:
            df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
            df = df[df["station_id"].notna()].copy()
            df["station_id"] = df["station_id"].astype("int64")
            df = df[df["station_id"].isin(TARGET_STATION_EVAS)].copy()

        # 3. Ground-truth delay in minutes
        if "scheduled_arrival" in df.columns and "actual_arrival" in df.columns:
            df["delay_minutes"] = (df["actual_arrival"] - df["scheduled_arrival"]).dt.total_seconds() / 60.0
        else:
            if "delay_minutes" not in df.columns:
                df["delay_minutes"] = 0.0

        # 4. Remove duplicates for identical (journey_id, station_id, scheduled_arrival)
        dup_subset = [c for c in ["journey_id", "station_id", "scheduled_arrival"] if c in df.columns]
        if dup_subset:
            before = len(df)
            df = df.drop_duplicates(subset=dup_subset, keep="first")
            self.quality_report["duplicates_dropped"] = before - len(df)

        # 5. Filter temporal leakage: observation_timestamp > actual_arrival
        if "observation_timestamp" in df.columns and "actual_arrival" in df.columns:
            mask_leak = df["observation_timestamp"].notna() & df["actual_arrival"].notna() & (df["observation_timestamp"] > df["actual_arrival"])
            self.quality_report["leakage_dropped"] = int(mask_leak.sum())
            df = df[~mask_leak].copy()

        # 6. Clip extreme delays and fill NaN safely
        df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce")
        df["delay_minutes"] = df["delay_minutes"].fillna(0.0)
        df["delay_minutes"] = df["delay_minutes"].clip(lower=0.0, upper=300.0)

        # Fill other numeric NaNs
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].fillna(0.0)

        self.quality_report["rows_output"] = len(df)
        self.quality_report["rows_input"] = n0

        logger.info("DelayDataCleaner quality report: %s", self.quality_report)
        return df.reset_index(drop=True)
