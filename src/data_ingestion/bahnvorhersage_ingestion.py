"""BahnVorhersage historical delay observation ingestion."""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns
REQUIRED_COLS = ["event_timestamp", "observation_timestamp"]
ARRIVAL_COL_CANDIDATES = ["actual_arrival", "actual_arrival_time", "arrival_actual"]


class DelayObservationIngestor:
    def __init__(self, source_path: Optional[Union[str, Path]] = None):
        self.source_path = Path(source_path) if source_path is not None else None
        self.df: Optional[pd.DataFrame] = None

    def load(self, source_path: Optional[Union[str, Path]] = None) -> "DelayObservationIngestor":
        path = Path(source_path) if source_path is not None else self.source_path
        if path is None or not Path(path).exists():
            logger.warning("No delay dump found at %s; creating empty frame.", path)
            self.df = pd.DataFrame(columns=REQUIRED_COLS)
            self._normalize()
            return self
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            self.df = pd.read_parquet(path)
        elif path.suffix.lower() in (".csv", ".txt"):
            self.df = pd.read_csv(path, low_memory=False)
        else:
            # try csv
            self.df = pd.read_csv(path, low_memory=False)
        self._normalize()
        return self

    def _find_arrival_col(self) -> Optional[str]:
        if self.df is None:
            return None
        for c in ARRIVAL_COL_CANDIDATES:
            if c in self.df.columns:
                return c
        return None

    def _normalize(self):
        if self.df is None:
            self.df = pd.DataFrame()
            return
        # Ensure required columns exist
        for c in REQUIRED_COLS:
            if c not in self.df.columns:
                self.df[c] = pd.NaT
        # Parse timestamps
        for c in REQUIRED_COLS:
            self.df[c] = pd.to_datetime(self.df[c], errors="coerce", utc=True)
        # arrival col
        arr_col = self._find_arrival_col()
        if arr_col is not None:
            self.df[arr_col] = pd.to_datetime(self.df[arr_col], errors="coerce", utc=True)
            # standardize name
            if arr_col != "actual_arrival":
                self.df["actual_arrival"] = self.df[arr_col]
        else:
            if "actual_arrival" not in self.df.columns:
                self.df["actual_arrival"] = pd.NaT
            else:
                self.df["actual_arrival"] = pd.to_datetime(self.df["actual_arrival"], errors="coerce", utc=True)
        # Flag invalid rows: observation_timestamp > actual_arrival
        self.df["invalid_leakage"] = False
        mask = self.df["observation_timestamp"].notna() & self.df["actual_arrival"].notna() & (self.df["observation_timestamp"] > self.df["actual_arrival"])
        self.df.loc[mask, "invalid_leakage"] = True

    def get_valid(self) -> pd.DataFrame:
        if self.df is None:
            self.load()
        return self.df[~self.df["invalid_leakage"]].copy()

    def get_invalid(self) -> pd.DataFrame:
        if self.df is None:
            self.load()
        return self.df[self.df["invalid_leakage"]].copy()

    def summary(self) -> dict:
        if self.df is None:
            self.load()
        return {
            "total": len(self.df),
            "invalid": int(self.df["invalid_leakage"].sum()) if "invalid_leakage" in self.df.columns else 0,
            "valid": int((~self.df["invalid_leakage"]).sum()) if "invalid_leakage" in self.df.columns else len(self.df),
        }
