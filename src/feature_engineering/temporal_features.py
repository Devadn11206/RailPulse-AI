"""Temporal feature engineering and multi-horizon target construction."""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TIME_COL_CANDIDATES = [
    "scheduled_arrival", "event_timestamp", "observation_timestamp",
    "timestamp", "time", "datetime",
]
_DELAY_COL_CANDIDATES = ["delay_minutes", "delay", "delay_min", "delay_mins"]
_TEMP_COL_CANDIDATES = ["temperature", "air_temperature", "air_temp", "temp", "TT_TU"]
_STATION_COL_CANDIDATES = ["station_id", "station", "eva", "stop_id"]


class TemporalFeatureBuilder:
    def __init__(self, lookback_steps: int = 12):
        self.lookback_steps = int(lookback_steps)
        self.horizons = (3, 6, 12)

    def _resolve_time_col(self, df: pd.DataFrame) -> str:
        for c in _TIME_COL_CANDIDATES:
            if c in df.columns:
                return c
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                continue
        raise ValueError("No temporal column found in dataframe.")

    def _resolve_delay_col(self, df: pd.DataFrame) -> str:
        for c in _DELAY_COL_CANDIDATES:
            if c in df.columns:
                return c
        for c in df.columns:
            if "delay" in c.lower():
                return c
        raise ValueError("No delay column found in dataframe.")

    def _resolve_temp_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in _TEMP_COL_CANDIDATES:
            if c in df.columns:
                return c
        return None

    def _resolve_station_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in _STATION_COL_CANDIDATES:
            if c in df.columns:
                return c
        return None

    def create_sliding_window_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if df is None or len(df) == 0:
            X = np.zeros((0, self.lookback_steps, 7), dtype=np.float32)
            Y = np.zeros((0, 3), dtype=np.float32)
            return X, Y

        work = df.copy()
        time_col = self._resolve_time_col(work)
        delay_col = self._resolve_delay_col(work)
        temp_col = self._resolve_temp_col(work)
        station_col = self._resolve_station_col(work)

        work[time_col] = pd.to_datetime(work[time_col], errors="coerce", utc=True)
        work[delay_col] = pd.to_numeric(work[delay_col], errors="coerce").fillna(0.0)
        if temp_col is not None:
            work[temp_col] = pd.to_numeric(work[temp_col], errors="coerce").fillna(0.0)
        else:
            work["_temp"] = 0.0
            temp_col = "_temp"

        L = self.lookback_steps
        max_horizon = max(self.horizons)

        X_list = []
        Y_list = []

        groups = [work] if station_col is None else [g for _, g in work.groupby(station_col, sort=False)]
        for g in groups:
            g = g.sort_values(time_col, kind="mergesort").reset_index(drop=True)
            delays = g[delay_col].to_numpy(dtype=np.float64)
            temps = g[temp_col].to_numpy(dtype=np.float64)
            times = g[time_col]
            n = len(g) - L - max_horizon
            if n < 0:
                continue
            Xg = np.zeros((n, L, 7), dtype=np.float32)
            Yg = np.zeros((n, 3), dtype=np.float32)
            for i in range(n):
                window_delays = delays[i:i + L]
                window_times = times.iloc[i:i + L]
                window_temps = temps[i:i + L]
                w_mean = float(np.mean(window_delays)) if L > 0 else 0.0
                for j in range(L):
                    d = float(window_delays[j])
                    ts = window_times.iloc[j]
                    if pd.isna(ts):
                        hour, dow = 0, 0
                    else:
                        hour = ts.hour
                        dow = ts.dayofweek
                    sin_h = np.sin(2 * np.pi * hour / 24.0)
                    cos_h = np.cos(2 * np.pi * hour / 24.0)
                    sin_d = np.sin(2 * np.pi * dow / 7.0)
                    cos_d = np.cos(2 * np.pi * dow / 7.0)
                    Xg[i, j, 0] = d
                    Xg[i, j, 1] = sin_h
                    Xg[i, j, 2] = cos_h
                    Xg[i, j, 3] = sin_d
                    Xg[i, j, 4] = cos_d
                    Xg[i, j, 5] = w_mean
                    Xg[i, j, 6] = float(window_temps[j])
                for k, h in enumerate(self.horizons):
                    target_idx = i + L + h
                    if target_idx < len(delays):
                        Yg[i, k] = delays[target_idx]
                    else:
                        Yg[i, k] = 0.0
            X_list.append(Xg)
            Y_list.append(Yg)

        if X_list:
            X = np.concatenate(X_list, axis=0).astype(np.float32)
            Y = np.concatenate(Y_list, axis=0).astype(np.float32)
        else:
            X = np.zeros((0, L, 7), dtype=np.float32)
            Y = np.zeros((0, 3), dtype=np.float32)
        return X, Y
def chronological_train_val_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if n == 0:
        empty = np.zeros((0, X.shape[1], X.shape[2]), dtype=X.dtype) if X.ndim == 3 else np.zeros((0,), dtype=X.dtype)
        return empty, Y[:0], empty, Y[:0], empty, Y[:0]

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    assert train_end > 0, "Train split is empty."
    assert val_end > train_end, "Validation split must start after train."
    assert n > val_end, "Test split is empty."
    assert train_end < val_end < n, "Chronological index bounds violated."

    X_train = X[:train_end]
    Y_train = Y[:train_end]
    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]
    X_test = X[val_end:]
    Y_test = Y[val_end:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

