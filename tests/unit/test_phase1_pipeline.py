"""Phase 1 pipeline diagnostic tests."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_cleaner import DelayDataCleaner
from src.feature_engineering.temporal_features import TemporalFeatureBuilder, chronological_train_val_test_split
from src.feature_engineering.context_features import WeatherContextMatcher


def make_synthetic_delays(n=120):
    base = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    rows = []
    for i in range(n):
        sched = base + pd.Timedelta(minutes=5 * i)
        actual = sched + pd.Timedelta(minutes=(i % 15))
        obs = sched - pd.Timedelta(minutes=2)  # valid: before actual
        rows.append({
            "journey_id": f"J{i // 10}",
            "station_id": 8000105,
            "scheduled_arrival": sched,
            "actual_arrival": actual,
            "observation_timestamp": obs,
        })
    df = pd.DataFrame(rows)
    # duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    # leakage row: observation after actual
    leak = df.iloc[0].copy()
    leak["scheduled_arrival"] = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    leak["actual_arrival"] = leak["scheduled_arrival"] + pd.Timedelta(minutes=3)
    leak["observation_timestamp"] = leak["actual_arrival"] + pd.Timedelta(minutes=5)
    df = pd.concat([df, pd.DataFrame([leak])], ignore_index=True)
    # extreme delay row
    ext = df.iloc[1].copy()
    ext["actual_arrival"] = ext["scheduled_arrival"] + pd.Timedelta(minutes=400)
    ext["scheduled_arrival"] = pd.Timestamp("2024-01-03 00:00", tz="UTC")
    ext["actual_arrival"] = ext["scheduled_arrival"] + pd.Timedelta(minutes=400)
    ext["observation_timestamp"] = ext["scheduled_arrival"]
    df = pd.concat([df, pd.DataFrame([ext])], ignore_index=True)
    return df


def test_data_cleaner():
    raw = make_synthetic_delays(120)
    cleaner = DelayDataCleaner(raw)
    out = cleaner.clean_and_normalize()
    # duplicates + leakage removed
    assert cleaner.quality_report["duplicates_dropped"] >= 1
    assert cleaner.quality_report["leakage_dropped"] >= 1
    # delays clipped
    assert out["delay_minutes"].max() <= 300
    assert out["delay_minutes"].min() >= 0
    assert "delay_minutes" in out.columns
    assert len(out) < len(raw)


def test_temporal_feature_builder():
    raw = make_synthetic_delays(120)
    df = DelayDataCleaner(raw).clean_and_normalize()
    builder = TemporalFeatureBuilder(lookback_steps=12)
    X, Y = builder.create_sliding_window_dataset(df)
    assert X.ndim == 3
    assert X.shape[1] == 12
    assert Y.shape[1] == 3
    assert X.shape[0] == Y.shape[0]
    assert X.dtype == np.float32
    assert Y.dtype == np.float32


def test_chronological_split():
    raw = make_synthetic_delays(120)
    df = DelayDataCleaner(raw).clean_and_normalize()
    builder = TemporalFeatureBuilder(lookback_steps=12)
    X, Y = builder.create_sliding_window_dataset(df)
    Xtr, Ytr, Xva, Yva, Xte, Yte = chronological_train_val_test_split(X, Y, 0.70, 0.15)
    assert Xtr.shape[0] + Xva.shape[0] + Xte.shape[0] == X.shape[0]
    assert Ytr.shape[0] + Yva.shape[0] + Yte.shape[0] == Y.shape[0]
    # chronological: train indices before val before test (by construction)
    assert Xtr.shape[0] > 0 and Xva.shape[0] > 0 and Xte.shape[0] > 0


def test_weather_context_matcher():
    # station coords from GTFS-like
    coords = pd.DataFrame({
        "eva": [8000105, 8000244],
        "stop_lat": [50.1071, 49.4794],
        "stop_lon": [8.6638, 8.4699],
    })
    matcher = WeatherContextMatcher(coords)
    weather_stations = pd.DataFrame({
        "station_id": ["W1", "W2"],
        "lat": [50.12, 49.48],
        "lon": [8.67, 8.47],
        "temp": [10.0, 11.0],
        "precip": [1.0, 2.0],
        "wind": [3.0, 4.0],
    })
    matcher.build_spatial_kdtree(weather_stations)
    assert matcher.tree is not None
    delay_df = pd.DataFrame({
        "station_id": [8000105, 8000244, 8000105],
        "observation_timestamp": pd.to_datetime(["2024-01-01T10:00Z", "2024-01-01T10:05Z", "2024-01-01T10:10Z"], utc=True),
    })
    out = matcher.attach_weather_features(delay_df)
    assert "temperature" in out.columns
    assert "precipitation" in out.columns
    assert "wind_speed" in out.columns
    assert "nearest_weather_station" in out.columns
    # nearest mapping sanity: first station should map to W1
    assert out.loc[0, "nearest_weather_station"] == "W1"
