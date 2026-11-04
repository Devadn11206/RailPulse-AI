"""Run the real Phase 1 corridor data pipeline."""

import numpy as np
import pandas as pd
import polars as pl

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TARGET_STATION_EVAS
from src.data_ingestion.organize_raw_files import organize_raw_files
from src.preprocessing.data_cleaner import DelayDataCleaner
from src.feature_engineering.context_features import WeatherContextMatcher
from src.feature_engineering.temporal_features import TemporalFeatureBuilder


def main():
    organize_raw_files()

    root = Path(__file__).resolve().parents[2]
    gtfs_path = root / "data" / "raw" / "gtfs" / "germany_rail.zip"
    delay_path = root / "data" / "raw" / "delays" / "data-2026-10.parquet"
    weather_path = root / "data" / "raw" / "weather" / "stundenwerte_TU_01420_19810101_20251231_hist.zip"
    out_path = root / "data" / "processed" / "frankfurt_corridor_features.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Total raw rows
    total_raw = pl.scan_parquet(str(delay_path)).select(pl.len()).collect().item()

    # Filter to target EVAs and required columns
    lf = pl.scan_parquet(str(delay_path))
    lf = lf.with_columns(pl.col("eva").cast(pl.Int64, strict=False).alias("eva_int"))
    lf = lf.filter(pl.col("eva_int").is_in(TARGET_STATION_EVAS))
    lf = lf.filter(pl.col("arrival_planned_time").is_not_null() & pl.col("arrival_change_time").is_not_null())
    cols = ["train_line_ride_id", "id", "eva", "arrival_planned_time", "arrival_change_time", "time"]
    lf = lf.select(cols)
    raw_df = lf.collect().to_pandas()

    raw_row_count = len(raw_df)

    # Station coords (approximate; weather matcher uses single DWD station)
    station_coords = pd.DataFrame({
        "eva": TARGET_STATION_EVAS,
        "lat": [50.0] * len(TARGET_STATION_EVAS),
        "lon": [8.0] * len(TARGET_STATION_EVAS),
    })

    matcher = WeatherContextMatcher(station_coords)
    matcher.build_spatial_kdtree(None)
    weather_df = matcher.parse_dwd_zip(weather_path)

    cleaner = DelayDataCleaner(raw_df)
    clean_df = cleaner.clean_and_normalize()

    clean_df = matcher.attach_weather_features(clean_df, weather_df)

    builder = TemporalFeatureBuilder(lookback_steps=12)
    X, Y = builder.create_sliding_window_dataset(clean_df)

    np.savez_compressed(out_path, X=X, Y=Y)

    print("=" * 60)
    print("PHASE 1 REAL DATA PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total raw rows (parquet)     : {total_raw:,}")
    print(f"Filtered target rows         : {raw_row_count:,}")
    print(f"Clean rows                   : {len(clean_df):,}")
    print(f"Dropped duplicates           : {cleaner.quality_report.get('duplicates_dropped', 0):,}")
    print(f"Dropped leakage records      : {cleaner.quality_report.get('leakage_dropped', 0):,}")
    print(f"Final X tensor shape         : {X.shape}")
    print(f"Final Y target shape         : {Y.shape}")
    print(f"Saved to                     : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
