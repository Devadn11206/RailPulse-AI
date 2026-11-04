"""Ensure raw data files live under data/raw/..."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def organize_raw_files() -> None:
    (RAW / "gtfs").mkdir(parents=True, exist_ok=True)
    (RAW / "delays").mkdir(parents=True, exist_ok=True)
    (RAW / "weather").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    moves = [
        (["latest.zip", "germany_rail.zip"], RAW / "gtfs" / "germany_rail.zip"),
        (["data-2026-10.parquet"], RAW / "delays" / "data-2026-10.parquet"),
        (["stundenwerte_TU_01420_19810101_20251231_hist.zip"], RAW / "weather" / "stundenwerte_TU_01420_19810101_20251231_hist.zip"),
    ]
    for candidates, target in moves:
        if target.exists():
            continue
        for name in candidates:
            src = ROOT / name
            if src.exists() and src.is_file():
                shutil.move(str(src), str(target))
                break


if __name__ == "__main__":
    organize_raw_files()
    print("Raw files organized.")
