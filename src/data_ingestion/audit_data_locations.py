"""Audit and normalize raw/processed data locations for RailPulse AI."""

import shutil
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
GTFS_DIR = RAW_DIR / "gtfs"
DELAYS_DIR = RAW_DIR / "delays"
WEATHER_DIR = RAW_DIR / "weather"


def _size_str(path: Path) -> str:
    if not path.exists():
        return "N/A"
    size = path.stat().st_size
    if size >= 1024 ** 3:
        return f"{size / (1024 ** 3):.2f} GB"
    return f"{size / (1024 ** 2):.2f} MB"


def _move_from_root(candidate_names: List[str], target: Path) -> Tuple[Optional[Path], str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return None, "FOUND"
    for name in candidate_names:
        src = ROOT / name
        if src.exists() and src.is_file():
            shutil.move(str(src), str(target))
            return src, "MOVED"
    return None, "MISSING"
def _gtfs_inspect(path: Path) -> str:
    if not path.exists():
        return "N/A (file missing)"
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            has_stops = any(n.endswith("stops.txt") for n in names)
            has_stop_times = any(n.endswith("stop_times.txt") for n in names)
            return f"zip entries={len(names)} | stops.txt={'FOUND' if has_stops else 'MISSING'} | stop_times.txt={'FOUND' if has_stop_times else 'MISSING'}"
    except Exception as e:
        return f"ERROR reading zip: {e}"


def _parquet_inspect(path: Path) -> str:
    if not path.exists():
        return "N/A (file missing)"
    try:
        import polars as pl

        lf = pl.scan_parquet(str(path))
        cols = lf.collect_schema().names()
        row_count = lf.select(pl.len()).collect().item()
        preview = ", ".join(cols[:12])
        if len(cols) > 12:
            preview += f", ... (+{len(cols) - 12} more)"
        return f"rows={row_count:,} | columns={len(cols)} | schema=[{preview}]"
    except Exception as e:
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            return f"rows={len(df):,} | columns={len(df.columns)} | schema=[{', '.join(df.columns[:12])}]"
        except Exception as e2:
            return f"ERROR reading parquet: {e} / {e2}"


def _weather_inspect(path: Path) -> str:
    if not path.exists():
        return "N/A (file missing)"
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            txts = [n for n in names if n.lower().endswith(".txt")]
            chosen = None
            for n in txts:
                if "produkt" in n.lower():
                    chosen = n
                    break
            if chosen is None and txts:
                chosen = max(txts, key=lambda n: z.getinfo(n).file_size)
            if chosen is None:
                return f"zip entries={len(names)} | no .txt found"
            with z.open(chosen) as f:
                line_count = sum(1 for _ in f)
            return f"zip entries={len(names)} | inner file={chosen} | lines={line_count:,}"
    except Exception as e:
        return f"ERROR reading zip: {e}"


def _npz_inspect(path: Path) -> str:
    if not path.exists():
        return "N/A (file missing)"
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as z:
            keys = list(z.files)
            details = []
            for k in keys[:8]:
                arr = z[k]
                details.append(f"{k}{tuple(arr.shape)}")
            if len(keys) > 8:
                details.append(f"... (+{len(keys)-8} more)")
            return f"arrays={len(keys)} | " + ", ".join(details)
    except Exception as e:
        return f"ERROR reading npz: {e}"


def main():
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    DELAYS_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("GTFS Static Feed", GTFS_DIR / "germany_rail.zip", ["germany_rail.zip", "latest.zip"], _gtfs_inspect),
        ("Historical Delay Parquet", DELAYS_DIR / "data-2026-10.parquet", ["data-2026-10.parquet"], _parquet_inspect),
        ("DWD Weather Zip", WEATHER_DIR / "stundenwerte_TU_01420_19810101_20251231_hist.zip",
         ["stundenwerte_TU_01420_19810101_20251231_hist.zip"], _weather_inspect),
        ("Processed Features NPZ", PROCESSED_DIR / "frankfurt_corridor_features.npz",
         ["frankfurt_corridor_features.npz"], _npz_inspect),
    ]

    records = []
    for label, target, candidates, inspector in targets:
        moved_src, status = _move_from_root(candidates, target)
        if status == "MOVED":
            detail = f"moved from {moved_src}"
        elif status == "FOUND":
            detail = "already in target location"
        else:
            detail = "not found in target or project root"
        records.append((label, target, status, detail, inspector(target)))

    print("=" * 96)
    print("RAILPULSE DATA LOCATION AUDIT REPORT")
    print("=" * 96)
    print(f"Project Root : {ROOT}")
    print(f"Raw Dir      : {RAW_DIR}")
    print(f"Processed Dir: {PROCESSED_DIR}")
    print("-" * 96)

    for i, (label, target, status, detail, content) in enumerate(records, start=1):
        print(f"[{i}] {label}")
        print(f"    File Name : {target.name}")
        print(f"    Path      : {target}")
        print(f"    Status    : {status}")
        print(f"    Size      : {_size_str(target)}")
        print(f"    Note      : {detail}")
        print(f"    Content   : {content}")
        print("-" * 96)

    found = sum(1 for r in records if r[2] in ("FOUND", "MOVED") and r[1].exists())
    missing = [r[0] for r in records if not r[1].exists()]
    print(f"Summary: {found}/{len(records)} files available | Missing: {missing if missing else 'None'}")
    print("=" * 96)


if __name__ == "__main__":
    main()

