"""Phase 0 setup diagnostic tests."""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_DIR, RAW_DIR, PROCESSED_DIR, GRAPH_DIR, GTFS_DIR, DELAYS_DIR,
    TARGET_STATION_EVAS, USE_AMP, DEVICE,
)
from src.data_ingestion.gtfs_ingestion import GTFSProcessor


DATA_SUBFOLDERS = [DATA_DIR, RAW_DIR, PROCESSED_DIR, GRAPH_DIR, GTFS_DIR, DELAYS_DIR,
                   RAW_DIR / "gtfs", RAW_DIR / "delays"]


def test_directory_write_access():
    for folder in DATA_SUBFOLDERS:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_probe"
        probe.write_text("ok")
        assert probe.read_text() == "ok"
        probe.unlink()
        assert not probe.exists()


def test_cuda_detection():
    assert torch.cuda.is_available(), "CUDA not available"
    assert DEVICE.type == "cuda"
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    assert vram_gb >= 3.5, f"Insufficient VRAM: {vram_gb:.2f} GB"
    print(f"GPU: {props.name}, VRAM: {vram_gb:.2f} GB")


def test_amp_initialization():
    assert USE_AMP is True
    # AMP autocast context
    with torch.amp.autocast("cuda", enabled=True):
        a = torch.randn(4, 4, device="cuda")
        b = torch.randn(4, 4, device="cuda")
        c = a @ b
    assert c.shape == (4, 4)
    # GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    assert scaler is not None
    # cleanup
    torch.cuda.empty_cache()


def test_gtfs_processor_mapping():
    proc = GTFSProcessor()
    proc.load()
    assert proc.validate_target_mapping(), "Not all target EVAs mapped"
    ts = proc.get_target_stops()
    mapped = set(ts["eva"].tolist())
    assert set(TARGET_STATION_EVAS).issubset(mapped)
    assert len(TARGET_STATION_EVAS) == 10
    seqs = proc.get_stop_sequences()
    assert len(seqs) == 10
    for eva in TARGET_STATION_EVAS:
        assert eva in seqs
