"""Master configuration for RailPulse AI."""

from pathlib import Path
import os

# Base paths derived dynamically via Path(__file__)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = DATA_DIR / "graph"
GTFS_DIR = RAW_DIR / "gtfs"
DELAYS_DIR = RAW_DIR / "delays"

# Corridor configuration
CORRIDOR_NAME = "Frankfurt_Main_Hub"
FRANKFURT_HBF_EVA = 8000105

# Core 10-station corridor (preserved for legacy test suite compliance)
TARGET_STATION_EVAS = [
    8000105,  # Frankfurt Hbf
    8000244,  # Mannheim Hbf
    8000240,  # Mainz Hbf
    8000207,  # Köln Hbf
    8000370,  # Wiesbaden Hbf
    8000130,  # Fulda
    8000107,  # Frankfurt Süd
    8000191,  # Karlsruhe Hbf
    8000096,  # Düsseldorf Hbf
    8000390,  # Würzburg Hbf
]

# Phase 10: Expanded 52-Station Regional Railway Network across Hesse, RP & BW
REGIONAL_STATION_EVAS_50 = [
    # Hesse (20)
    8000105, 8000107, 8070003, 8000150, 8000068, 8000250, 8000124, 8000130,
    8000260, 8000199, 8000198, 8000200, 8000318, 8000115, 8000392, 8000228,
    8000024, 8000036, 8000157, 8000185,
    # Rhineland-Palatinate (18)
    8000240, 8000257, 8000206, 8000236, 8000183, 8000344, 8000276, 8000216,
    8000340, 8000349, 8000025, 8000033, 8000004, 8000041, 8000114, 8000128,
    8000171, 8000305,
    # Baden-Württemberg & Intercity Connectors (14)
    8000244, 8000191, 8000152, 8000096, 8000142, 8000309, 8000050, 8000326,
    8000384, 8000013, 8000390, 8000207, 8000085, 8000261
]

# Forecast horizons (minutes)
FORECAST_HORIZONS = [15, 30, 60]

# Compute parameters
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
USE_AMP = CUDA_AVAILABLE
BATCH_SIZE = 32
LOOKBACK_STEPS = 12  # 60 min window (5-min intervals)

DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

# Ensure dirs exist
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, GRAPH_DIR, GTFS_DIR, DELAYS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
