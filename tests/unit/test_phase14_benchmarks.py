"""Unit tests for Phase 14: Rigorous Benchmark Audit of RailPulse ST-GATv2 vs Baselines."""

import json
from pathlib import Path
import pytest
import numpy as np

from src.analytics.benchmark_evaluator import BenchmarkAuditor, DEFAULT_BENCHMARK_PATH
from src.config import REGIONAL_STATION_EVAS_50


class TestPhase14Benchmarks:
    """Test suite verifying empirical audit results, metric validity, and regional coverage."""

    @pytest.fixture(scope="class")
    def benchmark_data(self):
        """Ensure benchmark results JSON exists and load it."""
        if not DEFAULT_BENCHMARK_PATH.exists():
            auditor = BenchmarkAuditor()
            data = auditor.evaluate_all()
        else:
            with open(DEFAULT_BENCHMARK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        return data

    def test_benchmark_metadata_integrity(self, benchmark_data):
        """Verify chronological split sample counts and regional node counts."""
        meta = benchmark_data["metadata"]
        assert meta["test_snapshots"] == 7679, f"Expected 7679 test snapshots, got {meta['test_snapshots']}"
        assert meta["total_snapshots"] == 51193, f"Expected 51193 total snapshots, got {meta['total_snapshots']}"
        assert meta["regional_station_nodes"] == len(REGIONAL_STATION_EVAS_50)
        assert meta["regional_track_edges"] > 100

    def test_st_gatv2_superior_r2_score(self, benchmark_data):
        """Verify ST-GATv2 achieves R^2 > 0.18 across all forecast horizons."""
        models = benchmark_data["models"]
        assert "st_gatv2" in models
        st = models["st_gatv2"]

        assert st["15m"]["r2"] > 0.18, f"Expected 15m R2 > 0.18, got {st['15m']['r2']}"
        assert st["30m"]["r2"] > 0.18, f"Expected 30m R2 > 0.18, got {st['30m']['r2']}"
        assert st["60m"]["r2"] > 0.18, f"Expected 60m R2 > 0.18, got {st['60m']['r2']}"
        assert st["overall"]["r2"] > 0.18, f"Expected overall R2 > 0.18, got {st['overall']['r2']}"

    def test_st_gatv2_outperforms_all_tabular_baselines(self, benchmark_data):
        """Verify ST-GATv2 MAE (~0.68m) is substantially lower than classical baselines (>7.0m)."""
        models = benchmark_data["models"]
        st_mae = models["st_gatv2"]["overall"]["mae"]
        assert st_mae < 1.0, f"ST-GATv2 MAE too high: {st_mae}"

        for baseline_key in ["naive_persistence", "historical_average", "random_forest", "xgboost", "lightgbm"]:
            if models.get(baseline_key):
                base_mae = models[baseline_key]["overall"]["mae"]
                base_r2 = models[baseline_key]["overall"]["r2"]
                assert st_mae < base_mae, f"ST-GATv2 MAE ({st_mae}) did not beat {baseline_key} MAE ({base_mae})"
                assert base_r2 < 0.05, f"Expected baseline R2 < 0.05, got {base_r2} for {baseline_key}"

    def test_regional_coverage_and_nontrivial_predictions(self, benchmark_data):
        """Verify all 52 regional stations produce non-trivial, finite predictions."""
        reg = benchmark_data["regional_coverage"]
        assert reg["num_stations"] == len(REGIONAL_STATION_EVAS_50)
        assert reg["all_finite"] is True
        assert reg["non_trivial_status"] is True
        assert reg["prediction_variance"] > 1e-4
        assert len(reg["stations_verified"]) == len(REGIONAL_STATION_EVAS_50)

    def test_wape_and_rmse_metrics_validity(self, benchmark_data):
        """Verify mathematical bounds of RMSE and WAPE across all models."""
        st = benchmark_data["models"]["st_gatv2"]
        for h in ["15m", "30m", "60m", "overall"]:
            assert st[h]["rmse"] > 0.0
            assert 0.0 < st[h]["wape"] < 2.0
            assert st[h]["mae"] < st[h]["rmse"]
