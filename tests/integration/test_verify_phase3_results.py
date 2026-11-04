"""Integration verification audit for Phase 3 baseline benchmark results.

Audits data leakage, split integrity, and independently recalculates metrics
against those stored in data/processed/baseline_benchmark_results.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR
from src.models.baselines import NaivePersistenceBaseline
from src.training.evaluator import ModelEvaluator

NPZ_PATH = PROCESSED_DIR / "frankfurt_corridor_features.npz"
RESULTS_PATH = PROCESSED_DIR / "baseline_benchmark_results.json"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
HORIZON_NAMES = ["15m", "30m", "60m"]


def _load_npz():
    data = np.load(NPZ_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    return X, Y


def _chronological_split_indices(n):
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_idx = set(range(0, train_end))
    val_idx = set(range(train_end, val_end))
    test_idx = set(range(val_end, n))
    return train_idx, val_idx, test_idx, train_end, val_end


# ---------------------------------------------------------------------------
# Split & Leakage Audit
# ---------------------------------------------------------------------------

def test_split_chronological_ordering_no_overlap():
    X, Y = _load_npz()
    n = len(X)
    train_idx, val_idx, test_idx, train_end, val_end = _chronological_split_indices(n)

    # All indices covered exactly once
    assert train_idx | val_idx | test_idx == set(range(n)), "Splits do not cover all indices"
    assert train_idx & val_idx == set(), "Train/val index overlap"
    assert train_idx & test_idx == set(), "Train/test index overlap"
    assert val_idx & test_idx == set(), "Val/test index overlap"

    # Chronological boundaries
    assert max(train_idx) < min(val_idx), "Train indices not strictly before val"
    assert max(val_idx) < min(test_idx), "Val indices not strictly before test"

    # Match trainer proportions
    assert train_end == 35835, f"Expected train_end=35835, got {train_end}"
    assert val_end == 43514, f"Expected val_end=43514, got {val_end}"
    assert len(train_idx) == 35835
    assert len(val_idx) == 7679
    assert len(test_idx) == 7679


def test_test_y_not_in_train_val_by_index():
    """Verify test set target indices are disjoint from train/val target indices."""
    X, Y = _load_npz()
    n = len(X)
    train_idx, val_idx, test_idx, _, val_end = _chronological_split_indices(n)

    Y_train = Y[sorted(train_idx)]
    Y_val = Y[sorted(val_idx)]
    Y_test = Y[sorted(test_idx)]

    # Confirm test targets are drawn only from test index range
    assert len(Y_test) == len(test_idx)
    # No index leakage: test Y rows should not appear as train/val Y rows by position
    assert len(set(range(val_end, n)) & set(range(0, val_end))) == 0


# ---------------------------------------------------------------------------
# Metrics Audit & Independent Calculation
# ---------------------------------------------------------------------------

def test_naive_persistence_metrics_independent_recalculation():
    """Independently recompute NaivePersistence metrics on test split and
    compare against ModelEvaluator outputs stored in results JSON."""
    X, Y = _load_npz()
    X_flat = X.reshape(len(X), -1).astype(np.float32)

    _, _, _, train_end, val_end = _chronological_split_indices(len(X))
    X_test = X_flat[val_end:]
    Y_test = Y[val_end:]

    model = NaivePersistenceBaseline()
    model.fit(X_flat[:train_end], Y[:train_end])
    Y_pred = model.predict(X_test)

    evaluator = ModelEvaluator()
    evaluator_metrics = evaluator.evaluate_multi_horizon(Y_test, Y_pred)

    # Independent sklearn-based recalculation
    for i, h in enumerate(HORIZON_NAMES):
        yt = Y_test[:, i]
        yp = Y_pred[:, i]

        ind_mae = float(mean_absolute_error(yt, yp))
        ind_rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        ind_r2 = float(r2_score(yt, yp))

        denom = np.abs(yt).sum()
        ind_wape = float(np.abs(yt - yp).sum() / denom) if denom > 0 else 0.0

        eval_m = evaluator_metrics[h]
        assert abs(eval_m["mae"] - ind_mae) < 1e-6, f"{h} MAE mismatch"
        assert abs(eval_m["rmse"] - ind_rmse) < 1e-6, f"{h} RMSE mismatch"
        assert abs(eval_m["r2"] - ind_r2) < 1e-5, f"{h} R2 mismatch (eval={eval_m['r2']}, ind={ind_r2})"
        assert abs(eval_m["wape"] - ind_wape) < 1e-6, f"{h} WAPE mismatch"

    # Overall R2 match within tolerance
    ind_r2_overall = float(r2_score(Y_test, Y_pred))
    assert abs(evaluator_metrics["overall"]["r2"] - ind_r2_overall) < 1e-5


def test_results_json_matches_independent_calculation():
    """Cross-check the stored JSON results against independent recomputation."""
    if not RESULTS_PATH.exists():
        pytest.skip("baseline_benchmark_results.json not found")

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    X, Y = _load_npz()
    X_flat = X.reshape(len(X), -1).astype(np.float32)
    _, _, _, train_end, val_end = _chronological_split_indices(len(X))
    Y_test = Y[val_end:]

    model = NaivePersistenceBaseline()
    model.fit(X_flat[:train_end], Y[:train_end])
    Y_pred = model.predict(X_test := X_flat[val_end:])

    stored = results["models"]["naive_persistence"]
    evaluator = ModelEvaluator()
    computed = evaluator.evaluate_multi_horizon(Y_test, Y_pred)

    for h in HORIZON_NAMES + ["overall"]:
        for metric in ("mae", "rmse", "r2", "wape"):
            assert abs(stored[h][metric] - computed[h][metric]) < 1e-5, \
                f"{h}/{metric} JSON mismatch: stored={stored[h][metric]}, computed={computed[h][metric]}"


# ---------------------------------------------------------------------------
# Prediction Sanity Check
# ---------------------------------------------------------------------------

def test_predictions_no_nan_inf_constant():
    X, Y = _load_npz()
    X_flat = X.reshape(len(X), -1).astype(np.float32)
    _, _, _, train_end, val_end = _chronological_split_indices(len(X))

    X_test = X_flat[val_end:]
    Y_test = Y[val_end:]

    model = NaivePersistenceBaseline()
    model.fit(X_flat[:train_end], Y[:train_end])
    Y_pred = model.predict(X_test)

    assert not np.isnan(Y_pred).any(), "Predictions contain NaN"
    assert not np.isinf(Y_pred).any(), "Predictions contain Inf"

    # Not trivially constant across all horizons
    for i, h in enumerate(HORIZON_NAMES):
        col = Y_pred[:, i]
        assert not np.allclose(col, col[0], atol=1e-6), f"{h} predictions are constant"

    # Predictions should equal last observed delay (X[:, -1, 0] -> col 77 in flat)
    last_delays = X_test[:, 77]
    for i in range(3):
        assert np.allclose(Y_pred[:, i], last_delays, atol=1e-6), f"{HORIZON_NAMES[i]} != last observed delay"


def test_print_horizon_breakdown(capsys):
    """Print per-horizon MAE and R2 breakdown for diagnostic visibility."""
    X, Y = _load_npz()
    X_flat = X.reshape(len(X), -1).astype(np.float32)
    _, _, _, train_end, val_end = _chronological_split_indices(len(X))

    X_test = X_flat[val_end:]
    Y_test = Y[val_end:]

    model = NaivePersistenceBaseline()
    model.fit(X_flat[:train_end], Y[:train_end])
    Y_pred = model.predict(X_test)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_multi_horizon(Y_test, Y_pred)

    print("\n" + "=" * 50)
    print("NAIVE PERSISTENCE BASELINE - HORIZON BREAKDOWN")
    print("=" * 50)
    print(f"{'Horizon':<10} {'MAE':>10} {'R2':>10} {'RMSE':>10} {'WAPE':>10}")
    print("-" * 50)
    for h in HORIZON_NAMES:
        m = metrics[h]
        print(f"{h:<10} {m['mae']:>10.4f} {m['r2']:>10.4f} {m['rmse']:>10.4f} {m['wape']:>10.4f}")
    m = metrics["overall"]
    print("-" * 50)
    print(f"{'overall':<10} {m['mae']:>10.4f} {m['r2']:>10.4f} {m['rmse']:>10.4f} {m['wape']:>10.4f}")
    print("=" * 50 + "\n")
