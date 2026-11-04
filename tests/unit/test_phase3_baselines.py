"""Phase 3 classical baseline diagnostic tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.baselines import NaivePersistenceBaseline, TreeBaselineSuite
from src.training.evaluator import ModelEvaluator


# ---------------------------------------------------------------------------
# ModelEvaluator tests
# ---------------------------------------------------------------------------

def test_compute_metrics_structure_and_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y_pred = np.array([1.5, 2.5, 2.5, 4.5], dtype=np.float32)
    evaluator = ModelEvaluator()
    m = evaluator.compute_metrics(y_true, y_pred)
    assert set(m.keys()) == {"mae", "rmse", "r2", "wape"}
    assert m["mae"] == pytest.approx(0.5, abs=1e-6)
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean((y_true - y_pred) ** 2)), abs=1e-6)
    assert 0.0 <= m["wape"] <= 1.0 + 1e-6


def test_compute_metrics_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    evaluator = ModelEvaluator()
    m = evaluator.compute_metrics(y_true, y_true.copy())
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert m["wape"] == 0.0


def test_evaluate_multi_horizon_structure():
    evaluator = ModelEvaluator()
    y_true = np.random.rand(10, 3).astype(np.float32)
    y_pred = np.random.rand(10, 3).astype(np.float32)
    res = evaluator.evaluate_multi_horizon(y_true, y_pred)
    assert set(res.keys()) == {"15m", "30m", "60m", "overall"}
    for h in ("15m", "30m", "60m", "overall"):
        assert set(res[h].keys()) == {"mae", "rmse", "r2", "wape"}


# ---------------------------------------------------------------------------
# NaivePersistenceBaseline tests
# ---------------------------------------------------------------------------

def test_naive_persistence_shape_and_echo():
    n = 20
    lookback = 12
    n_feat = 7
    X = np.zeros((n, lookback, n_feat), dtype=np.float32)
    # Set last-step delay values
    last_delays = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 4, dtype=np.float32)
    X[:, -1, 0] = last_delays
    X_flat = X.reshape(n, -1)

    Y = np.zeros((n, 3), dtype=np.float32)

    model = NaivePersistenceBaseline(lookback_steps=lookback, n_features=n_feat)
    model.fit(X_flat, Y)
    preds = model.predict(X_flat)

    assert preds.shape == (n, 3)
    assert preds.dtype == np.float32
    # Each horizon column should equal last observed delay
    for h in range(3):
        assert np.allclose(preds[:, h], last_delays, atol=1e-6)


def test_naive_persistence_on_3d_input():
    n = 10
    X = np.zeros((n, 12, 7), dtype=np.float32)
    X[:, -1, 0] = np.arange(n, dtype=np.float32)
    model = NaivePersistenceBaseline()
    preds = model.predict(X)
    assert preds.shape == (n, 3)
    assert np.allclose(preds[:, 0], np.arange(n, dtype=np.float32))


# ---------------------------------------------------------------------------
# TreeBaselineSuite tests
# ---------------------------------------------------------------------------

def test_tree_suite_fit_predict_dummy():
    n = 50
    X = np.random.rand(n, 84).astype(np.float32)
    Y = np.random.rand(n, 3).astype(np.float32)

    for model_type in ("random_forest", "xgboost", "lightgbm"):
        model = TreeBaselineSuite(model_type=model_type, random_state=42)
        model.fit(X, Y)
        preds = model.predict(X)
        assert preds.shape == (n, 3)
        assert preds.dtype == np.float32


def test_tree_suite_all_models():
    suites = TreeBaselineSuite.all_models(random_state=42)
    assert set(suites.keys()) == {"random_forest", "xgboost", "lightgbm"}
    n = 30
    X = np.random.rand(n, 84).astype(np.float32)
    Y = np.random.rand(n, 3).astype(np.float32)
    for name, model in suites.items():
        model.fit(X, Y)
        preds = model.predict(X)
        assert preds.shape == (n, 3)
        assert preds.dtype == np.float32
