"""Evaluation metrics engine for RailPulse AI baseline benchmarks."""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes regression metrics for multi-horizon delay forecasting."""

    def __init__(self):
        self.horizon_names = ["15m", "30m", "60m"]

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute MAE, RMSE, R2, and WAPE for a single set of targets/predictions."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        abs_err = np.abs(y_true - y_pred)
        abs_true = np.abs(y_true)
        denom = abs_true.sum()
        wape = float(abs_err.sum() / denom) if denom > 0 else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "wape": wape,
        }

    def evaluate_multi_horizon(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate metrics independently per horizon column."""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        results: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.horizon_names):
            results[name] = self.compute_metrics(y_true[:, i], y_pred[:, i])
        results["overall"] = self.compute_metrics(y_true, y_pred)
        return results
