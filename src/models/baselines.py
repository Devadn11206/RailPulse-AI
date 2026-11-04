"""Classical machine learning baseline estimators for RailPulse AI Phase 3 Enhanced."""

import logging
from typing import Dict, Optional

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

HORIZON_NAMES = ["15m", "30m", "60m"]
N_HORIZONS = 3


class NaivePersistenceBaseline:
    """Persistence baseline: predicts each horizon as the last observed delay.

    The lookback window X has shape [N, 12, 7] where feature index 0 is delay.
    The last observed delay is X[:, -1, 0]. When flattened to [N, 84], this is
    at column index (12 - 1) * 7 + 0 = 77.
    """

    def __init__(self, lookback_steps: int = 12, n_features: int = 7):
        self.lookback_steps = int(lookback_steps)
        self.n_features = int(n_features)
        self.last_delay_col = (self.lookback_steps - 1) * self.n_features + 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaivePersistenceBaseline":
        """No-op fit for interface compatibility."""
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict last observed delay for all three horizons."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:
            last_delays = X[:, -1, 0]
        elif X.ndim == 2:
            last_delays = X[:, self.last_delay_col]
        else:
            raise ValueError(f"Unexpected X dimension: {X.ndim}")
        return np.tile(last_delays[:, None], (1, 3)).astype(np.float32)


class TreeBaselineSuite:
    """Wrapper around tree-based regressors with per-horizon single-output models.

    Trains a separate regressor per prediction horizon (15m, 30m, 60m) so each
    estimator specializes on its time target instead of a naive multi-output wrapper.
    """

    def __init__(self, model_type: str = "random_forest", random_state: int = 42, **kwargs):
        self.model_type = model_type.lower()
        self.random_state = int(random_state)
        self.kwargs = kwargs
        self.models: Dict[int, object] = {}  # horizon_idx -> fitted model

    def _build_model(self, horizon_idx: int = 0):
        """Build a single-output regressor for a given horizon."""
        if self.model_type == "random_forest":
            params = {
                "random_state": self.random_state,
                "n_estimators": 500,
                "max_depth": 8,
                "n_jobs": -1,
            }
            params.update(self.kwargs)
            return RandomForestRegressor(**params)
        elif self.model_type == "xgboost":
            params = {
                "random_state": self.random_state,
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "n_jobs": -1,
            }
            params.update(self.kwargs)
            return XGBRegressor(**params)
        elif self.model_type == "lightgbm":
            params = {
                "random_state": self.random_state,
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbosity": -1,
                "n_jobs": -1,
            }
            params.update(self.kwargs)
            # LightGBM uses different parameter names for subsampling
            lgb_params = {}
            for k, v in params.items():
                if k == "subsample":
                    lgb_params["bagging_fraction"] = v
                    lgb_params["bagging_freq"] = 1
                elif k == "colsample_bytree":
                    lgb_params["feature_fraction"] = v
                else:
                    lgb_params[k] = v
            return LGBMRegressor(**lgb_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeBaselineSuite":
        """Fit a separate single-output model per horizon.

        Args:
            X: [N, n_features] float32.
            y: [N, 3] float32 (horizons 15m, 30m, 60m).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.models = {}
        for h in range(N_HORIZONS):
            model = self._build_model(horizon_idx=h)
            model.fit(X, y[:, h])
            self.models[h] = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all three horizons using per-horizon models.

        Returns:
            [N, 3] float32 array.
        """
        X = np.asarray(X, dtype=np.float32)
        if not self.models:
            raise RuntimeError("Model not fitted. Call fit() before predict().")
        preds = []
        for h in range(N_HORIZONS):
            preds.append(self.models[h].predict(X))
        return np.column_stack(preds).astype(np.float32)

    @classmethod
    def all_models(cls, random_state: int = 42) -> Dict[str, "TreeBaselineSuite"]:
        return {
            "random_forest": cls("random_forest", random_state=random_state),
            "xgboost": cls("xgboost", random_state=random_state),
            "lightgbm": cls("lightgbm", random_state=random_state),
        }
